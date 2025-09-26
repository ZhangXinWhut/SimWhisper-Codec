#!/usr/bin/env python3
"""Recipe for training a hifi-gan vocoder on HuBERT continuous representations with LJSpeech dataset.
For more details about hifi-gan: https://arxiv.org/pdf/2010.05646.pdf
For more details about speech synthesis using self-supervised representations: https://arxiv.org/pdf/2104.00355.pdf

To run this recipe, do the following:
> python train_ljspeech.py hparams/train_ljspeech.yaml --data_folder=/path/to/LJSpeech

Copied and modified from:
https://github.com/speechbrain/speechbrain/blob/develop/recipes/LJSpeech/TTS/vocoder/hifigan_discrete/train.py

Authors
 * Modified for LJSpeech dataset with HuBERT continuous features
"""

import copy
import pathlib as pl
import random
import sys

import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import scalarize


class ContinuousHifiGanBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator
        for synthesized waveforms and real waveforms.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        Generator and Discriminator outputs
        """
        batch = batch.to(self.device)

        x, _ = batch.features  # continuous features from HuBERT encoder
        y, _ = batch.sig

        # generate synthesized waveforms using continuous features (no speaker embedding)
        y_g_hat, (log_dur_pred, log_dur) = self.modules.generator(x)
        y_g_hat = y_g_hat[:, :, : y.size(2)]

        # get scores and features from discriminator for real and synthesized waveforms
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y)

        return (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs."""
        batch = batch.to(self.device)

        x, _ = batch.features
        y, _ = batch.sig

        # Hold on to the batch for the inference sample
        self.last_batch = (x, y)

        (
            y_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        ) = predictions

        loss_g = self.hparams.generator_loss(
            stage,
            y_hat,
            y,
            scores_fake,
            feats_fake,
            feats_real,
            log_dur_pred,
            log_dur,
        )

        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)

        return loss

    def fit_batch(self, batch):
        """Fits a single batch."""
        batch = batch.to(self.device)
        y, _ = batch.sig

        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        ) = outputs
        
        # calculate discriminator loss with the latest updated generator
        loss_d = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "D_loss"
        ]
        # First train the discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # calculate generator loss with the latest updated discriminator
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat)
        scores_real, feats_real = self.modules.discriminator(y)
        outputs = (
            y_g_hat,
            scores_fake,
            feats_fake,
            scores_real,
            feats_real,
            log_dur_pred,
            log_dur,
        )
        loss_g = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "G_loss"
        ]
        # Then train the generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch."""
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``."""
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers."""
        if self.opt_class is not None:
            (
                opt_g_class,
                opt_d_class,
                sch_g_class,
                sch_d_class,
            ) = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(
                self.modules.discriminator.parameters()
            )
            self.optimizers_dict = {
                "optimizer_g": self.optimizer_g,
                "optimizer_d": self.optimizer_d,
            }

            self.scheduler_g = sch_g_class(self.optimizer_g)
            self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_g", self.scheduler_g
                )
                self.checkpointer.add_recoverable(
                    "scheduler_d", self.scheduler_d
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]

            stats = {
                **self.last_loss_stats[sb.Stage.VALID],
            }

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=stats,
            )

            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=stats,
                )

            # Save the current checkpoint and delete previous checkpoints
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta=epoch_metadata,
                    end_of_epoch=True,
                    min_keys=["loss"],
                    ckpt_predicate=(
                        (
                            lambda ckpt: (
                                ckpt.meta["epoch"]
                                % self.hparams.keep_checkpoint_interval
                                != 0
                            )
                        )
                        if self.hparams.keep_checkpoint_interval is not None
                        else None
                    ),
                )

            # Generate validation audio sample every epoch (not just at intervals)
            self.run_inference_sample("Valid", epoch)

        # We also write statistics about test data to stdout and to the TensorboardLogger
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=self.last_loss_stats[sb.Stage.TEST],
                )
            self.run_inference_sample("Test", epoch)

    def run_inference_sample(self, name, epoch):
        """Produces a sample in inference mode."""
        with torch.no_grad():
            if self.last_batch is None:
                return
            x, y = self.last_batch

            # Preparing model for inference by removing weight norm
            inference_generator = copy.deepcopy(self.hparams.generator)
            inference_generator.remove_weight_norm()
            
            # Use inference method which expects [batch, time, 1, dim] format
            sig_out = inference_generator.inference(x)
            spec_out = self.hparams.mel_spectogram(
                audio=sig_out.squeeze(0).cpu()
            )
            
        if self.hparams.use_tensorboard:
            self.tensorboard_logger.log_audio(
                f"{name}/audio_target", y.squeeze(0), self.hparams.sample_rate
            )
            self.tensorboard_logger.log_audio(
                f"{name}/audio_pred",
                sig_out.squeeze(0),
                self.hparams.sample_rate,
            )
            self.tensorboard_logger.log_figure(f"{name}/mel_target", x)
            self.tensorboard_logger.log_figure(f"{name}/mel_pred", spec_out)
        else:
            # folder name is the current epoch for validation and "test" for test
            folder = (
                self.hparams.epoch_counter.current
                if name == "Valid"
                else "test"
            )
            self.save_audio("target", y.squeeze(0), folder)
            self.save_audio("synthesized", sig_out.squeeze(0), folder)

    def save_audio(self, name, data, epoch):
        """Saves a single wav file."""
        target_path = pl.Path(self.hparams.progress_sample_path) / str(epoch)
        target_path.mkdir(parents=True, exist_ok=True)
        file_name = target_path / f"{name}.wav"
        torchaudio.save(file_name.as_posix(), data.cpu(), 16000)


def sample_interval(seqs, segment_size):
    """This function sample an interval of audio and continuous features according to segment size."""
    N = max([v.shape[-1] for v in seqs])
    seq_len = segment_size if segment_size > 0 else N
    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm
    
    if interval_end <= interval_start:
        start_step = interval_start
    else:
        start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]

    return new_seqs


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class."""
    segment_size = hparams["segment_size"]
    feature_hop_size = hparams["feature_hop_size"]
    features_folder = pl.Path(hparams["continuous_features_folder"])

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("id", "wav", "segment")
    @sb.utils.data_pipeline.provides("features", "sig")
    def audio_pipeline(utt_id, wav, segment):
        info = torchaudio.info(wav)
        audio = sb.dataio.dataio.read_audio(wav)
        
        # Resample to 16kHz if necessary (LJSpeech is 22050Hz, HuBERT needs 16kHz)
        if info.sample_rate != hparams["sample_rate"]:
            audio = torchaudio.transforms.Resample(
                info.sample_rate,
                hparams["sample_rate"],
            )(audio)

        # Load continuous features [time, 1, dim]
        features = np.load(features_folder / f"{utt_id}.npy")
        features = torch.FloatTensor(features)

        # Trim end of audio to match features
        feature_length = min(audio.shape[0] // feature_hop_size, features.shape[0])
        features = features[:feature_length]
        audio = audio[: feature_length * feature_hop_size]

        # Ensure minimum length
        while audio.shape[0] < segment_size:
            audio = torch.hstack([audio, audio])
            features = torch.vstack([features, features])
        
        audio = audio.unsqueeze(0)

        if segment:
            features = features.permute(1, 2, 0)
            # For continuous features with shape [T, 1, D], sample along time dimension
            audio, features = sample_interval([audio, features], segment_size)
            features = features.permute(2, 0, 1)

        return features, audio

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "features", "sig"],
        )

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # LJSpeech data preparation
    from data_prepare import prepare_ljspeech

    sb.utils.distributed.run_on_main(
        prepare_ljspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Feature extraction using HuBERT
    from extract_continuous_features import extract_ljspeech_continuous_features

    sb.utils.distributed.run_on_main(
        extract_ljspeech_continuous_features,
        kwargs={
            "data_folder": hparams["save_folder"],  # JSON files location
            "splits": hparams["splits"],
            "hubert_source": hparams["hubert_model_name"],
            "target_layer": hparams["target_layer"],
            "save_folder": hparams["save_folder"],
            "sample_rate": hparams["sample_rate"],
            "skip_extract": hparams["skip_extract"],
        },
    )

    datasets = dataio_prepare(hparams)

    # Brain class initialization
    continuous_hifi_gan_brain = ContinuousHifiGanBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["use_tensorboard"]:
        continuous_hifi_gan_brain.tensorboard_logger = (
            sb.utils.train_logger.TensorboardLogger(
                save_dir=hparams["output_folder"] + "/tensorboard"
            )
        )

    # Training
    continuous_hifi_gan_brain.fit(
        continuous_hifi_gan_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        continuous_hifi_gan_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )