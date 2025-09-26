"""
Extract continuous HuBERT features for HiFi-GAN training.
Modified for LJSpeech English single-speaker speech synthesis with facebook/hubert-base-ls960.
Saves continuous features instead of discrete tokens.

Authors
 * Based on SpeechBrain code, adapted for continuous HuBERT features
"""

import json
import logging
import pathlib as pl
import os

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.dataio import load_pkl, save_pkl
from speechbrain.utils.logger import get_logger

# Import continuous HuBERT module
from continuous_hubert_ssl import ContinuousHubertSSL

OPT_FILE = "opt_ljspeech_extract_continuous_features.pkl"
TRAIN_JSON = "train.json"
VALID_JSON = "valid.json"
TEST_JSON = "test.json"


def setup_logger():
    """Set up a logger with a log format and logging level."""
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = get_logger(__name__)
    return logger


def get_device(use_cuda):
    """Determine and return the appropriate device for computation."""
    use_cuda = use_cuda and torch.cuda.is_available()
    print("\n" + "=" * 30)
    print("USE_CUDA SET TO: {}".format(use_cuda))
    print("CUDA AVAILABLE?: {}".format(torch.cuda.is_available()))
    print("=" * 30 + "\n")
    return torch.device("cuda" if use_cuda else "cpu")


def np_array(tensor):
    """Convert a Pytorch tensor to a Numpy array."""
    tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    return tensor.numpy()


def skip(splits, save_folder, conf):
    """
    Detects if the LJSpeech continuous feature extraction has been already done.

    Arguments
    ---------
    splits : list
        List of splits to check for existence.
    save_folder : str
        Folder containing prepared data.
    conf : dict
        The loaded configuration options.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    split_files = {
        "train": TRAIN_JSON,
        "valid": VALID_JSON,
        "test": TEST_JSON,
    }

    for split in splits:
        if not (save_folder / split_files[split]).exists():
            skip = False

    # Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(save_opt.as_posix())
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False
    return skip


def extract_ljspeech_continuous_features(
    data_folder,
    splits,
    hubert_source="facebook/hubert-base-ls960",
    target_layer=16,
    save_folder=None,
    sample_rate=16000,
    skip_extract=False,
):
    """
    Extract continuous speech features using SpeechBrain HuBERT encoder for LJSpeech dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the prepared json files are stored.
    splits : list
        List of splits to prepare.
    hubert_source : str
        HuBERT model source (e.g., 'facebook/hubert-base-ls960').
    target_layer : int
        Which layer to extract features from.
    save_folder: str
        Path to the folder where the speech features are stored.
    sample_rate: int
        Dataset sample rate
    skip_extract: Bool
        If True, skip extraction.

    Returns
    -------
    None
    """
    logger = setup_logger()

    if skip_extract:
        return

    # Create configuration for easily skipping feature extraction stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "save_folder": save_folder,
        "hubert_source": hubert_source,
        "target_layer": target_layer,
    }

    save_folder = pl.Path(save_folder)
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping continuous feature extraction, completed in previous run.")
        return

    # Fetch device
    device = get_device(use_cuda=True)

    save_opt = save_folder / OPT_FILE
    data_folder = pl.Path(data_folder)
    save_path = save_folder / "savedir"
    features_folder = save_folder / "continuous_features"
    features_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using SpeechBrain HuBERT model: {hubert_source} for continuous feature extraction...")
    
    # Initialize continuous HuBERT encoder
    continuous_encoder = ContinuousHubertSSL(
        save_path=save_path.as_posix(),
        hubert_source=hubert_source,
        layer_id=target_layer,
        device=device
    )

    for split in splits:
        dataset_path = data_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        
        if not dataset_path.exists():
            logger.warning(f"Dataset file {dataset_path} does not exist, skipping...")
            continue
            
        meta_json = json.load(open(dataset_path, encoding="utf-8"))
        
        for key in tqdm(meta_json.keys(), desc=f"Extracting continuous features for {split}"):
            item = meta_json[key]
            wav = item["wav"]
            
            try:
                with torch.no_grad():
                    info = torchaudio.info(wav)
                    audio = sb.dataio.dataio.read_audio(wav)
                    
                    # Resample if necessary
                    if info.sample_rate != sample_rate:
                        audio = torchaudio.transforms.Resample(
                            info.sample_rate,
                            sample_rate,
                        )(audio)
                    
                    audio = audio.unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Extract continuous features
                    features = continuous_encoder(audio)  # [batch, time, dim]
                    features = features.squeeze(0)  # Remove batch dimension: [time, dim]
                    
                    # Reshape to [time, 1, dim] for single layer (layer_nums=1)
                    features = features.unsqueeze(1)  # [time, 1, dim]
                    features = np_array(features)
                    
                # Save extracted continuous features
                np.save(features_folder / f"{key}.npy", features)
                
            except Exception as e:
                logger.error(f"Error processing {wav}: {e}")
                continue

    logger.info("Continuous feature extraction completed.")
    save_pkl(conf, save_opt)


# Alias function for compatibility
extract_ljspeech = extract_ljspeech_continuous_features