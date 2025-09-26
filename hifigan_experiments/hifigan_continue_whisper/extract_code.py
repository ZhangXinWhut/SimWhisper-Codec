"""
Extract continuous features using custom Whisper-based encoder for continuous HiFi-GAN training.
Modified for LJSpeech English single-speaker speech synthesis.

Authors
 * Based on SpeechBrain code, adapted for custom continuous features
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

# Import local custom feature extractor
from local_whisper_ssl import CustomWhisperExtractor

OPT_FILE = "opt_ljspeech_extract_custom_features.pkl"
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
    tensor = tensor.detach().cpu()
    return tensor.numpy()


def skip(splits, save_folder, conf):
    """
    Detects if the LJSpeech custom feature extraction has been already done.
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


def extract_ljspeech_custom(
    data_folder,
    splits,
    whisper_model_name="openai/whisper-small",
    feature_extractor_config=None,
    encoder_config=None,
    save_folder=None,
    sample_rate=16000,
    skip_extract=False,
    freeze_encoder=True,  # Add freeze_encoder parameter
    layer_id=-1,  # Add layer_id parameter
):
    """
    Extract continuous features using custom Whisper-based encoder for LJSpeech dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the prepared json files are stored.
    splits : list
        List of splits to prepare.
    whisper_model_name : str
        Whisper model name for initialization.
    feature_extractor_config : dict
        Configuration for feature extractor.
    encoder_config : dict
        Configuration for encoder.
    save_folder: str
        Path to the folder where the features are stored.
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
        "whisper_model_name": whisper_model_name,
        "feature_extractor_config": feature_extractor_config,
        "encoder_config": encoder_config,
        "sample_rate": sample_rate,
        "freeze_encoder": freeze_encoder,  # Add to config
        "layer_id": layer_id,  # Add layer_id to config
    }

    save_folder = pl.Path(save_folder)
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping custom feature extraction, completed in previous run.")
        return

    # Fetch device
    device = get_device(use_cuda=True)

    save_opt = save_folder / OPT_FILE
    data_folder = pl.Path(data_folder)
    features_folder = save_folder / "custom_features"
    features_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using custom encoder with Whisper initialization: {whisper_model_name}")
    
    # Initialize custom feature extractor
    feature_extractor = CustomWhisperExtractor(
        whisper_model_name=whisper_model_name,
        feature_extractor_config=feature_extractor_config,
        encoder_config=encoder_config,
        device=device,
        freeze_encoder=freeze_encoder,  # Pass freeze_encoder parameter
        layer_id=layer_id  # Pass layer_id parameter
    )
    
    # Remove the manual freeze call since it's now handled in __init__
    # feature_extractor.freeze_encoder()  # This is now done automatically if freeze_encoder=True

    for split in splits:
        dataset_path = data_folder / f"{split}.json"
        logger.info(f"Reading dataset from {dataset_path} ...")
        
        if not dataset_path.exists():
            logger.warning(f"Dataset file {dataset_path} does not exist, skipping...")
            continue
            
        meta_json = json.load(open(dataset_path, encoding="utf-8"))
        
        for key in tqdm(meta_json.keys(), desc=f"Extracting {split}"):
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
                    
                    # Extract continuous features using custom encoder with batch processing
                    features, output_length = feature_extractor.extract_features([audio])
                    features = features.squeeze(0)  # Remove batch dimension [time, feature_dim]
                    
                    # Trim to actual output length
                    actual_length = output_length.item()
                    features = features[:actual_length, :]
                    # Add layer dimension for compatibility with UnitHifiganGenerator
                    # Convert from [T, D] to [T, 1, D] since we only use single layer
                    features = features.unsqueeze(1)  # [T, 1, D]
                    features = np_array(features)
                    
                # Save extracted features
                np.save(features_folder / f"{key}.npy", features)
                
            except Exception as e:
                logger.error(f"Error processing {wav}: {e}")
                continue

    logger.info("Custom feature extraction completed.")
    save_pkl(conf, save_opt)


# Alias for compatibility
extract_ljspeech_whisper = extract_ljspeech_custom