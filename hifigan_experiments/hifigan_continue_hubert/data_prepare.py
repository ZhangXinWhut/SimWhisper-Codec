"""
LJSpeech dataset preparation for continuous HiFi-GAN training with HuBERT features
Uses fixed data split for reproducible experiments across different models.

Authors
 * Based on SpeechBrain's LibriTTS preparation
 * Modified for LJSpeech single-speaker English dataset with HuBERT features
"""

import json
import os
import random
import logging
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_ljspeech(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=1234,
    skip_prep=False,
):
    """
    Prepares the json files for the LJSpeech dataset for continuous HiFi-GAN training with HuBERT.
    Uses fixed data split for reproducible experiments across different models:
    - First 10000 files: training set
    - Files 10000-10900: validation set  
    - Remaining files: test set

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LJSpeech dataset is stored.
        Should contain wavs/ subfolder and metadata.csv.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    train_ratio : float
        Ratio of data to use for training (kept for compatibility, not used)
    valid_ratio : float
        Ratio of data to use for validation (kept for compatibility, not used)
    test_ratio : float
        Ratio of data to use for testing (kept for compatibility, not used)
    seed : int
        Seed value (kept for compatibility, not used in fixed split)
    skip_prep: Bool
        If True, skip preparation.

    Returns
    -------
    None
    """

    if skip_prep:
        return

    # Verify ratios sum to 1.0
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train/valid/test ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}")

    # Setting the seed value
    random.seed(seed)

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # Collect all wav files from LJSpeech
    data_folder = Path(data_folder)
    wavs_folder = data_folder / "wavs"
    
    if not wavs_folder.exists():
        raise ValueError(f"WAV folder not found: {wavs_folder}")
    
    # Get all wav files and sort them for reproducible splitting
    wav_files = sorted(list(wavs_folder.glob("*.wav")))
    
    logger.info(f"Found {len(wav_files)} total wav files")
    
    if len(wav_files) == 0:
        raise ValueError(f"No wav files found in {wavs_folder}")
    
    # Fixed split indices for reproducible experiments across different models
    # Total: 13100 files -> Train: 10000, Valid: 900, Test: 2200
    total_files = len(wav_files)
    train_end = 10000
    valid_end = 10900
    
    if total_files < valid_end:
        logger.warning(f"Dataset has only {total_files} files, but expecting at least {valid_end}")
        # Fall back to proportional split if dataset is smaller
        train_end = int(total_files * 0.8)
        valid_end = train_end + int(total_files * 0.1)
    
    # Split the files with fixed indices (no shuffling for reproducibility)
    train_files = wav_files[:train_end]
    valid_files = wav_files[train_end:valid_end]
    test_files = wav_files[valid_end:]
    
    logger.info(f"Split: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test files")
    
    # Creating json files
    create_json(train_files, save_json_train, sample_rate, is_train=True)
    create_json(valid_files, save_json_valid, sample_rate, is_train=False)
    create_json(test_files, save_json_test, sample_rate, is_train=False)


def create_json(wav_list, json_file, sample_rate, is_train=False):
    """
    Creates the json file given a list of wav files for LJSpeech dataset.
    
    Arguments
    ---------
    wav_list : list of Path
        The list of wav file paths.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    is_train : bool
        Whether this is the training set (affects segmentation setting)
    """

    json_dict = {}

    # Process all the wav files in the list
    for wav_file in tqdm(wav_list, desc=f"Processing {json_file}"):
        try:
            # Read the signal to get duration
            signal, sig_sr = torchaudio.load(wav_file)
            duration = signal.shape[1] / sig_sr

            # Skip very short utterances (less than 0.5 seconds)
            if duration < 0.5:
                logger.warning(f"Skipping {wav_file}: too short ({duration:.2f}s)")
                continue

            if signal.shape[1] == 0:
                logger.warning(f"Skipping {wav_file}: empty audio")
                continue

            if torch.all(signal == 0):
                logger.warning(f"Skipping {wav_file}: silent audio")
                continue

            # Extract utterance ID from filename
            # LJSpeech format: LJ001-0001.wav -> utterance ID: LJ001-0001
            uttid = wav_file.stem

            # For LJSpeech (single speaker), we can set a constant speaker ID
            spk_id = "LJ"  # Single speaker dataset

            # Create an entry for the utterance
            json_dict[uttid] = {
                "uttid": uttid,
                "wav": str(wav_file),
                "duration": duration,
                "spk_id": spk_id,
                "segment": is_train,  # Only segment during training
            }

        except Exception as e:
            logger.warning(f"Error processing {wav_file}: {e}")
            continue

    # Write the dictionary to the json file
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2, ensure_ascii=False)

    logger.info(f"{json_file} successfully created with {len(json_dict)} entries!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : tuple
        Set of filenames to check for existence.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare LJSpeech dataset for HuBERT HiFi-GAN training")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to LJSpeech dataset folder")
    parser.add_argument("--save_folder", type=str, required=True,
                        help="Folder to save the JSON files")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate for the dataset (16000 for HuBERT)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio for training set (not used in fixed split)")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Ratio for validation set (not used in fixed split)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio for test set (not used in fixed split)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed (not used in fixed split)")
    
    args = parser.parse_args()
    
    # Create save folder if it doesn't exist
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Define output JSON paths
    train_json = os.path.join(args.save_folder, "train.json")
    valid_json = os.path.join(args.save_folder, "valid.json")
    test_json = os.path.join(args.save_folder, "test.json")
    
    # Prepare the dataset
    prepare_ljspeech(
        data_folder=args.data_folder,
        save_json_train=train_json,
        save_json_valid=valid_json,
        save_json_test=test_json,
        sample_rate=args.sample_rate,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        skip_prep=False,
    )