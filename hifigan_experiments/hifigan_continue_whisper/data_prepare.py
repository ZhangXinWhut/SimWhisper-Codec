"""
LJSpeech dataset preparation for continuous HiFi-GAN training
Splits the dataset into 80% train, 10% valid, 10% test.

Authors
 * Based on SpeechBrain's LibriTTS preparation
 * Modified for LJSpeech single-speaker English dataset
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
    split_ratio=None,  # Not used since we use existing splits
    seed=1234,
    skip_prep=False,
):
    """
    Prepares the json files for the LJSpeech dataset for continuous HiFi-GAN training.
    Uses the existing train/dev/test folder structure.
    Note: Removed speaker embedding logic.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LJSpeech dataset is stored.
        Should contain train/, dev/, and test/ subfolders.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    split_ratio : list
        Not used since we use existing train/dev/test splits
    seed : int
        Seed value for reproducible behavior
    skip_prep: Bool
        If True, skip preparation.

    Returns
    -------
    None
    """

    if skip_prep:
        return

    # Setting the seed value
    random.seed(seed)

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # Use existing THCHS30 train/dev/test splits
    data_folder = Path(data_folder)
    
    # Collect files from each existing split
    train_files = collect_split_files(data_folder / "train")
    dev_files = collect_split_files(data_folder / "dev") 
    test_files = collect_split_files(data_folder / "test")
    
    logger.info(f"Found {len(train_files)} train files")
    logger.info(f"Found {len(dev_files)} dev files") 
    logger.info(f"Found {len(test_files)} test files")
    
    if len(train_files) == 0:
        raise ValueError(f"No wav files found in {data_folder}/train")
    
    # Creating json files using existing splits
    create_json(train_files, save_json_train, sample_rate)
    create_json(dev_files, save_json_valid, sample_rate)  # dev -> valid
    create_json(test_files, save_json_test, sample_rate)


def collect_split_files(split_folder):
    """
    Collects all wav files from a specific LJSpeech split folder.
    
    Arguments
    ---------
    split_folder : Path
        Path to a specific split folder (train, dev, or test)
        
    Returns
    -------
    wav_list : list
        List of wav file paths in this split
    """
    wav_list = []
    
    if split_folder.exists():
        wav_files = list(split_folder.glob("*.wav"))
        wav_list.extend(wav_files)
        logger.info(f"Found {len(wav_files)} files in {split_folder.name}")
    else:
        logger.warning(f"Split folder {split_folder} does not exist")
    
    return wav_list


def create_json(wav_list, json_file, sample_rate):
    """
    Creates the json file given a list of wav files for LJSpeech dataset.
    Note: Removed speaker_id field as requested.
    
    Arguments
    ---------
    wav_list : list of Path
        The list of wav file paths.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    """

    json_dict = {}

    # Process all the wav files in the list
    for wav_file in tqdm(wav_list, desc=f"Processing {json_file}"):
        try:
            # Read the signal to get duration
            signal, sig_sr = torchaudio.load(wav_file)
            duration = signal.shape[1] / sig_sr

            # Skip very short utterances (less than 1.0 seconds)
            if duration < 1.0:
                logger.warning(f"Skipping {wav_file}: too short ({duration:.2f}s)")
                continue

            if signal.shape[1] == 0:
                logger.warning(f"Skipping {wav_file}: empty audio")
                continue

            if torch.all(signal == 0):
                logger.warning(f"Skipping {wav_file}: silent audio")
                continue

            # Extract utterance ID from filename
            uttid = wav_file.stem

            # Create an entry for the utterance (removed spk_id field)
            json_dict[uttid] = {
                "uttid": uttid,
                "wav": str(wav_file),
                "duration": duration,
                "segment": True if "train" in json_file else False,
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