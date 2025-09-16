import logging
import torchaudio
import os
import sys
import glob
import debugpy
import torch
import numpy as np
import re
from typing import List, Dict, Any, Iterator
from torch.utils.data import Sampler

def count_params_by_module(model_name, model):
    logging.info(f"Counting num_parameters of {model_name}:")
    
    param_stats = {}
    total_params = 0  # Count total parameters
    total_requires_grad_params = 0  # Count parameters with requires_grad=True
    total_no_grad_params = 0  # Count parameters with requires_grad=False
    
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in param_stats:
            param_stats[module_name] = {'total': 0, 'requires_grad': 0, 'no_grad': 0}
        
        param_num = param.numel()
        param_stats[module_name]['total'] += param_num
        total_params += param_num
        
        if param.requires_grad:
            param_stats[module_name]['requires_grad'] += param_num
            total_requires_grad_params += param_num
        else:
            param_stats[module_name]['no_grad'] += param_num
            total_no_grad_params += param_num
    
    # Calculate maximum width for each column
    max_module_name_length = max(len(module) for module in param_stats)
    max_param_length = max(len(f"{stats['total'] / 1e6:.2f}M") for stats in param_stats.values())
    
    # Output parameter statistics for each module
    for module, stats in param_stats.items():
        logging.info(f"\t{module:<{max_module_name_length}}: "
                     f"Total: {stats['total'] / 1e6:<{max_param_length}.2f}M, "
                     f"Requires Grad: {stats['requires_grad'] / 1e6:<{max_param_length}.2f}M, "
                     f"No Grad: {stats['no_grad'] / 1e6:<{max_param_length}.2f}M")
    
    # Output total parameter statistics
    logging.info(f"\tTotal parameters: {total_params / 1e6:.2f}M parameters")
    logging.info(f"\tRequires Grad parameters: {total_requires_grad_params / 1e6:.2f}M parameters")
    logging.info(f"\tNo Grad parameters: {total_no_grad_params / 1e6:.2f}M parameters")
    logging.info(f"################################################################")


def load_and_resample_audio(audio_path, target_sample_rate):
    wav, raw_sample_rate = torchaudio.load(audio_path) # (1, T)   tensor 
    if raw_sample_rate != target_sample_rate:   
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate) # tensor 
    return wav.squeeze()

def set_logging(level="INFO"):
    """
    Set global logging configuration.
    Args:
        level: Logging level, can be a string such as "DEBUG"/"INFO"/"WARNING"/"ERROR" or a logging constant
    """
    # If level is a string, convert to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    rank = os.environ.get("RANK", 0)
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format=f"%(asctime)s [RANK {rank}] (%(module)s:%(lineno)d) %(levelname)s : %(message)s",
    )
    
def waiting_for_debug(ip, port):
    rank = os.environ.get("RANK", "0")
    debugpy.listen((ip, port)) # Replace localhost with cluster node IP
    logging.info(f"[rank = {rank}] Waiting for debugger attach...")
    debugpy.wait_for_client()
    logging.info(f"[rank = {rank}] Debugger attached")
    
def load_audio(audio_path, target_sample_rate):
    # Load audio file, wav shape: (channels, time)
    wav, raw_sample_rate = torchaudio.load(audio_path)
    
    # If multi-channel, convert to mono by averaging across channels
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # Average across channels, keep channel dim
    
    # Resample if necessary
    if raw_sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, raw_sample_rate, target_sample_rate)
    
    # Convert to numpy, add channel dimension, then back to tensor with desired shape
    wav = np.expand_dims(wav.squeeze(0).numpy(), axis=1)  # Shape: (time, 1)
    wav = torch.tensor(wav).reshape(1, 1, -1)  # Shape: (1, 1, time)
    
    return wav

def save_audio(audio_outpath, audio_out, sample_rate):
    torchaudio.save(
        audio_outpath, 
        audio_out, 
        sample_rate=sample_rate, 
        encoding='PCM_S', 
        bits_per_sample=16
    )
    logging.info(f"Successfully saved audio at {audio_outpath}")
    
def find_audio_files(input_dir):
    audio_extensions = ['*.flac', '*.mp3', '*.wav']
    audios_input = []
    for ext in audio_extensions:
        audios_input.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    logging.info(f"Found {len(audios_input)} audio files in {input_dir}")
    return sorted(audios_input)

class DistributedWeightedRandomSampler(Sampler):
    def __init__(self, weights: List[float], num_samples_per_epoch: int, 
                 num_replicas: int, rank: int, replacement: bool = True):
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0 # Track the current epoch
        self.num_samples_per_replica = num_samples_per_epoch // self.num_replicas
        self.num_samples_per_epoch = self.num_samples_per_replica * self.num_replicas
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        # Key: re-sample at the beginning of each iteration (epoch)
        # 1. Use the current epoch as seed so all processes generate the same random sequence for the epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 2. Generate a full weighted sampling index list (identical across processes)
        full_indices = torch.multinomial(self.weights, self.num_samples_per_epoch, 
                                         self.replacement, generator=g).tolist()
        
        # 3. Extract the subset of indices that belong to this process
        indices_for_this_replica = full_indices[self.rank : self.num_samples_per_epoch : self.num_replicas]
        
        return iter(indices_for_this_replica)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

def filter_dataset_by_duration(entries: List[Dict[str, Any]], min_duration: float, max_duration: float):
    """
    Filter out manifest entries based on duration.

    Args:
        entries: List of manifest entry dictionaries.
        min_duration: Minimum duration below which entries are removed.
        max_duration: Maximum duration above which entries are removed.

    Returns:
        filtered_entries: List of manifest entries after filtering.
        total_hours: Total duration of original dataset, in hours
        filtered_hours: Total duration of dataset after filtering, in hours
    """
    filtered_entries = []
    total_duration = 0.0
    filtered_duration = 0.0
    for entry in entries:
        duration = entry["duration"]
        total_duration += duration
        if (min_duration and duration < min_duration) or (max_duration and duration > max_duration):
            continue

        filtered_duration += duration
        filtered_entries.append(entry)

    total_hours = total_duration / 3600.0
    filtered_hours = filtered_duration / 3600.0

    return filtered_entries, total_hours, filtered_hours


def read_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Read manifest file in JSONL format.

    Args:
        manifest_path: Path to the manifest file.

    Returns:
        List of manifest entries as dictionaries.
    """
    import json

    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse line in {manifest_path}: {e}")
                    continue

    return entries