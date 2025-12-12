import logging
import torchaudio
import os
import sys
import glob
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

class DistributedWeightedSamplerWrapper(Sampler):
    """
    分布式加权采样器包装器
    
    核心逻辑：
    1. 在每个epoch开始时，全局生成 (batch_size * steps * world_size) 个加权采样索引
    2. 使用相同的随机种子确保各GPU生成相同的全局索引序列
    3. 每个GPU从全局索引序列中取自己的部分
    """
    def __init__(
        self, 
        dataset, 
        weights,  # 直接传入weights，而不是weighted_sampler
        batch_size: int,
        steps_per_epoch: int,  # 每个GPU的步数
        num_replicas: int, 
        rank: int,
        replacement: bool = True,
        seed: int = 0
    ):
        """
        Args:
            dataset: 数据集
            weights: 样本权重列表或tensor
            batch_size: 每个GPU的batch_size
            steps_per_epoch: 每个GPU的步数（如2000）
            num_replicas: GPU数量
            rank: 当前GPU的rank
            replacement: 是否有放回采样
            seed: 随机种子
        """
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.replacement = replacement
        
        # 转换权重为tensor
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        
        # 计算采样数量
        # 每个GPU需要的样本数
        self.num_samples_per_replica = batch_size * steps_per_epoch
        # 全局总样本数
        self.total_size = self.num_samples_per_replica * num_replicas
        
    def __iter__(self) -> Iterator[int]:
        """
        每次迭代时重新生成加权采样索引
        """
        # 使用 epoch + seed 生成随机数生成器，确保：
        # 1. 不同epoch有不同的采样结果
        # 2. 同一epoch内各GPU生成相同的全局索引序列
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 全局加权采样
        if self.replacement:
            # 有放回采样：生成全局索引
            indices = torch.multinomial(
                self.weights, 
                self.total_size, 
                replacement=True,
                generator=g
            ).tolist()
        else:
            # 无放回采样：需要特殊处理
            indices = torch.multinomial(
                self.weights,
                min(len(self.weights), self.total_size),
                replacement=False,
                generator=g
            ).tolist()
            # padding if needed
            if len(indices) < self.total_size:
                indices += indices[:(self.total_size - len(indices))]
        
        # 当前GPU从全局索引中取自己的部分
        # rank=0: [0, 2, 4, 6, ...]
        # rank=1: [1, 3, 5, 7, ...]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica, \
            f"Expected {self.num_samples_per_replica} samples, got {len(indices)}"
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int) -> None:
        """
        设置当前epoch，影响随机数生成
        """
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