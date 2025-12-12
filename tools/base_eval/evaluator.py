#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioCodec 评估
"""
import torch
import numpy as np
from pesq import pesq
import logging

logger = logging.getLogger(__name__)

def calculate_pesq(original_audio: torch.Tensor, reconstructed_audio: torch.Tensor,
                  sample_rate: int = 16000, mode: str = 'wb') -> float:
    """
    计算PESQ分数

    Args:
        original_audio: 原始音频 (1D tensor)
        reconstructed_audio: 重建音频 (1D tensor)
        sample_rate: 采样率
        mode: PESQ模式 ('wb' for wideband, 'nb' for narrowband)

    Returns:
        PESQ分数
    """
    try:
        # 确保音频长度相同
        min_length = min(len(original_audio), len(reconstructed_audio))
        original_audio = original_audio[:min_length]
        reconstructed_audio = reconstructed_audio[:min_length]

        # 转换为numpy并归一化
        original_np = original_audio.detach().cpu().numpy()
        reconstructed_np = reconstructed_audio.detach().cpu().numpy()

        # 归一化到[-1, 1]范围
        original_np = np.clip(original_np, -1.0, 1.0)
        reconstructed_np = np.clip(reconstructed_np, -1.0, 1.0)

        # 计算PESQ
        pesq_score = pesq(sample_rate, original_np, reconstructed_np, mode)
        return float(pesq_score)

    except Exception as e:
        logger.warning(f"PESQ({mode})计算失败: {e}")
        return 0.0

def calculate_pesq_wb_nb(original_audio: torch.Tensor, reconstructed_audio: torch.Tensor,
                        sample_rate: int = 16000) -> tuple[float, float]:
    """
    同时计算宽带和窄带PESQ分数

    Args:
        original_audio: 原始音频 (1D tensor)
        reconstructed_audio: 重建音频 (1D tensor)
        sample_rate: 采样率

    Returns:
        (pesq_wb, pesq_nb) 元组
    """
    pesq_wb = calculate_pesq(original_audio, reconstructed_audio, sample_rate, 'wb')
    pesq_nb = calculate_pesq(original_audio, reconstructed_audio, sample_rate, 'nb')
    return pesq_wb, pesq_nb

def calculate_stoi(original_audio: torch.Tensor, reconstructed_audio: torch.Tensor,
                   sample_rate: int = 16000) -> float:
    """
    计算STOI分数

    Args:
        original_audio: 原始音频 (1D tensor)
        reconstructed_audio: 重建音频 (1D tensor)
        sample_rate: 采样率

    Returns:
        STOI分数
    """
    try:
        from pystoi import stoi

        # 确保音频长度相同
        min_length = min(len(original_audio), len(reconstructed_audio))
        original_audio = original_audio[:min_length]
        reconstructed_audio = reconstructed_audio[:min_length]

        # 转换为numpy并归一化
        original_np = original_audio.detach().cpu().numpy()
        reconstructed_np = reconstructed_audio.detach().cpu().numpy()

        # 归一化到[-1, 1]范围
        original_np = np.clip(original_np, -1.0, 1.0)
        reconstructed_np = np.clip(reconstructed_np, -1.0, 1.0)

        # 计算STOI
        stoi_score = stoi(original_np, reconstructed_np, sample_rate, extended=False)
        return float(stoi_score)

    except Exception as e:
        logger.warning(f"STOI计算失败: {e}")
        return 0.0


