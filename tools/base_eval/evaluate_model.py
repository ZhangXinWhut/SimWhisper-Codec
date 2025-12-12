#!/usr/bin/env python3
"""
音频质量评估脚本
计算原始音频与合成音频之间的 PESQ 和 STOI 指标
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from tools.base_eval.evaluator import calculate_pesq_wb_nb, calculate_stoi

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio(file_path: str) -> torch.Tensor:
    """
    加载音频文件并返回单声道张量

    Args:
        file_path: 音频文件路径

    Returns:
        单声道音频张量
    """
    try:
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(file_path)

        # 转换为单声道（如果是立体声，取第一个通道）
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        # 展平成一维张量
        waveform = waveform.squeeze(0)

        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        return waveform

    except Exception as e:
        logger.error(f"加载音频文件失败 {file_path}: {e}")
        return None

def get_audio_files(audio_dir: str) -> list[str]:
    """
    获取音频文件夹中的所有音频文件

    Args:
        audio_dir: 音频文件夹路径

    Returns:
        音频文件路径列表
    """
    audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)

def main():
    parser = argparse.ArgumentParser(description='音频质量评估脚本')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='原始音频文件夹路径')
    parser.add_argument('--synthesized_dir', type=str, required=True,
                       help='合成音频文件夹路径')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='采样率 (默认: 16000)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')

    args = parser.parse_args()

    # 获取音频文件列表
    original_files = get_audio_files(args.original_dir)
    synthesized_files = get_audio_files(args.synthesized_dir)

    if len(original_files) != len(synthesized_files):
        logger.error(f"音频文件数量不匹配: 原始音频 {len(original_files)} 个, 合成音频 {len(synthesized_files)} 个")
        return

    if args.verbose:
        logger.info(f"找到 {len(original_files)} 对音频文件")

    # 存储所有评估结果
    all_pesq_wb = []
    all_pesq_nb = []
    all_stoi = []

    # 使用进度条显示评估进度
    with tqdm(total=len(original_files), desc="评估音频质量") as pbar:
        for original_file, synthesized_file in zip(original_files, synthesized_files):
            # 获取文件名（用于显示）
            original_name = Path(original_file).name
            synthesized_name = Path(synthesized_file).name

            if args.verbose:
                logger.info(f"处理: {original_name} vs {synthesized_name}")

            # 加载音频
            original_audio = load_audio(original_file)
            synthesized_audio = load_audio(synthesized_file)

            if original_audio is None or synthesized_audio is None:
                logger.warning(f"跳过文件对: {original_name} vs {synthesized_name}")
                pbar.update(1)
                continue

            # 确保音频长度相同
            min_length = min(len(original_audio), len(synthesized_audio))
            original_audio = original_audio[:min_length]
            synthesized_audio = synthesized_audio[:min_length]

            # 计算评估指标
            try:
                pesq_wb, pesq_nb = calculate_pesq_wb_nb(original_audio, synthesized_audio, args.sample_rate)
                stoi_score = calculate_stoi(original_audio, synthesized_audio, args.sample_rate)

                all_pesq_wb.append(pesq_wb)
                all_pesq_nb.append(pesq_nb)
                all_stoi.append(stoi_score)

                if args.verbose:
                    logger.info(f"  PESQ-WB: {pesq_wb:.3f}, PESQ-NB: {pesq_nb:.3f}, STOI: {stoi_score:.3f}")

            except Exception as e:
                logger.warning(f"评估失败 {original_name}: {e}")

            pbar.update(1)

    # 计算平均值
    if all_pesq_wb:
        avg_pesq_wb = np.mean(all_pesq_wb)
        avg_pesq_nb = np.mean(all_pesq_nb)
        avg_stoi = np.mean(all_stoi)

        print("\n" + "="*50)
        print("评估结果汇总:")
        print(f"平均 PESQ-WB: {avg_pesq_wb:.3f}")
        print(f"平均 PESQ-NB: {avg_pesq_nb:.3f}")
        print(f"平均 STOI: {avg_stoi:.3f}")
        print("="*50)

        if args.verbose:
            logger.info(f"评估完成，共处理 {len(all_pesq_wb)} 对音频文件")
    else:
        logger.error("没有成功评估任何音频文件")

if __name__ == "__main__":
    main()
