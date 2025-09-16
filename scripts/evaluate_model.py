#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AudioCodec 模型评估脚本

功能：
- 加载指定模型权重进行推理
- 计算平均PESQ分数
- 记录最高三个PESQ分数的音频文件名/路径
- 默认使用latest模型评估
- 使用YAML配置文件中的测试数据集
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import torchaudio

# 添加项目根目录到模块路径
current_file = os.path.abspath(__file__)
project_root = current_file
for _ in range(2):
    project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from audiocodec.model import AudioCodec
from audiocodec.trainer.dataset import get_audio_dataloader
from audiocodec.trainer.evaluator import calculate_pesq_wb_nb, calculate_stoi
from audiocodec.trainer.trainer import AudioCodecTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# 定义评估结果结构
EvaluationResult = namedtuple('EvaluationResult', [
    'pesq_wb', 'pesq_nb', 'stoi_score',  # 各项指标分数
    'audio_filepath', 'dataset_name',    # 文件信息
    'reconstructed_audio', 'original_audio'  # 音频数据
])

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config_path: str, model_tag: str = "latest", device: str = "auto", training_stage: str = None, save_all_audio: bool = False):
        """
        初始化评估器

        Args:
            config_path: YAML配置文件路径
            model_tag: 模型标签 (latest, best_model, 或具体epoch标签)
            device: 设备 ('auto', 'cpu', 'cuda', 'cuda:0'等)
            training_stage: 训练阶段 ('pretrain', 'posttrain', 或 None自动检测)
            save_all_audio: 是否保存所有生成的音频文件
        """
        self.config_path = config_path
        self.model_tag = model_tag
        self.device = self._setup_device(device)
        self.save_all_audio = save_all_audio

        # 加载配置
        self.config = self._load_config()

        # 自动检测训练阶段
        self.training_stage = training_stage or self._detect_training_stage()

        # 初始化模型
        self.model = self._initialize_model()

        # 加载模型权重
        self._load_model_weights()

        # 创建测试数据加载器
        self.test_loader = self._create_test_loader()

        logger.info("✅ 模型评估器初始化完成")
        logger.info(f"  📋 配置文件: {config_path}")
        logger.info(f"  🏷️  模型标签: {model_tag}")
        logger.info(f"  🎯 训练阶段: {self.training_stage}")
        logger.info(f"  💻 设备: {self.device}")
        logger.info(f"  📊 测试数据集大小: {len(self.test_loader.dataset)}")

    def _detect_training_stage(self) -> str:
        """自动检测训练阶段"""
        # 检查配置文件中是否有pretrained_checkpoint
        training_config = self.config.get('training', {})

        # 如果有pretrained_checkpoint，则是后训练阶段
        if 'pretrained_checkpoint' in training_config:
            logger.info("🔍 检测到预训练检查点配置，判断为后训练模型")
            return 'posttrain'
        else:
            logger.info("🔍 未检测到预训练检查点配置，判断为预训练模型")
            return 'pretrain'

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise

    def _initialize_model(self) -> AudioCodec:
        """初始化模型"""
        try:
            model_config = self.config.get('model', {})

            # AudioCodec 构造函数需要 generator_params 参数
            model = AudioCodec(
                generator_params=model_config,
                initialize_whisper=False  # 评估时不需要初始化 Whisper 权重，会从检查点加载
            )

            model.to(self.device)
            model.eval()

            logger.info("✅ 模型初始化成功")
            return model

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise

    def _load_model_weights(self):
        """加载模型权重"""
        try:
            # 确定检查点根目录 - 基于配置文件和训练阶段
            config_checkpoint_dir = self.config.get('checkpoint_dir')

            if config_checkpoint_dir:
                # 如果配置文件中有checkpoint_dir，使用它
                checkpoint_root = config_checkpoint_dir
                logger.info(f"📂 使用配置文件中的检查点目录: {checkpoint_root}")
            else:
                # 如果没有，根据训练阶段自动确定
                if self.training_stage == 'posttrain':
                    checkpoint_root = 'outputs/posttrain/checkpoints'
                    logger.info(f"📂 后训练阶段，使用默认检查点目录: {checkpoint_root}")
                else:
                    checkpoint_root = 'outputs/pretrain/checkpoints'
                    logger.info(f"📂 预训练阶段，使用默认检查点目录: {checkpoint_root}")

            # 加载模型权重
            success = AudioCodecTrainer.load_model_weights(
                self.model,
                checkpoint_root,
                tag=self.model_tag
            )

            if success:
                logger.info(f"✅ 模型权重加载成功: {checkpoint_root} (tag: {self.model_tag})")
            else:
                logger.warning(f"⚠️ 模型权重加载可能不完整: {checkpoint_root} (tag: {self.model_tag})")

        except Exception as e:
            logger.error(f"❌ 模型权重加载失败: {e}")
            raise

    def _create_test_loader(self):
        """创建测试数据加载器"""
        try:
            test_loader = get_audio_dataloader(
                self.config,
                stage='test',
                world_size=1,
                rank=0
            )
            logger.info("✅ 测试数据加载器创建成功")
            return test_loader
        except Exception as e:
            logger.error(f"❌ 测试数据加载器创建失败: {e}")
            raise

    def evaluate_model(self, max_samples: Optional[int] = None, output_dir: str = "evaluation_results") -> Dict:
        """
        评估模型性能

        Args:
            max_samples: 最大评估样本数，None表示全部
            output_dir: 输出目录路径

        Returns:
            评估结果字典
        """
        logger.info("🚀 开始模型评估...")
        
        # 如果需要保存所有音频，创建输出目录
        all_audio_save_dir = None
        if self.save_all_audio:
            all_audio_save_dir = Path(output_dir) / "all_audio_samples"
            all_audio_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🎵 将保存所有生成的音频到: {all_audio_save_dir}")

        pesq_scores = []
        pesq_results = []  # 保存所有PESQ结果用于排序

        # 计算总样本数用于进度条
        if max_samples:
            total_samples = max_samples
        else:
            # 计算数据集的总样本数
            total_samples = len(self.test_loader.dataset)

        progress_bar = tqdm(
            total=total_samples,
            desc="评估进度",
            unit="样本",
            leave=True,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        sample_count = 0

        # 存储各项指标的列表
        pesq_wb_scores = []
        pesq_nb_scores = []
        stoi_scores = []
        eval_results = []  # 存储评估结果用于后续排序

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if max_samples and sample_count >= max_samples:
                    break

                try:
                    # 准备数据
                    audio_list = batch['audio_list']
                    audio_lengths = batch['audio_lens'].to(self.device)
                    audio_filepaths = batch['audio_filepaths']
                    dataset_names = batch['dataset_names']

                    # 模型推理 - 使用检测到的训练阶段
                    outputs = self.model(
                        audio_list=audio_list,
                        audio_lengths=audio_lengths,
                        training_stage=self.training_stage
                    )

                    # 获取重建音频
                    if 'reconstructed_audio' not in outputs:
                        logger.warning(f"批次 {batch_idx}: 未找到重建音频，跳过")
                        continue

                    reconstructed_audio = outputs['reconstructed_audio']

                    # 计算每个样本的PESQ分数
                    for i in range(len(audio_list)):
                        original_audio = audio_list[i]
                        reconstructed = reconstructed_audio[i, 0]  # 假设是 (batch, 1, seq_len) 格式

                        # 确保长度匹配
                        min_length = min(len(original_audio), len(reconstructed))
                        original_audio = original_audio[:min_length]
                        reconstructed = reconstructed[:min_length]

                        # 计算各项指标
                        try:
                            # 计算PESQ-WB和PESQ-NB
                            pesq_wb, pesq_nb = calculate_pesq_wb_nb(
                                original_audio=original_audio,
                                reconstructed_audio=reconstructed,
                                sample_rate=self.config.get('sample_rate', 16000)
                            )

                            # 计算STOI
                            stoi_score = calculate_stoi(
                                original_audio=original_audio,
                                reconstructed_audio=reconstructed,
                                sample_rate=self.config.get('sample_rate', 16000)
                            )

                            # 添加到各个指标列表
                            pesq_wb_scores.append(pesq_wb)
                            pesq_nb_scores.append(pesq_nb)
                            stoi_scores.append(stoi_score)

                            # 保存评估结果用于后续排序（包含音频数据）
                            result = EvaluationResult(
                                pesq_wb=pesq_wb,
                                pesq_nb=pesq_nb,
                                stoi_score=stoi_score,
                                audio_filepath=audio_filepaths[i],
                                dataset_name=dataset_names[i],
                                reconstructed_audio=reconstructed.detach().cpu(),  # 保存重建音频
                                original_audio=original_audio.detach().cpu()       # 保存原始音频
                            )
                            eval_results.append(result)

                            # 如果启用了保存所有音频功能，保存当前音频
                            if self.save_all_audio and all_audio_save_dir is not None:
                                self._save_single_audio_sample(result, sample_count, all_audio_save_dir)

                            sample_count += 1

                            # 每处理一个样本就更新进度条
                            progress_bar.update(1)
                            if pesq_wb_scores:
                                avg_pesq_wb = np.mean(pesq_wb_scores)
                                avg_pesq_nb = np.mean(pesq_nb_scores)
                                avg_stoi = np.mean(stoi_scores)
                                postfix_dict = {
                                    'PESQ-WB': f'{avg_pesq_wb:.3f}',
                                    'PESQ-NB': f'{avg_pesq_nb:.3f}',
                                    'STOI': f'{avg_stoi:.3f}',
                                    '样本数': f'{sample_count}'
                                }
                                if self.save_all_audio:
                                    postfix_dict['音频已保存'] = '✓'
                                progress_bar.set_postfix(postfix_dict)

                        except Exception as e:
                            logger.warning(f"样本 {i} PESQ计算失败: {e}")
                            # 即使PESQ计算失败，也要更新进度条以反映处理进度
                            progress_bar.update(1)
                            continue

                except Exception as e:
                    logger.warning(f"批次 {batch_idx} 处理失败: {e}")
                    # 批次失败时，需要更新对应数量的进度条
                    progress_bar.update(len(audio_list))
                    continue

        progress_bar.close()

        # 计算最终统计结果
        if not pesq_wb_scores:
            logger.error("❌ 没有成功计算任何评估指标")
            return {}

        # 计算各项指标的统计信息
        avg_pesq_wb = np.mean(pesq_wb_scores)
        std_pesq_wb = np.std(pesq_wb_scores)
        min_pesq_wb = np.min(pesq_wb_scores)
        max_pesq_wb = np.max(pesq_wb_scores)

        avg_pesq_nb = np.mean(pesq_nb_scores)
        std_pesq_nb = np.std(pesq_nb_scores)

        avg_stoi = np.mean(stoi_scores)
        std_stoi = np.std(stoi_scores)

        # 获取前5个最高PESQ-WB分数的样本
        top_5_results = sorted(eval_results, key=lambda x: x.pesq_wb, reverse=True)[:5]

        # 打印评估结果
        logger.info("\n" + "="*60)
        logger.info("📊 模型评估结果")
        logger.info("="*60)
        logger.info(f"🎯 平均PESQ-WB: {avg_pesq_wb:.4f} (±{std_pesq_wb:.4f})")
        logger.info(f"🎯 平均PESQ-NB: {avg_pesq_nb:.4f} (±{std_pesq_nb:.4f})")
        logger.info(f"🎯 平均STOI: {avg_stoi:.4f} (±{std_stoi:.4f})")
        logger.info(f"📊 评估样本数: {len(pesq_wb_scores)}")
        logger.info(f"🔺 最高PESQ-WB: {max_pesq_wb:.4f}")
        logger.info(f"🔻 最低PESQ-WB: {min_pesq_wb:.4f}")

        logger.info("\n🏆 前5个最高PESQ-WB分数的音频文件:")
        for i, result in enumerate(top_5_results, 1):
            logger.info(f"  {i}. PESQ-WB: {result.pesq_wb:.4f}, PESQ-NB: {result.pesq_nb:.4f}, STOI: {result.stoi_score:.4f}")
            logger.info(f"     文件: {result.audio_filepath}")
            logger.info(f"     数据集: {result.dataset_name}")

        # 保存前5个最高PESQ-WB分数的音频文件
        self._save_top5_audio_files(top_5_results)
        
        # 如果启用了保存所有音频功能，记录保存的音频文件数量
        if self.save_all_audio and all_audio_save_dir is not None:
            logger.info(f"\n📁 所有音频文件已保存到: {all_audio_save_dir}")
            logger.info(f"🎵 共保存了 {len(pesq_wb_scores)} 个音频样本（每个样本包含原始音频和重建音频）")

        # 保存详细结果到文件
        self._save_evaluation_results(eval_results, avg_pesq_wb, std_pesq_wb, avg_pesq_nb, std_pesq_nb, avg_stoi, std_stoi)

        return {
            'avg_pesq_wb': avg_pesq_wb,
            'std_pesq_wb': std_pesq_wb,
            'avg_pesq_nb': avg_pesq_nb,
            'std_pesq_nb': std_pesq_nb,
            'avg_stoi': avg_stoi,
            'std_stoi': std_stoi,
            'min_pesq_wb': min_pesq_wb,
            'max_pesq_wb': max_pesq_wb,
            'num_samples': len(pesq_wb_scores),
            'top_5_results': top_5_results,
            'all_pesq_wb_scores': pesq_wb_scores,
            'all_pesq_nb_scores': pesq_nb_scores,
            'all_stoi_scores': stoi_scores
        }

    def _save_single_audio_sample(self, result: EvaluationResult, sample_count: int, save_dir: Path):
        """保存单个音频样本（原始音频和重建音频）"""
        try:
            sample_rate = self.config.get('sample_rate', 16000)
            
            # 从文件路径中提取文件名（不含扩展名）
            original_filename = Path(result.audio_filepath).stem
            dataset_name = result.dataset_name
            
            # 创建数据集子目录
            dataset_dir = save_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # 创建保存的文件名（使用样本计数确保唯一性）
            original_file = dataset_dir / f"sample_{sample_count:06d}_{original_filename}_original.wav"
            reconstructed_file = dataset_dir / f"sample_{sample_count:06d}_{original_filename}_reconstructed.wav"
            
            # 保存原始音频
            torchaudio.save(str(original_file), result.original_audio.unsqueeze(0), sample_rate)
            
            # 保存重建音频
            torchaudio.save(str(reconstructed_file), result.reconstructed_audio.unsqueeze(0), sample_rate)
            
            # 创建元数据文件记录评估指标
            metadata_file = dataset_dir / f"sample_{sample_count:06d}_{original_filename}_metadata.json"
            metadata = {
                "sample_id": sample_count,
                "original_filename": original_filename,
                "dataset_name": dataset_name,
                "audio_filepath": result.audio_filepath,
                "pesq_wb": float(result.pesq_wb),
                "pesq_nb": float(result.pesq_nb),
                "stoi_score": float(result.stoi_score),
                "saved_files": {
                    "original": str(original_file.name),
                    "reconstructed": str(reconstructed_file.name)
                }
            }
            
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"保存音频样本 {sample_count} 失败: {e}")

    def _save_top5_audio_files(self, top_5_results: List[EvaluationResult]):
        """保存前5个最高PESQ-WB分数的音频文件"""
        try:
            # 创建音频保存目录
            audio_save_dir = Path("evaluation_results/audio_samples")
            audio_save_dir.mkdir(parents=True, exist_ok=True)

            sample_rate = self.config.get('sample_rate', 16000)

            logger.info(f"\n🎵 保存前5个最高PESQ-WB分数的音频文件到: {audio_save_dir}")

            for i, result in enumerate(top_5_results, 1):
                # 从文件路径中提取文件名（不含扩展名）
                original_filename = Path(result.audio_filepath).stem
                dataset_name = result.dataset_name

                # 创建保存的文件名
                original_file = audio_save_dir / f"top{i:02d}_{original_filename}_original.wav"
                reconstructed_file = audio_save_dir / f"top{i:02d}_{original_filename}_reconstructed.wav"

                # 保存原始音频
                torchaudio.save(original_file, result.original_audio.unsqueeze(0), sample_rate)

                # 保存重建音频
                torchaudio.save(reconstructed_file, result.reconstructed_audio.unsqueeze(0), sample_rate)

                logger.info(f"  ✅ 已保存第{i}名音频文件:")
                logger.info(f"     原始音频: {original_file}")
                logger.info(f"     重建音频: {reconstructed_file}")
                logger.info(f"     PESQ-WB: {result.pesq_wb:.4f}, PESQ-NB: {result.pesq_nb:.4f}, STOI: {result.stoi_score:.4f}")
                logger.info(f"     数据集: {dataset_name}")

        except Exception as e:
            logger.warning(f"保存音频文件失败: {e}")
            import traceback
            logger.warning(f"详细错误: {traceback.format_exc()}")

    def _save_evaluation_results(self, eval_results: List[EvaluationResult],
                                avg_pesq_wb: float, std_pesq_wb: float,
                                avg_pesq_nb: float, std_pesq_nb: float,
                                avg_stoi: float, std_stoi: float):
        """保存评估结果到文件"""
        try:
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)

            # 保存详细结果
            results_file = results_dir / f"evaluation_{self.model_tag}_{Path(self.config_path).stem}.json"

            # 提取各项指标的分数列表
            pesq_wb_scores = [result.pesq_wb for result in eval_results]
            pesq_nb_scores = [result.pesq_nb for result in eval_results]
            stoi_scores = [result.stoi_score for result in eval_results]

            # 获取前5个最高PESQ-WB的结果
            top_5_results = sorted(eval_results, key=lambda x: x.pesq_wb, reverse=True)[:5]

            results_data = {
                'model_tag': self.model_tag,
                'config_file': str(self.config_path),
                'evaluation_time': str(torch.tensor(0.).new_zeros(1).to(self.device).device),  # 获取当前时间
                'statistics': {
                    'avg_pesq_wb': float(avg_pesq_wb),
                    'std_pesq_wb': float(std_pesq_wb),
                    'avg_pesq_nb': float(avg_pesq_nb),
                    'std_pesq_nb': float(std_pesq_nb),
                    'avg_stoi': float(avg_stoi),
                    'std_stoi': float(std_stoi),
                    'min_pesq_wb': float(np.min(pesq_wb_scores)),
                    'max_pesq_wb': float(np.max(pesq_wb_scores)),
                    'num_samples': len(pesq_wb_scores)
                },
                'top_5_results': [
                    {
                        'rank': i+1,
                        'pesq_wb': float(result.pesq_wb),
                        'pesq_nb': float(result.pesq_nb),
                        'stoi_score': float(result.stoi_score),
                        'audio_filepath': result.audio_filepath,
                        'dataset_name': result.dataset_name
                    }
                    for i, result in enumerate(top_5_results)
                ],
                'all_scores': [
                    {
                        'pesq_wb': float(result.pesq_wb),
                        'pesq_nb': float(result.pesq_nb),
                        'stoi_score': float(result.stoi_score),
                        'audio_filepath': result.audio_filepath,
                        'dataset_name': result.dataset_name
                    }
                    for result in eval_results
                ]
            }

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 评估结果已保存到: {results_file}")

        except Exception as e:
            logger.warning(f"保存评估结果失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AudioCodec 模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pretrain_config.yaml",
        help="YAML配置文件路径 (默认: config/pretrain_config.yaml)"
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="latest",
        help="模型标签 (latest, best_model, 或具体epoch标签，默认: latest)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto, cpu, cuda, cuda:0等，默认: auto)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评估样本数，None表示全部 (默认: None)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="结果输出目录 (默认: evaluation_results)"
    )
    parser.add_argument(
        "--training-stage",
        type=str,
        default=None,
        choices=['pretrain', 'posttrain'],
        help="训练阶段 (pretrain/posttrain，默认自动检测)"
    )
    parser.add_argument(
        "--save-all-audio",
        action="store_true",
        help="保存所有生成的音频文件（原始音频和重建音频）"
    )

    args = parser.parse_args()

    try:
        # 创建评估器
        evaluator = ModelEvaluator(
            config_path=args.config,
            model_tag=args.model_tag,
            device=args.device,
            training_stage=args.training_stage,
            save_all_audio=args.save_all_audio
        )

        # 执行评估
        results = evaluator.evaluate_model(max_samples=args.max_samples, output_dir=args.output_dir)

        if results:
            logger.info("✅ 模型评估完成！")
        else:
            logger.error("❌ 模型评估失败！")

    except Exception as e:
        logger.error(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
