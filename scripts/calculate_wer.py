#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WER计算脚本

功能：
- 使用HuBERT模型转录重建音频
- 与LibriSpeech test-clean的参考转录文本计算WER
- 支持批量处理所有生成的重建音频
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

import torch
import torchaudio
from transformers import Wav2Vec2Processor, HubertForCTC
import jiwer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wer_calculation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class WERCalculator:
    """WER计算器"""

    def __init__(self, hubert_model_path: str, device: str = "auto"):
        """
        初始化WER计算器

        Args:
            hubert_model_path: HuBERT模型路径
            device: 计算设备
        """
        self.device = self._setup_device(device)
        self.hubert_model_path = hubert_model_path
        
        # 加载HuBERT模型和处理器
        self._load_hubert_model()
        
        logger.info("✅ WER计算器初始化完成")
        logger.info(f"  🤖 HuBERT模型: {hubert_model_path}")
        logger.info(f"  💻 设备: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_hubert_model(self):
        """加载HuBERT模型"""
        try:
            logger.info("🔄 正在加载HuBERT模型...")
            
            # 检查是否是本地路径
            if os.path.exists(self.hubert_model_path):
                # 从本地路径加载
                self.processor = Wav2Vec2Processor.from_pretrained(self.hubert_model_path)
                self.model = HubertForCTC.from_pretrained(self.hubert_model_path)
                logger.info(f"✅ 从本地路径加载HuBERT模型: {self.hubert_model_path}")
            else:
                # 从HuggingFace Hub加载
                self.processor = Wav2Vec2Processor.from_pretrained(self.hubert_model_path)
                self.model = HubertForCTC.from_pretrained(self.hubert_model_path)
                logger.info(f"✅ 从HuggingFace Hub加载HuBERT模型: {self.hubert_model_path}")
            
            # 移动模型到指定设备
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ HuBERT模型加载失败: {e}")
            raise

    def transcribe_audio(self, audio_path: str, sample_rate: int = 16000) -> str:
        """
        转录单个音频文件

        Args:
            audio_path: 音频文件路径
            sample_rate: 采样率

        Returns:
            转录文本
        """
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样到16kHz（如果需要）
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 转换为numpy并归一化
            audio_array = waveform.squeeze().numpy()
            
            # 使用处理器处理音频
            inputs = self.processor(
                audio_array, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # 移动到设备
            input_values = inputs.input_values.to(self.device)
            
            # 推理
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                
            # 解码
            transcription = self.processor.decode(predicted_ids[0])
            
            return transcription.strip()
            
        except Exception as e:
            logger.warning(f"转录音频 {audio_path} 失败: {e}")
            return ""

    def load_librispeech_transcripts(self, manifest_path: str) -> Dict[str, str]:
        """
        加载LibriSpeech测试集的参考转录文本

        Args:
            manifest_path: manifest文件路径

        Returns:
            {utterance_id: text} 字典
        """
        transcripts = {}
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    utterance_id = data['utterance_id']
                    text = data['text']
                    transcripts[utterance_id] = text
                    
            logger.info(f"✅ 加载了 {len(transcripts)} 个参考转录文本")
            return transcripts
            
        except Exception as e:
            logger.error(f"❌ 加载参考转录文本失败: {e}")
            raise

    def find_metadata_files(self, eval_results_dir: str) -> List[str]:
        """
        查找所有元数据文件

        Args:
            eval_results_dir: 评估结果目录

        Returns:
            元数据文件路径列表
        """
        metadata_files = []
        eval_path = Path(eval_results_dir)
        
        # 查找所有元数据文件
        for metadata_file in eval_path.rglob("*_metadata.json"):
            metadata_files.append(str(metadata_file))
        
        logger.info(f"✅ 找到 {len(metadata_files)} 个音频元数据文件")
        return sorted(metadata_files)

    def calculate_wer(self, eval_results_dir: str, manifest_path: str, 
                     output_dir: str = "wer_results", max_samples: Optional[int] = None) -> Dict:
        """
        计算WER分数

        Args:
            eval_results_dir: 评估结果目录
            manifest_path: LibriSpeech manifest文件路径
            output_dir: 输出目录
            max_samples: 最大处理样本数

        Returns:
            WER计算结果
        """
        logger.info("🚀 开始WER计算...")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载参考转录文本
        reference_transcripts = self.load_librispeech_transcripts(manifest_path)
        
        # 查找所有元数据文件
        metadata_files = self.find_metadata_files(eval_results_dir)
        
        if max_samples:
            metadata_files = metadata_files[:max_samples]
            logger.info(f"🔢 限制处理样本数: {max_samples}")
        
        # 存储结果
        results = []
        hypothesis_texts = []
        reference_texts = []
        successful_transcriptions = 0
        
        # 创建进度条
        progress_bar = tqdm(
            metadata_files,
            desc="转录音频",
            unit="文件",
            leave=True,
            dynamic_ncols=True
        )
        
        for metadata_file in progress_bar:
            try:
                # 读取元数据
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                sample_id = metadata['sample_id']
                original_filename = metadata['original_filename']
                dataset_name = metadata['dataset_name']
                
                # 获取重建音频路径
                metadata_dir = Path(metadata_file).parent
                reconstructed_audio_path = metadata_dir / metadata['saved_files']['reconstructed']
                
                if not reconstructed_audio_path.exists():
                    logger.warning(f"重建音频文件不存在: {reconstructed_audio_path}")
                    continue
                
                # 获取参考转录文本
                if original_filename not in reference_transcripts:
                    logger.warning(f"找不到参考转录文本: {original_filename}")
                    continue
                
                reference_text = reference_transcripts[original_filename]
                
                # 转录重建音频
                hypothesis_text = self.transcribe_audio(str(reconstructed_audio_path))
                
                if not hypothesis_text:
                    logger.warning(f"转录失败: {reconstructed_audio_path}")
                    continue
                
                # 计算单个样本的WER
                try:
                    sample_wer = jiwer.wer(reference_text, hypothesis_text)
                except:
                    sample_wer = 1.0  # 如果计算失败，设为100%错误率
                
                # 保存结果
                result = {
                    'sample_id': sample_id,
                    'original_filename': original_filename,
                    'dataset_name': dataset_name,
                    'reference_text': reference_text,
                    'hypothesis_text': hypothesis_text,
                    'wer': sample_wer,
                    'pesq_wb': metadata.get('pesq_wb', 0.0),
                    'pesq_nb': metadata.get('pesq_nb', 0.0),
                    'stoi_score': metadata.get('stoi_score', 0.0)
                }
                results.append(result)
                
                # 用于总体WER计算
                hypothesis_texts.append(hypothesis_text)
                reference_texts.append(reference_text)
                
                successful_transcriptions += 1
                
                # 更新进度条
                if successful_transcriptions > 0:
                    current_wer = jiwer.wer(' '.join(reference_texts), ' '.join(hypothesis_texts))
                    progress_bar.set_postfix({
                        'WER': f'{current_wer:.3f}',
                        '成功': f'{successful_transcriptions}'
                    })
                
            except Exception as e:
                logger.warning(f"处理文件 {metadata_file} 失败: {e}")
                continue
        
        progress_bar.close()
        
        if not results:
            logger.error("❌ 没有成功处理任何音频文件")
            return {}
        
        # 计算总体WER
        overall_wer = jiwer.wer(' '.join(reference_texts), ' '.join(hypothesis_texts))
        
        # 计算统计信息
        individual_wers = [r['wer'] for r in results]
        avg_wer = np.mean(individual_wers)
        std_wer = np.std(individual_wers)
        min_wer = np.min(individual_wers)
        max_wer = np.max(individual_wers)
        
        # 准备最终结果
        final_results = {
            'overall_wer': overall_wer,
            'avg_individual_wer': avg_wer,
            'std_individual_wer': std_wer,
            'min_wer': min_wer,
            'max_wer': max_wer,
            'num_samples': len(results),
            'successful_transcriptions': successful_transcriptions,
            'total_metadata_files': len(metadata_files),
            'individual_results': results
        }
        
        # 打印结果
        logger.info("\n" + "="*60)
        logger.info("📊 WER计算结果")
        logger.info("="*60)
        logger.info(f"🎯 总体WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
        logger.info(f"📈 平均个体WER: {avg_wer:.4f} (±{std_wer:.4f})")
        logger.info(f"🔺 最高WER: {max_wer:.4f}")
        logger.info(f"🔻 最低WER: {min_wer:.4f}")
        logger.info(f"📊 成功处理样本数: {successful_transcriptions}/{len(metadata_files)}")
        
        # 保存详细结果
        results_file = output_path / "wer_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 详细结果已保存到: {results_file}")
        
        # 保存转录文本对比
        transcription_file = output_path / "transcriptions.jsonl"
        with open(transcription_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"📄 转录文本对比已保存到: {transcription_file}")
        
        return final_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WER计算脚本")
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        required=True,
        help="评估结果目录（包含音频和元数据文件）"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="/root/autodl-tmp/WhisperCodec/data/manifests/librispeech/librispeech_test_clean.jsonl",
        help="LibriSpeech test-clean manifest文件路径"
    )
    parser.add_argument(
        "--hubert-model-path",
        type=str,
        default="/root/autodl-tmp/hubert_large_model",
        help="HuBERT模型路径（本地路径或HuggingFace模型名）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto, cpu, cuda, cuda:0等)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wer_results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数，None表示全部"
    )

    args = parser.parse_args()

    try:
        # 创建WER计算器
        calculator = WERCalculator(
            hubert_model_path=args.hubert_model_path,
            device=args.device
        )

        # 计算WER
        results = calculator.calculate_wer(
            eval_results_dir=args.eval_results_dir,
            manifest_path=args.manifest_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )

        if results:
            logger.info("✅ WER计算完成！")
            logger.info(f"🎯 总体WER: {results['overall_wer']:.4f}")
        else:
            logger.error("❌ WER计算失败！")

    except Exception as e:
        logger.error(f"❌ WER计算过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
