#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZX-Tokenizer 简化训练脚本
严格按照DeepSpeed规范，集成模型选择、早停和测试集评估
"""

import os
import sys
import yaml
import argparse
import torch.distributed as dist
import deepspeed
from pathlib import Path
import gc

# 添加项目根目录到模块路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from audiocodec.trainer.trainer import AudioCodecTrainer
from audiocodec.model import AudioCodec
from utils.helpers import set_logging
from utils.weight_init import load_whisper_weights

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ZX-Tokenizer Training")
    
    # 基本参数
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--stage', type=str, choices=['pretrain', 'posttrain'], 
                        default='pretrain', help='训练阶段')

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')              

    # 让脚本接受 DeepSpeed/torch 启动器注入的 local_rank 参数
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help='由分布式启动器传入的本地进程序号')

    # DeepSpeed会自动添加这些参数
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        
    return rank, world_size, local_rank


def load_configuration(config_path: str, args) -> dict:
    """加载并更新配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    return config

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    set_logging("INFO" if rank == 0 else "WARNING")
    config = load_configuration(args.config, args)

    if rank == 0:
        print(f"🚀 启动AudioCodec训练 - {args.stage}阶段")

    # ========================== 核心逻辑最终版 ==========================
    # 1. 创建模型
    should_initialize_whisper = (args.stage == 'pretrain' and not args.resume)
    model = AudioCodec(config['model'], initialize_whisper=should_initialize_whisper)

    # 2. 处理权重加载
    if not args.resume and args.stage == "posttrain":
        pretrained_root = config.get('training', {}).get('pretrained_checkpoint', None)
        if pretrained_root:
            if rank == 0:
                print(f"🔷 从头开始后训练，加载预训练 Codec 权重从 {pretrained_root}...")
            AudioCodecTrainer.load_model_weights(model, pretrained_root)
        elif rank == 0:
            print("⚠️ 未提供预训练 Codec 权重，后训练将从随机初始化开始！")

    # 3. 创建 Trainer
    trainer = AudioCodecTrainer(
        config=config,
        training_stage=args.stage,
        args=args,
        model=model,
        resume_from=args.resume
    )

    # 同步所有进程
    if world_size > 1:
        dist.barrier()

    # 4. 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            print("训练被用户中断")
    except Exception as e:
        if rank == 0:
            print(f"训练过程中发生错误: {e}")
        raise
    finally:
        if rank == 0:
            print("训练结束!")

        # 5. 显式删除 trainer 对象
        if rank == 0:
            print("正在释放 Trainer 资源...")
        del trainer
        gc.collect() # 建议进行垃圾回收

        # 6. 同步并安全地关闭分布式环境
        if world_size > 1:
            dist.barrier()
            
        if rank == 0:
            print("训练流程结束，正在清理分布式环境...")
            
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        
        if rank == 0:
            print("程序正常退出。")


if __name__ == "__main__":
    main()