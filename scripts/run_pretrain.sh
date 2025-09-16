#!/bin/bash
# ZX-Tokenizer 预训练启动脚本 - DeepSpeed Zero Stage 2

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "🚀 启动 WhisperCodec 预训练 - 2张GPU，DeepSpeed Zero Stage 2"
echo "配置: Zero Stage 2 + CPU Offload + Activation Checkpointing"

# 验证模块冻结设置
echo "🔍 验证模块冻结设置..."
python scripts/validate_model_freezing.py

# 验证DeepSpeed配置
echo "🔍 验证DeepSpeed配置..."
python scripts/validate_deepspeed.py

echo "✅ 验证通过，开始训练..."

deepspeed scripts/train.py \
          --config config/pretrain_config.yaml \
          --stage pretrain \
          --deepspeed_config config/deepspeed_pretrain.json
# 恢复训练
export OMPI_MCA_btl=^openib  # 禁用InfiniBand
deepspeed scripts/train.py \
          --config config/pretrain_config.yaml \
          --stage pretrain \
          --deepspeed_config config/deepspeed_pretrain.json \
          --resume outputs/pretrain/checkpoints

echo "预训练完成！"
echo "TensorBoard监控: tensorboard --logdir logs/deepspeed_tensorboard_pretrain/"