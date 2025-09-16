#!/bin/bash
# ZX-Tokenizer 后训练启动脚本 - DeepSpeed Zero Stage 1

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "🚀 启动ZX-Tokenizer后训练 - 1张GPU，DeepSpeed Zero Stage 1"
echo "配置: Zero Stage 1 (无CPU offload)"

# 检查预训练检查点是否存在
PRETRAIN_CHECKPOINT="outputs/pretrain/checkpoints/best_model"
if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then
    echo "❌ 预训练检查点未找到: $PRETRAIN_CHECKPOINT"
    echo "请先完成预训练或指定正确的检查点路径"
    exit 1
fi

deepspeed --master_port=29500 \
          scripts/train.py \
          --config config/posttrain_config.yaml \
          --stage posttrain \
          --deepspeed_config config/deepspeed_posttrain.json

deepspeed --master_port=29500 \
          scripts/train.py \
          --config config/posttrain_config.yaml \
          --stage posttrain \
          --deepspeed_config config/deepspeed_posttrain.json \
          --resume outputs/posttrain/checkpoints

echo "后训练完成！"
echo "TensorBoard监控: tensorboard --logdir logs/deepspeed_tensorboard_posttrain/"