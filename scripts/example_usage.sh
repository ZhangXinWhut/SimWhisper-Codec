# 测试评估脚本的基本功能
python scripts/evaluate_model.py --max-samples 10 --device cpu

# 评估best_model
python scripts/evaluate_model.py --model-tag best_model --max-samples 20

# 评估特定epoch的模型
python scripts/evaluate_model.py --model-tag epoch_150_step_46950 --max-samples 50

# 使用GPU进行完整评估
python scripts/evaluate_model.py --device cuda:0

# 手动指定训练阶段（覆盖自动检测）
python scripts/evaluate_model.py --config config/pretrain_config.yaml --training-stage posttrain --max-samples 50

# 权重加载逻辑：
# 1. 使用pretrain_config.yaml时，默认加载outputs/pretrain/checkpoints的权重
# 2. 使用posttrain_config.yaml时，默认加载outputs/posttrain/checkpoints的权重
# 3. 可以手动指定--training-stage来覆盖自动检测

# 评估完成后会自动保存：
# 1. evaluation_results/evaluation_latest_*.json (评估结果)
#    - 包含PESQ-WB、PESQ-NB、STOI三项指标的统计信息
#    - 自动检测训练阶段：pretrain(不使用量化) vs posttrain(使用量化)
# 2. evaluation_results/audio_samples/ 目录下的前5个最高PESQ-WB分数的音频文件
#    - top01_*_original.wav 和 top01_*_reconstructed.wav (第1名)
#    - top02_*_original.wav 和 top02_*_reconstructed.wav (第2名)
#    - top03_*_original.wav 和 top03_*_reconstructed.wav (第3名)
#    - top04_*_original.wav 和 top04_*_reconstructed.wav (第4名)
#    - top05_*_original.wav 和 top05_*_reconstructed.wav (第5名)
