# AudioCodec 模型评估脚本使用说明

## 新增功能：保存所有生成音频

该脚本现在支持保存评估过程中生成的所有音频文件，包括原始音频和重建音频。

## 功能特点

1. **原有功能保持不变**：
   - 计算平均PESQ分数（WB和NB）
   - 计算平均STOI分数
   - 保存前5个最高PESQ-WB分数的音频文件
   - 生成详细的评估结果JSON文件

2. **新增功能**：
   - 可选择保存所有评估样本的音频文件
   - 按数据集组织保存的音频文件
   - 为每个音频样本生成元数据文件
   - 实时进度显示音频保存状态

## 使用方法

### 基本评估（不保存所有音频）
```bash
python scripts/evaluate_model.py \
    --config config/posttrain_config.yaml \
    --model-tag latest \
    --device auto
```

### 评估并保存所有音频
```bash
python scripts/evaluate_model.py \
    --config config/posttrain_config.yaml \
    --model-tag latest \
    --device auto \
    --save-all-audio \
    --output-dir my_evaluation_results
```

### 完整参数示例
```bash
python scripts/evaluate_model.py \
    --config config/posttrain_config.yaml \
    --model-tag best_model \
    --device cuda:0 \
    --max-samples 1000 \
    --training-stage posttrain \
    --save-all-audio \
    --output-dir evaluation_results_20240101
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `config/pretrain_config.yaml` | YAML配置文件路径 |
| `--model-tag` | str | `latest` | 模型标签 (latest, best_model, 或具体epoch标签) |
| `--device` | str | `auto` | 计算设备 (auto, cpu, cuda, cuda:0等) |
| `--max-samples` | int | `None` | 最大评估样本数，None表示全部 |
| `--output-dir` | str | `evaluation_results` | 结果输出目录 |
| `--training-stage` | str | `None` | 训练阶段 (pretrain/posttrain，默认自动检测) |
| `--save-all-audio` | flag | `False` | 保存所有生成的音频文件 |

## 输出文件结构

当启用 `--save-all-audio` 选项时，输出目录结构如下：

```
evaluation_results/
├── audio_samples/                    # 前5名音频文件
│   ├── top01_filename_original.wav
│   ├── top01_filename_reconstructed.wav
│   ├── ...
│   └── top05_filename_reconstructed.wav
├── all_audio_samples/                # 所有音频文件（按数据集分组）
│   ├── dataset_0/                    # 第一个数据集
│   │   ├── sample_000001_filename_original.wav
│   │   ├── sample_000001_filename_reconstructed.wav
│   │   ├── sample_000001_filename_metadata.json
│   │   ├── sample_000002_filename_original.wav
│   │   ├── sample_000002_filename_reconstructed.wav
│   │   ├── sample_000002_filename_metadata.json
│   │   └── ...
│   └── dataset_1/                    # 第二个数据集
│       └── ...
└── evaluation_latest_config_name.json # 评估结果JSON文件
```

## 元数据文件格式

每个音频样本都会生成一个对应的元数据文件，包含以下信息：

```json
{
  "sample_id": 1,
  "original_filename": "example_audio",
  "dataset_name": "librispeech",
  "audio_filepath": "/path/to/original/file.wav",
  "pesq_wb": 2.85,
  "pesq_nb": 3.12,
  "stoi_score": 0.892,
  "saved_files": {
    "original": "sample_000001_example_audio_original.wav",
    "reconstructed": "sample_000001_example_audio_reconstructed.wav"
  }
}
```

## 注意事项

1. **存储空间**：启用 `--save-all-audio` 会显著增加存储空间需求，请确保有足够的磁盘空间。

2. **评估时间**：保存所有音频文件会增加评估时间，特别是在处理大量样本时。

3. **内存使用**：音频数据会临时存储在内存中，处理大型数据集时请注意内存使用量。

4. **文件命名**：音频文件使用样本编号和原始文件名组合命名，确保唯一性。

## 使用建议

- 对于快速评估，使用默认设置（不保存所有音频）
- 对于详细分析，启用 `--save-all-audio` 并设置合适的 `--max-samples` 限制
- 使用 `--output-dir` 指定不同的输出目录以组织不同的评估结果
- 在评估大型数据集前，先用小样本测试确保存储空间充足

## 故障排除

1. **内存不足**：减少 `--max-samples` 或增加系统内存
2. **磁盘空间不足**：清理输出目录或增加存储空间
3. **CUDA内存不足**：使用 `--device cpu` 或减少批处理大小
