# WER计算脚本使用说明

## 功能概述

此脚本用于计算重建音频的词错误率（Word Error Rate, WER），通过对比HuBERT模型转录的重建音频与LibriSpeech test-clean数据集的参考转录文本。

## 核心功能

1. **自动音频转录**：使用HuBERT大模型转录重建的音频文件
2. **WER计算**：计算转录文本与参考文本的词错误率
3. **批量处理**：支持处理大量音频文件
4. **结果保存**：保存详细的转录对比和统计结果

## 依赖要求

### Python包依赖
```bash
pip install jiwer transformers torch torchaudio tqdm numpy
```

### 模型依赖
- HuBERT-Large模型（本地路径或HuggingFace模型名）
- 默认使用：`facebook/hubert-large-ls960-ft`

## 使用方法

### 基本用法
```bash
python scripts/calculate_wer.py \
    --eval-results-dir post_evaluation_results \
    --manifest-path data/manifests/librispeech/librispeech_test_clean.jsonl \
    --hubert-model-path /root/autodl-tmp/hubert_large_model \
    --device auto \
    --output-dir wer_results
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval-results-dir` | str | **必需** | 评估结果目录（包含音频和元数据文件） |
| `--manifest-path` | str | `librispeech_test_clean.jsonl` | LibriSpeech test-clean manifest文件路径 |
| `--hubert-model-path` | str | `/root/autodl-tmp/hubert_large_model` | HuBERT模型路径 |
| `--device` | str | `auto` | 计算设备 (auto, cpu, cuda, cuda:0等) |
| `--output-dir` | str | `wer_results` | 结果输出目录 |
| `--max-samples` | int | `None` | 最大处理样本数，None表示全部 |

### 完整示例

```bash
# 处理所有2620个样本
python scripts/calculate_wer.py \
    --eval-results-dir post_evaluation_results \
    --manifest-path data/manifests/librispeech/librispeech_test_clean.jsonl \
    --hubert-model-path /root/autodl-tmp/hubert_large_model \
    --device cuda:0 \
    --output-dir wer_full_results

# 测试前100个样本
python scripts/calculate_wer.py \
    --eval-results-dir post_evaluation_results \
    --manifest-path data/manifests/librispeech/librispeech_test_clean.jsonl \
    --hubert-model-path /root/autodl-tmp/hubert_large_model \
    --device auto \
    --output-dir wer_test_results \
    --max-samples 100

# 后台运行完整计算
nohup python scripts/calculate_wer.py \
    --eval-results-dir post_evaluation_results \
    --manifest-path data/manifests/librispeech/librispeech_test_clean.jsonl \
    --hubert-model-path /root/autodl-tmp/hubert_large_model \
    --device auto \
    --output-dir wer_full_results > wer_calculation.log 2>&1 &
```

## 输出结果

### 控制台输出
```
📊 WER计算结果
============================================================
🎯 总体WER: 0.0102 (1.02%)
📈 平均个体WER: 0.0065 (±0.0129)
🔺 最高WER: 0.0323
🔻 最低WER: 0.0000
📊 成功处理样本数: 5/5
```

### 文件输出结构
```
wer_results/
├── wer_results.json          # 详细统计结果
├── transcriptions.jsonl      # 逐条转录对比
└── wer_calculation.log       # 运行日志
```

### 结果文件内容

#### `wer_results.json`
```json
{
  "overall_wer": 0.0102,
  "avg_individual_wer": 0.0065,
  "std_individual_wer": 0.0129,
  "min_wer": 0.0000,
  "max_wer": 0.0323,
  "num_samples": 5,
  "successful_transcriptions": 5,
  "total_metadata_files": 5,
  "individual_results": [...]
}
```

#### `transcriptions.jsonl`
每行包含一个样本的详细信息：
```json
{
  "sample_id": 0,
  "original_filename": "6930-75918-0000",
  "dataset_name": "librispeech_test_clean",
  "reference_text": "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS",
  "hypothesis_text": "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS",
  "wer": 0.0,
  "pesq_wb": 2.1805808544158936,
  "pesq_nb": 2.962635040283203,
  "stoi_score": 0.9201543337841255
}
```

## 性能优化建议

### 硬件要求
- **GPU内存**：建议8GB以上VRAM（用于HuBERT模型推理）
- **系统内存**：建议16GB以上RAM
- **存储空间**：确保有足够空间存储转录结果

### 批处理优化
- 对于大量样本，建议分批处理：
  ```bash
  # 分批处理，每次1000个样本
  python scripts/calculate_wer.py --max-samples 1000 --output-dir wer_batch1
  python scripts/calculate_wer.py --max-samples 1000 --output-dir wer_batch2 # 需要修改脚本支持起始偏移
  ```

### 设备选择
- **GPU推理**：`--device cuda:0`（推荐，速度快）
- **CPU推理**：`--device cpu`（较慢但内存占用小）
- **自动选择**：`--device auto`（默认，自动检测最优设备）

## 结果解读

### WER指标说明
- **总体WER**：所有样本连接后计算的WER（推荐指标）
- **平均个体WER**：每个样本WER的平均值
- **标准差**：WER分布的离散程度
- **最值**：最高和最低WER分数

### 质量评估标准
- **优秀**：WER < 5%
- **良好**：WER 5-15%
- **可接受**：WER 15-30%
- **较差**：WER > 30%

### 与其他指标的关系
脚本同时报告：
- **PESQ分数**：感知音频质量评估
- **STOI分数**：短时间客观可懂度指数

## 故障排除

### 常见问题

1. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用 `--device cpu` 或减少 `--max-samples`

2. **模型加载失败**
   ```
   OSError: Can't load tokenizer
   ```
   解决：检查HuBERT模型路径，或使用HuggingFace模型名

3. **音频文件不存在**
   ```
   WARNING: 重建音频文件不存在
   ```
   解决：检查 `--eval-results-dir` 路径是否正确

4. **转录失败**
   ```
   WARNING: 转录失败
   ```
   解决：检查音频文件格式和完整性

### 调试模式
启用详细日志：
```bash
python scripts/calculate_wer.py --max-samples 5  # 先用少量样本测试
```

### 性能监控
```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 监控CPU和内存使用
htop

# 查看后台任务进度
tail -f wer_calculation.log
```

## 高级用法

### 自定义HuBERT模型
使用不同的HuBERT模型：
```bash
# 使用HuggingFace模型
python scripts/calculate_wer.py \
    --hubert-model-path facebook/hubert-large-ls960-ft \
    --eval-results-dir post_evaluation_results

# 使用其他本地模型
python scripts/calculate_wer.py \
    --hubert-model-path /path/to/custom/hubert/model \
    --eval-results-dir post_evaluation_results
```

### 结果合并
如果分批处理，可以手动合并结果：
```python
import json

# 合并多个批次的结果
results = []
for i in range(1, 4):  # 3个批次
    with open(f'wer_batch{i}/transcriptions.jsonl', 'r') as f:
        for line in f:
            results.append(json.loads(line))

# 计算总体WER
import jiwer
references = [r['reference_text'] for r in results]
hypotheses = [r['hypothesis_text'] for r in results]
overall_wer = jiwer.wer(' '.join(references), ' '.join(hypotheses))
print(f"合并后总体WER: {overall_wer:.4f}")
```

## 注意事项

1. **数据一致性**：确保评估结果目录中的音频与manifest文件对应
2. **模型版本**：不同版本的HuBERT模型可能产生不同的转录结果
3. **采样率**：脚本自动处理采样率转换，但建议使用16kHz音频
4. **文本规范化**：转录文本会自动大写化，与LibriSpeech格式保持一致

## 扩展功能

该脚本可以轻松扩展支持：
- 其他ASR模型（Whisper、Wav2Vec2等）
- 其他数据集格式
- 多语言WER计算
- 字符级错误率计算（CER）

如需定制化功能，请修改 `WERCalculator` 类的相应方法。
