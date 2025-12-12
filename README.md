<div align="center">

# 🎙️ SimWhisper-Codec

### Speaking Clearly: A Simplified Whisper-Based Codec for Low-Bitrate Speech Coding

<p align="center">
  <img src="docs/assets/SimWhisper-Codec.png" width="80%" alt="SimWhisper-Codec Architecture">
</p>

<p>
  <a href="https://zhangxinwhut.github.io/SimWhisper-Codec/"><img src="https://img.shields.io/badge/🎧_Demo-Online-brightgreen" alt="Demo"></a>
  <a href="https://arxiv.org/pdf/2510.20504"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="paper"></a>
  <a href="https://huggingface.co/xxx123456/SimWhisper_Codec"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
</p>

*A semantic-first speech codec that achieves superior performance through architectural simplification rather than complex supervision.*

</div>

---

## ✨ Highlights

- 🚀 **low Bitrate**: Only **1.1 kbps** at 16 kHz sampling rate
- 🔊 **High Quality Speech Reconstruction**: Achieving UTMOS 4.00 WER 2.75 (hubert-large-ls960-ft) sim 0.83 (wavlm_large_finetune) stoi 0.93 pesq-nb 3.29 pesq-wb 2.72 on librispeech-test-clean reconstruction (gt: WER 2.16 UTMOS 4.09)
- 🧊 **Frozen Encoder**: No fine-tuning of Whisper encoder required
- ⚡ **Simple & Efficient**: Architectural simplification over complex supervision

## 📊 Performance

| Model | Bitrate | WER ↓ | PESQ-NB ↑ | PESQ-WB ↑ | STOI ↑ | SIM ↑ | UTMOS ↑ |
|:------|:-------:|:-----:|:---------:|:---------:|:------:|:-----:|:-------:|
| XCodec2.0 | 0.8 kbps | 2.61 | 3.04 | 2.43 | 0.92 | 0.82 | **4.13** |
| XY-Tokenizer | 1.0 kbps | **2.46** | 3.00 | 2.41 | 0.91 | **0.84** | 3.98 |
| **SimWhisper-Codec** | 1.1 kbps | 2.75 | **3.29** | **2.72** | **0.93** | 0.83 | 4.00 |

*Evaluated on LibriSpeech test-clean*

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ZhangXinWhut/SimWhisper-Codec.git && cd SimWhisper-Codec

# Create and activate conda environment
conda create -n swcodec python=3.10 -y && conda activate swcodec

# Install dependencies
pip install -r requirements.txt
```

## Available Models 🗂️

| Model Name | Hugging Face | Training Data |
|:----------:|:-------------:|:---------------:|
| SimWhisper-Codec | [🤗](https://huggingface.co/xxx123456/SimWhisper_Codec) | LibriSpeech |


### Download Model Weights

You need to download the SimWhisper-Codec model weights. You can find the weights in the [SimWhisper-Codec Hugging Face repository](https://huggingface.co/xxx123456/SimWhisper_Codec).

```bash
mkdir -p ./weights && huggingface-cli download xxx123456/SimWhisper_Codec SimWhisperCodec.pt --local-dir ./weights/
```

### Inference

```python
python inference.py --input_dir /path/to/LibriSpeech/test-clean
```

The reconstructed audio files will be available in the `output_wavs/` directory.

## 🙏 Acknowledgements

Our codebase builds upon the [XY-Tokenizer](https://github.com/gyt1145028706/XY-Tokenizer). We thank the authors for their excellent work.

## 📝 Citation

If you find this work useful in your research, please cite our paper:

```
@misc{zhang2025speakingclearlysimplifiedwhisperbased,
      title={Speaking Clearly: A Simplified Whisper-Based Codec for Low-Bitrate Speech Coding}, 
      author={Xin Zhang and Lin Li and Xiangni Lu and Jianquan Liu and Kong Aik Lee},
      year={2025},
      eprint={2510.20504},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.20504}, 
}
```
