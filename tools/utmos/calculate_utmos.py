import torch, torchaudio
from pathlib import Path
from utmos import UTMOSScore

wav_dir = Path("path/to/output_wavs")
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = UTMOSScore(device)

scores = []
for wav_path in sorted(wav_dir.glob("*.wav")):
    wav, sr = torchaudio.load(wav_path)
    # 转单声道
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # 重采样到16k
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    wav = wav.squeeze(0)  # [T]
    with torch.no_grad():
        score = scorer.score(wav.to(device)).item()
    scores.append(score)
    print(f"{wav_path.name}: {score:.4f}")

if scores:
    print(f"\n平均 UTMOS: {sum(scores)/len(scores):.4f}，样本数: {len(scores)}")
else:
    print("没有找到音频。")