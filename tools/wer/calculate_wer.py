#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只计算重建音频的 WER (词错误率)
"""

import os
import sys
import glob
import torch
import re
import unicodedata
from tqdm import tqdm
from transformers import HubertForCTC, Wav2Vec2Processor

try:
    import soundfile
except ImportError:
    import librosa
    soundfile = None

remove_tag = True
spacelist = [' ', '\t', '\r', '\n']
puncts = [
    '!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』',
    '《', '》'
]


def characterize(string):
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist:
            i += 1
            continue
        if cat1 == 'Lo':
            res.append(char)
            i += 1
        else:
            sep = ' '
            if char == '<': sep = '>'
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in spacelist) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == '>':
                j += 1
            res.append(string[i:j])
            i = j
    return res


def stripoff_tags(x):
    if not x: return ''
    chars = []
    i = 0
    T = len(x)
    while i < T:
        if x[i] == '<':
            while i < T and x[i] != '>':
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """sentence, ignore_words are both in unicode"""
    new_sentence = []
    for token in sentence:
        x = token
        if not cs:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:
    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {}
        self.cost['cor'] = 0
        self.cost['sub'] = 1
        self.cost['del'] = 1
        self.cost['ins'] = 1

    def calculate(self, lab, rec):
        # Initialization
        lab.insert(0, '')
        rec.insert(0, '')
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            for element in row:
                element['dist'] = 0
                element['error'] = 'non'
            while len(row) < len(rec):
                row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)):
            self.space[i][0]['dist'] = i
            self.space[i][0]['error'] = 'del'
        for j in range(len(rec)):
            self.space[0][j]['dist'] = j
            self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for token in lab:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        for token in rec:
            if token not in self.data and len(token) > 0:
                self.data[token] = {
                    'all': 0,
                    'cor': 0,
                    'sub': 0,
                    'ins': 0,
                    'del': 0
                }
        # Computing edit distance
        for i, lab_token in enumerate(lab):
            for j, rec_token in enumerate(rec):
                if i == 0 or j == 0:
                    continue
                min_dist = sys.maxsize
                min_error = 'none'
                dist = self.space[i - 1][j]['dist'] + self.cost['del']
                error = 'del'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                dist = self.space[i][j - 1]['dist'] + self.cost['ins']
                error = 'ins'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                if lab_token == rec_token:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['cor']
                    error = 'cor'
                else:
                    dist = self.space[i - 1][j - 1]['dist'] + self.cost['sub']
                    error = 'sub'
                if dist < min_dist:
                    min_dist = dist
                    min_error = error
                self.space[i][j]['dist'] = min_dist
                self.space[i][j]['error'] = min_error
        # Tracing back
        result = {
            'lab': [],
            'rec': [],
            'all': 0,
            'cor': 0,
            'sub': 0,
            'ins': 0,
            'del': 0
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            if self.space[i][j]['error'] == 'cor':
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
                    result['all'] = result['all'] + 1
                    result['cor'] = result['cor'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'sub':
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
                    result['all'] = result['all'] + 1
                    result['sub'] = result['sub'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, rec[j])
                i = i - 1
                j = j - 1
            elif self.space[i][j]['error'] == 'del':
                if len(lab[i]) > 0:
                    self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
                    self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
                    result['all'] = result['all'] + 1
                    result['del'] = result['del'] + 1
                result['lab'].insert(0, lab[i])
                result['rec'].insert(0, "")
                i = i - 1
            elif self.space[i][j]['error'] == 'ins':
                if len(rec[j]) > 0:
                    self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
                    result['ins'] = result['ins'] + 1
                result['lab'].insert(0, "")
                result['rec'].insert(0, rec[j])
                j = j - 1
            elif self.space[i][j]['error'] == 'non':
                break
            else:
                print(
                    'this should not happen , i = {i} , j = {j} , error = {error}'
                    .format(i=i, j=j, error=self.space[i][j]['error']))
        return result


# ============================================================================
# 音频处理和WER计算
# ============================================================================

def read_audio(audio_path):
    """读取音频文件"""
    if soundfile is not None:
        audio, sr = soundfile.read(audio_path)
    else:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
    return audio, sr


def collect_transcripts(dataset_dir: str):
    """收集所有转录文本"""
    print("正在收集转录文本...")
    transcript_dict = {}
    
    # 遍历所有 .trans.txt 文件
    trans_files = glob.glob(os.path.join(dataset_dir, '*/*/*.trans.txt'))
    
    for trans_file in tqdm(trans_files, desc="读取转录文件"):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 格式: FILEID TEXT
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcript_dict[file_id] = text
    
    print(f"收集到 {len(transcript_dict)} 条转录文本")
    return transcript_dict


def calculate_wer_for_folder(audio_folder: str, ref_text_dict: dict, device, 
                             processor, hubert, calculator, 
                             folder_type: str = "重建音频",
                             audio_ext: str = ".wav",
                             find_original: bool = False,
                             original_base_dir: str = None):
    """
    计算某个文件夹中音频的WER
    
    Args:
        audio_folder: 音频文件夹路径
        ref_text_dict: 参考文本字典
        device: 设备
        processor: Wav2Vec2Processor
        hubert: HubertForCTC模型
        calculator: WER计算器
        folder_type: 文件夹类型描述
        audio_ext: 音频文件扩展名
        find_original: 是否需要在原始数据集中查找音频文件
        original_base_dir: 原始数据集基础目录
    """
    results = []
    processed_count = 0
    
    # 获取音频文件列表
    if find_original:
        # 从 ref_text_dict 中获取所有文件ID，然后在原始数据集中查找
        audio_files = []
        for file_id in sorted(ref_text_dict.keys()):
            speaker_id = file_id.split('-')[0]
            chapter_id = file_id.split('-')[1]
            audio_path = os.path.join(original_base_dir, speaker_id, chapter_id, file_id + audio_ext)
            if os.path.exists(audio_path):
                audio_files.append((file_id, audio_path))
    else:
        # 直接从文件夹中获取音频文件
        audio_files = []
        for f in sorted(os.listdir(audio_folder)):
            if f.endswith(audio_ext):
                file_id = os.path.splitext(f)[0]
                audio_files.append((file_id, os.path.join(audio_folder, f)))
    
    print(f"\n{'='*60}")
    print(f"计算 {folder_type} 的 WER")
    print(f"{'='*60}")
    print(f"找到 {len(audio_files)} 个音频文件")
    
    for file_id, audio_path in tqdm(audio_files, desc=f"处理{folder_type}"):
        # 检查是否有对应的参考文本
        if file_id not in ref_text_dict:
            continue
        
        ref_text = ref_text_dict[file_id]
        
        try:
            # 读取音频
            audio, sr = read_audio(audio_path)
            
            # 转换为tensor并移到GPU
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
            
            # 使用Hubert进行语音识别
            with torch.no_grad():
                logits = hubert(audio_tensor).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.decode(predicted_ids[0])
            
            # 标准化转录文本和参考文本
            transcription_normalized = normalize(characterize(transcription), [], False, None)
            ref_text_normalized = normalize(characterize(ref_text), [], False, None)
            
            # 计算此utterance的WER
            result = calculator.calculate(transcription_normalized, ref_text_normalized)
            results.append(result)
            processed_count += 1
            
        except Exception as e:
            print(f"\n处理 {file_id} 时出错: {e}")
            continue
    
    # 汇总统计
    if len(results) == 0:
        print(f"\n警告: {folder_type} 没有成功处理任何音频文件!")
        return None
    
    N, C, S, D, I = 0, 0, 0, 0, 0
    for result in results:
        N += result["all"]      # 总词数
        C += result["cor"]      # 正确
        S += result["sub"]      # 替换
        D += result["del"]      # 删除
        I += result["ins"]      # 插入
    
    # 计算WER
    wer = (S + D + I) / N * 100 if N > 0 else 0.0
    
    return {
        'type': folder_type,
        'processed': processed_count,
        'total_words': N,
        'correct': C,
        'substitutions': S,
        'deletions': D,
        'insertions': I,
        'wer': wer
    }


def calculate_wer(rec_folder: str, original_folder: str, ref_text_dict: dict, 
                  device, hubert_model_path: str):
    """计算重建音频和原始音频的WER"""
    
    # 加载模型
    print("正在加载 Hubert 模型...")
    processor = Wav2Vec2Processor.from_pretrained(hubert_model_path)
    hubert = HubertForCTC.from_pretrained(
        hubert_model_path,
        weights_only=False  # 解决安全性限制问题
    ).to(device)
    hubert.eval()
    calculator = Calculator()
    print("模型加载完成!\n")
    
    all_results = []
    
    # 1. 计算重建音频的WER
    result_rec = calculate_wer_for_folder(
        audio_folder=rec_folder,
        ref_text_dict=ref_text_dict,
        device=device,
        processor=processor,
        hubert=hubert,
        calculator=calculator,
        folder_type="重建音频",
        audio_ext=".wav",
        find_original=False
    )
    if result_rec:
        all_results.append(result_rec)
    
    # 2. 计算原始音频的WER
    result_orig = calculate_wer_for_folder(
        audio_folder=original_folder,
        ref_text_dict=ref_text_dict,
        device=device,
        processor=processor,
        hubert=hubert,
        calculator=calculator,
        folder_type="原始音频",
        audio_ext=".flac",
        find_original=True,
        original_base_dir=original_folder
    )
    if result_orig:
        all_results.append(result_orig)
    
    return all_results


def main():
    """主函数"""
    # 设置路径
    original_folder = "path/to/LibriSpeech/test-clean"
    rec_folder = "path/to/output_wavs"
    hubert_model_path = "path/to/hubert_large_ls960_ft"
    
    # 检查路径是否存在
    if not os.path.exists(original_folder):
        print(f"错误: 原始音频文件夹不存在: {original_folder}")
        return
    
    if not os.path.exists(rec_folder):
        print(f"错误: 重建音频文件夹不存在: {rec_folder}")
        return
    
    if not os.path.exists(hubert_model_path):
        print(f"错误: Hubert 模型路径不存在: {hubert_model_path}")
        return
    
    # 收集转录文本
    transcript_dict = collect_transcripts(original_folder)
    
    if len(transcript_dict) == 0:
        print("错误: 没有找到任何转录文本!")
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")
    
    # 计算WER (包括重建音频和原始音频)
    results = calculate_wer(rec_folder, original_folder, transcript_dict, device, hubert_model_path)
    
    if not results or len(results) == 0:
        print("错误: 没有成功计算任何WER结果!")
        return
    
    # 打印结果
    print("\n" + "="*70)
    print("WER 计算结果对比")
    print("="*70)
    
    for result in results:
        print(f"\n【{result['type']}】")
        print(f"  处理的音频文件数: {result['processed']}")
        print(f"  总词数 (N):       {result['total_words']}")
        print(f"  正确 (C):         {result['correct']}")
        print(f"  替换 (S):         {result['substitutions']}")
        print(f"  删除 (D):         {result['deletions']}")
        print(f"  插入 (I):         {result['insertions']}")
        print(f"  {'-'*65}")
        print(f"  WER:              {result['wer']:.2f}%")
    
    # 如果两个都有结果，计算WER增加量
    if len(results) == 2:
        wer_orig = results[1]['wer']  # 原始音频
        wer_rec = results[0]['wer']   # 重建音频
        wer_increase = wer_rec - wer_orig
        
        print(f"\n{'='*70}")
        print("对比分析")
        print("="*70)
        print(f"  原始音频 WER:     {wer_orig:.2f}%")
        print(f"  重建音频 WER:     {wer_rec:.2f}%")
        print(f"  WER 增加量:       {wer_increase:+.2f}%")
        if wer_orig > 0:
            relative_increase = (wer_increase / wer_orig) * 100
            print(f"  相对增加:         {relative_increase:+.2f}%")
    
    print("="*70)


if __name__ == "__main__":
    main()
