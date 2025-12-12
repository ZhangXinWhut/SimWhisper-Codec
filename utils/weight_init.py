#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Whisper weight initialization utility for SimWhisper-Codec.
"""

import torch
from transformers import WhisperModel


def load_whisper_weights(encoder, whisper_model_name="openai/whisper-small", verbose=False, is_acoustic=False, local_files_only=False):
    """
    Load pretrained Whisper encoder weights into SimWhisper encoder.
    Skips positional embeddings since both use sinusoidal embeddings.

    Args:
        encoder: SimWhisper encoder instance
        whisper_model_name: Pretrained Whisper model name or local path
        verbose: Print loading progress
        is_acoustic: Whether this is the acoustic encoder (with modifications)
        local_files_only: If True, only load from local files/cache, don't download

    Returns:
        encoder: Encoder with loaded Whisper weights
    """
    if verbose:
        print(f"Loading Whisper weights from {whisper_model_name}...")
        if local_files_only:
            print("  Using local files only (no downloads)")

    # Load Whisper model
    # 支持本地路径和 Hugging Face 模型名称
    try:
        whisper_model = WhisperModel.from_pretrained(
            whisper_model_name,
            local_files_only=local_files_only
        )
        whisper_encoder = whisper_model.encoder
    except Exception as e:
        if "local_files_only" in str(e) and not local_files_only:
            print(f"Failed to load from {whisper_model_name}, trying with local_files_only=True...")
            try:
                whisper_model = WhisperModel.from_pretrained(
                    whisper_model_name,
                    local_files_only=True
                )
                whisper_encoder = whisper_model.encoder
            except Exception as e2:
                raise RuntimeError(f"Failed to load Whisper model from {whisper_model_name}: {e2}")
        else:
            raise RuntimeError(f"Failed to load Whisper model from {whisper_model_name}: {e}")
    
    # Get state dictionaries
    state_dict = encoder.state_dict()
    whisper_state_dict = whisper_encoder.state_dict()
    
    # Copy matching weights (skip positional_embedding as both use sinusoidal)
    matched = 0
    for key in state_dict.keys():
        if key == 'positional_embedding':
            continue
        if key in whisper_state_dict:
            state_dict[key].copy_(whisper_state_dict[key])
            matched += 1
            if verbose:
                print(f"  ✓ {key}")
    
    # Load updated weights
    encoder.load_state_dict(state_dict)

    if verbose:
        print(f"Successfully loaded {matched} weight tensors")
        if is_acoustic:
            print("Note: Acoustic encoder loaded with Whisper weights (some modifications may apply)")

    return encoder


# Example usage
if __name__ == "__main__":
    from audiocodec.nn.modules import OmniAudioEncoder
    
    # Create encoder
    encoder = OmniAudioEncoder(d_model=768, num_mel_bins=80, num_layers=12)
    
    # Load Whisper weights
    encoder = load_whisper_weights(encoder)
    
    print("Encoder ready for training!")