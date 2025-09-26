#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper weight initialization utility for OmniAudioEncoder
"""

import torch
from transformers import WhisperModel


def load_whisper_weights(encoder, whisper_model_name="openai/whisper-small", verbose=False, is_acoustic=False):
    """
    Load pretrained Whisper encoder weights into OmniAudioEncoder.
    
    Args:
        encoder: OmniAudioEncoder instance
        whisper_model_name: Pretrained Whisper model name
        verbose: Print loading progress
        is_acoustic: Whether this is the acoustic encoder (with modifications)
    
    Returns:
        encoder: Encoder with loaded Whisper weights
    """
    if verbose:
        print(f"Loading Whisper weights from {whisper_model_name}...")
    
    # Load Whisper model
    whisper_model = WhisperModel.from_pretrained(whisper_model_name)
    whisper_encoder = whisper_model.encoder
    
    # Get state dictionaries
    encoder_state_dict = encoder.state_dict()
    whisper_state_dict = whisper_encoder.state_dict()
    
    # Copy matching weights (skip positional_embedding as both use sinusoidal)
    matched = 0
    for key in encoder_state_dict.keys():
        if key == 'positional_embedding':
            continue
        if key in whisper_state_dict:
            encoder_state_dict[key].copy_(whisper_state_dict[key])
            matched += 1
            if verbose:
                print(f"  ✓ {key}")
    
    # Load updated weights
    encoder.load_state_dict(encoder_state_dict)
    
    if verbose:
        print(f"Successfully loaded {matched} weight tensors")
        if is_acoustic:
            print("Note: Acoustic encoder loaded with Whisper weights (some modifications may apply)")
    
    return encoder


# Example usage
if __name__ == "__main__":
    from nn.modules import OmniAudioEncoder
    
    # Create encoder
    encoder = OmniAudioEncoder(d_model=768, num_mel_bins=80, num_layers=12)
    
    # Load Whisper weights
    encoder = load_whisper_weights(encoder)
    
    print("Encoder ready for training!")