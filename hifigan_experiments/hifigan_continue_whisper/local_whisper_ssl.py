"""
Local custom encoder SSL module for continuous speech feature extraction.
Uses custom OmniAudioEncoder with Whisper weight initialization.

Authors
 * Modified for custom encoder with Whisper initialization
"""

import os
import torch
import numpy as np
from torch import nn

from nn.modules import OmniAudioEncoder
from nn.feature_extractor import MelFeatureExtractor
from utils.weight_init import load_whisper_weights
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class CustomWhisperExtractor(nn.Module):
    """
    Custom Whisper-based continuous feature extractor using OmniAudioEncoder.
    
    Arguments
    ---------
    whisper_model_name : str
        Whisper model name for weight initialization.
    feature_extractor_config : dict
        Configuration for mel feature extractor.
    encoder_config : dict
        Configuration for the encoder.
    device : str
        Device to use for computation.
    """

    def __init__(
        self,
        whisper_model_name="openai/whisper-small",
        feature_extractor_config=None,
        encoder_config=None,
        device="cuda",
        freeze_encoder=True,  # Parameter to control freezing
        layer_id=-1,  # Which layer to extract features from (-1 for last layer)
    ):
        super().__init__()
        
        self.device = device
        self.whisper_model_name = whisper_model_name
        self.layer_id = layer_id  # Store layer selection
        
        # Default configurations
        if feature_extractor_config is None:
            feature_extractor_config = {
                "chunk_length": 30,
                "feature_size": 80,
                "hop_length": 160,
                "n_fft": 400,
                "sampling_rate": 16000,
                "padding_value": 0.0,
                "return_attention_mask": False,
            }
        
        if encoder_config is None:
            encoder_config = {
                "num_mel_bins": 80,
                "sampling_rate": 16000,
                "hop_length": 160,
                "stride_size": 2,
                "kernel_size": 3,
                "d_model": 768,  # whisper-small dimension
                "scale_embedding": False,
                "max_audio_seconds": 30,
                "encoder_layers": 12,  # whisper-small layers
                "encoder_attention_heads": 12,  # whisper-small heads
                "encoder_ffn_dim": 3072,  # whisper-small ffn_dim
                "is_acoustic": True,
            }
        
        # Initialize feature extractor
        logger.info(f"Initializing mel feature extractor...")
        self.feature_extractor = MelFeatureExtractor(**feature_extractor_config)
        
        # Initialize encoder
        logger.info(f"Initializing custom encoder...")
        self.encoder = OmniAudioEncoder(**encoder_config).to(device)
        
        # Load Whisper weights
        logger.info(f"Loading Whisper weights from {whisper_model_name}...")
        self.encoder = load_whisper_weights(
            self.encoder, 
            whisper_model_name=whisper_model_name,
            verbose=True,
            is_acoustic=encoder_config.get("is_acoustic", False)
        )
        
        # Set to eval mode and optionally freeze encoder parameters
        self.encoder.eval()
        if freeze_encoder:
            self.freeze_encoder()  # Freeze encoder parameters
            logger.info("Encoder frozen for feature extraction only.")
        else:
            logger.info("Encoder unfrozen - parameters will be updated during training.")
        
        # Get feature dimension and other properties
        self.feature_dim = encoder_config["d_model"]
        self.hop_length = encoder_config["hop_length"]
        self.stride_size = encoder_config["stride_size"]
        self.sample_rate = feature_extractor_config.get("sampling_rate", 16000)
        
        logger.info(f"CustomWhisperExtractor initialized:")
        logger.info(f"  - Model: {whisper_model_name}")
        logger.info(f"  - Layer: {layer_id} ({'last layer' if layer_id == -1 else f'layer {layer_id}'})")
        logger.info(f"  - Feature dim: {self.feature_dim}")
        logger.info(f"  - Hop length: {self.hop_length}")
        logger.info(f"  - Stride size: {self.stride_size}")
        logger.info(f"  - Device: {device}")

    def extract_features(self, audio, input_lengths=None):
        """
        Extract continuous features from custom encoder using batch processing.
        
        Arguments
        ---------
        audio : torch.Tensor or list
            Input waveform [batch, time] or list of waveforms.
        input_lengths : torch.Tensor, optional
            Length of each audio sequence [batch].
            
        Returns
        -------
        features : torch.Tensor
            Continuous features [batch, time, feature_dim].
        output_lengths : torch.Tensor
            Length of each feature sequence [batch].
        """
        with torch.no_grad():
            # Handle different input formats
            if isinstance(audio, list):
                # Already a list of numpy arrays or tensors
                if input_lengths is None:
                    input_lengths = torch.tensor([len(x) for x in audio], device=self.device)
                list_x = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in audio]
            elif isinstance(audio, torch.Tensor):
                if len(audio.shape) == 1:
                    # Single audio sequence
                    list_x = [audio.cpu().numpy()]
                    input_lengths = torch.tensor([len(audio)], device=self.device)
                elif len(audio.shape) == 2:
                    # Batch of audio sequences
                    if input_lengths is None:
                        # Assume all sequences have the same length
                        input_lengths = torch.full((audio.shape[0],), audio.shape[1], device=self.device)
                    list_x = [audio[i, :input_lengths[i]].cpu().numpy() for i in range(audio.shape[0])]
                else:
                    raise ValueError(f"Unsupported audio shape: {audio.shape}")
            else:
                # Numpy array
                audio = torch.from_numpy(audio)
                if len(audio.shape) == 1:
                    list_x = [audio.numpy()]
                    input_lengths = torch.tensor([len(audio)], device=self.device)
                else:
                    if input_lengths is None:
                        input_lengths = torch.full((audio.shape[0],), audio.shape[1], device=self.device)
                    list_x = [audio[i, :input_lengths[i]].numpy() for i in range(audio.shape[0])]
            
            # Extract mel features using batch processing
            features = self.feature_extractor(
                list_x,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            input_mel = features['input_features'].to(self.device)  # (B, D, 3000)
            audio_attention_mask = features['attention_mask'].to(self.device)  # (B, 3000)
            
            # Get batch size and sequence length of the input
            mel_output_length = torch.sum(audio_attention_mask, dim=-1).long()  # (B,)
            
            # Get all layers' hidden states from encoder
            encoder_output, encoder_output_length, all_hidden_states = self.encoder(
                input_mel, mel_output_length, output_hidden_states=True
            )
            
            # Select features from specified layer
            if self.layer_id == -1:
                # Use last layer (encoder_output)
                selected_features = encoder_output  # [batch, time, d_model]
            else:
                # Use specific layer from all_hidden_states
                if self.layer_id >= len(all_hidden_states):
                    raise ValueError(f"Layer {self.layer_id} not available. Available layers: 0-{len(all_hidden_states)-1}")
                selected_features = all_hidden_states[self.layer_id]  # [batch, time, d_model]
            
            return selected_features, encoder_output_length

    def forward(self, audio, input_lengths=None):
        """
        Forward pass: extract continuous features.
        
        Arguments
        ---------
        audio : torch.Tensor or list
            Input waveform.
        input_lengths : torch.Tensor, optional
            Length of each audio sequence.
            
        Returns
        -------
        features : torch.Tensor
            Continuous features.
        output_lengths : torch.Tensor
            Length of each feature sequence.
        """
        return self.extract_features(audio, input_lengths)

    def get_feature_dim(self):
        """
        Get the feature dimension.
        
        Returns
        -------
        int
            Feature dimension.
        """
        return self.feature_dim

    def get_hop_length(self):
        """
        Get the effective hop length for the features.
        This is the original hop length multiplied by stride size.
        
        Returns
        -------
        int
            Effective hop length in samples.
        """
        return self.hop_length * self.stride_size

    def freeze_encoder(self):
        """
        Freeze encoder parameters for feature extraction only.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen.")

    def unfreeze_encoder(self):
        """
        Unfreeze encoder parameters for fine-tuning.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder parameters unfrozen.")