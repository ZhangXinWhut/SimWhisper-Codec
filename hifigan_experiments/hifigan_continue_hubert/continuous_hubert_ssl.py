"""
Continuous HuBERT SSL module for continuous speech feature extraction.
Uses SpeechBrain's HuBERT module directly without k-means quantization.

Authors
 * Modified for continuous features extraction with SpeechBrain HuBERT
"""

import torch
import numpy as np
from torch import nn

from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class ContinuousHubertSSL(nn.Module):
    """
    Continuous HuBERT SSL model for extracting continuous speech features.
    
    Arguments
    ---------
    save_path : str
        Path to save model cache.
    hubert_source : str
        HuBERT model source (e.g., 'facebook/hubert-base-ls960').
    layer_id : int
        Which layer to extract features from.
    device : str
        Device to use for computation.
    sample_rate : int
        Sample rate of input audio.
    """

    def __init__(
        self,
        save_path,
        hubert_source,
        layer_id=-1,
        device="cuda",
        sample_rate=16000,
    ):
        super().__init__()
        
        self.device = device
        self.layer_id = layer_id
        self.sample_rate = sample_rate
        
        # Load SpeechBrain HuBERT model
        logger.info(f"Loading SpeechBrain HuBERT model: {hubert_source}")
        self.hubert_model = HuBERT(
            source=hubert_source,
            save_path=save_path,
            output_norm=False,
            freeze=True,
            freeze_feature_extractor=True,
            apply_spec_augment=False,
            output_all_hiddens=True,
        ).to(device)
        self.hubert_model.eval()
        
        logger.info(f"ContinuousHubertSSL initialized:")
        logger.info(f"  - Model: {hubert_source}")
        logger.info(f"  - Layer: {layer_id}")
        logger.info(f"  - Device: {device}")

    def extract_features(self, wav, wav_lens=None):
        """
        Extract continuous features from SpeechBrain HuBERT model.
        
        Arguments
        ---------
        wav : torch.Tensor
            Input waveform [batch, time].
        wav_lens : torch.Tensor
            Relative lengths of waveforms.
            
        Returns
        -------
        features : torch.Tensor
            Features from the specified layer [batch, time, dim].
        """
        with torch.no_grad():
            # Extract features using SpeechBrain HuBERT
            # The output shape is [num_layers, batch, time, dim] when output_all_hiddens=True
            all_features = self.hubert_model(wav, wav_lens)
            
            # Select the target layer
            if self.layer_id >= all_features.shape[0]:
                raise ValueError(f"Layer {self.layer_id} not available. Available layers: 0-{all_features.shape[0]-1}")
            
            features = all_features[self.layer_id]  # [batch, time, dim]
                
        return features

    def forward(self, wav, wav_lens=None):
        """
        Forward pass: extract continuous features.
        
        Arguments
        ---------
        wav : torch.Tensor
            Input waveform [batch, time].
        wav_lens : torch.Tensor
            Relative lengths of waveforms.
            
        Returns
        -------
        features : torch.Tensor
            Continuous features [batch, time, dim].
        """
        return self.extract_features(wav, wav_lens)

    def encode(self, wav, wav_lens=None):
        """
        Encode waveform to continuous features.
        
        Arguments
        ---------
        wav : torch.Tensor
            Input waveform [batch, time].
        wav_lens : torch.Tensor
            Relative lengths of waveforms.
            
        Returns
        -------
        features : torch.Tensor
            Continuous features [batch, time, dim].
        """
        return self.extract_features(wav, wav_lens)