# -*- coding: utf-8 -*-
import os
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


from audiocodec.nn.feature_extractor import MelFeatureExtractor
from audiocodec.nn.modules import OmniAudioEncoder, OmniAudioDecoder, FrameStackDownConv, UpConv, Vocos
from audiocodec.nn.quantizer import GroupFiniteScalarQuantizer
from utils.weight_init import load_whisper_weights

class AudioCodec(nn.Module):
    def __init__(self, generator_params, initialize_whisper: bool = True):
        super().__init__()
        # Basic parameters
        self.sample_rate = generator_params['sample_rate']
        self.downsample_rate = 1280 
        
        # GroupFSQ parameters
        fsq_params = generator_params['quantizer']
        self.num_groups = fsq_params['num_groups']
        self.codebook_dim_per_group = len(fsq_params['num_levels_per_group'])
        
        ## Codec part

        ## Acoustic channel
        # Extract acoustic_encoder params and separate out the 'freeze' option
        acoustic_encoder_params = generator_params['acoustic_encoder'].copy()
        self.freeze_acoustic_encoder_flag = acoustic_encoder_params.pop('freeze', False)
        self.acoustic_encoder = OmniAudioEncoder(**acoustic_encoder_params)

        if initialize_whisper:
            self._initialize_acoustic_encoder_weights()
        else:
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                print("ℹ️  Skipping Whisper weight initialization (will be restored from checkpoint later).")

        self.downsample = FrameStackDownConv(**generator_params['downsample'])

        self.quantizer = GroupFiniteScalarQuantizer(**generator_params['quantizer'])

        ## Acoustic channel
        self.upsample = UpConv(**generator_params['upsample'])

        self.acoustic_decoder = OmniAudioDecoder(**generator_params['acoustic_decoder'])

        self.vocos = Vocos(**generator_params['vocos'])

        ## Feature extractor
        self.feature_extractor = MelFeatureExtractor(**generator_params['feature_extractor'])

    def _initialize_acoustic_encoder_weights(self):
        """Load Whisper weights and optionally freeze parameters when creating the model"""
        # (This method loads weights and may set requires_grad=False as before)
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print("🔄 Loading pretrained Whisper weights and applying freeze according to config...")
        try:
            load_whisper_weights(
                self.acoustic_encoder,
                whisper_model_name="openai/whisper-small",
                verbose=(rank == 0),
                is_acoustic=True
            )
            if rank == 0:
                print("✅ Whisper weights loaded successfully!")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Failed to load Whisper weights: {e}")

    def _apply_freezing_logic(self):
        """Freeze the acoustic encoder according to self.freeze_acoustic_encoder_flag"""
        if self.freeze_acoustic_encoder_flag:
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                print("🔒 Freezing acoustic_encoder parameters according to configuration...")
            for param in self.acoustic_encoder.parameters():
                param.requires_grad = False
            if rank == 0:
                print("✅ acoustic_encoder parameters have been frozen.")
    
    def forward(self, audio_list, audio_lengths, text_targets=None, text_lengths=None, training_stage="pretrain"):
        """
        Training forward pass

        Data flow:
        Pretrain: audio -> MEL -> acoustic encoder -> downsample -> upsample -> acoustic decoder -> Vocos -> reconstructed audio
        Posttrain: audio -> MEL -> acoustic encoder -> downsample -> FSQ quantization -> upsample -> acoustic decoder -> Vocos -> reconstructed audio

        Args:
            audio_list: unpadded list of audio waveforms [Tensor(T1,), Tensor(T2,), ...]
            audio_lengths: audio lengths (B,)
            training_stage: training stage ("pretrain" or "posttrain"), controls whether FSQ quantization is used

        Returns:
            dict: contains reconstructed audio and length information
        """
        # 1. Feature extraction (efficient: use the audio list directly)
        # Convert possible CUDA tensors to CPU numpy arrays with error handling
        cpu_audio_list = []
        for i, audio in enumerate(audio_list):
            try:
                if isinstance(audio, torch.Tensor):
                    cpu_audio_list.append(audio.cpu().numpy())
                else:
                    cpu_audio_list.append(audio)
            except Exception as e:
                raise RuntimeError(f"Failed to convert audio {i} to CPU numpy array: {e}")

        # Determine padding length dynamically based on maximum audio length in batch
        # Use actual audio_lengths to compute proper max_length
        max_audio_samples = max(len(audio) for audio in cpu_audio_list)
        # Round up to the nearest hop_length to ensure mel feature alignment
        hop_length = self.feature_extractor.hop_length
        max_length = ((max_audio_samples + hop_length - 1) // hop_length) * hop_length
        # Cap the maximum length to avoid excessively long sequences
        max_length = min(max_length, 30 * self.sample_rate)  # limit to 30 seconds

        features = self.feature_extractor(
            cpu_audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=max_length
        )
        # Get device info (from audio_lengths)
        device = audio_lengths.device
        dtype = audio_lengths.dtype
        
        input_mel = features['input_features'].to(device)  # (B, D, 3000)
        audio_attention_mask = features['attention_mask'].to(device)  # (B, 3000)
        mel_output_length = torch.sum(audio_attention_mask, dim=-1).long()  # (B,)       
        
        # Acoustic channel - obtain hidden states from all layers (B, D, T)
        acoustic_encoder_output, acoustic_encoder_output_length = self.acoustic_encoder(
            input_mel, mel_output_length
        )
        
        # 5. Downsampling (50Hz -> 12.5Hz)
        downsample_output, downsample_output_length = self.downsample(
            acoustic_encoder_output, acoustic_encoder_output_length
        )

        # apply FSQ quantization between downsampling and upsampling
        quantized_output, codes = self.quantizer(downsample_output, downsample_output_length)
        upsample_output, upsample_output_length = self.upsample(
            quantized_output, downsample_output_length
        )

        # Acoustic decoder (50Hz -> 100Hz)
        acoustic_decoder_output, acoustic_decoder_output_length = self.acoustic_decoder(
            upsample_output, upsample_output_length
        )
        
        # Vocos generator (100Hz -> 16kHz)
        reconstructed_audio, vocos_output_length = self.vocos(
            acoustic_decoder_output, acoustic_decoder_output_length
        )
        outputs = {}
        # 12. Outputs
        outputs.update({
            'reconstructed_audio': reconstructed_audio,  # (B, 1, T_audio)
            'audio_lengths': vocos_output_length
        })
        
        return outputs
    
    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        # Load model from configuration file and checkpoint
        logging.info(f"Loading model from {config_path} and {ckpt_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model instance
        model = cls(config['generator_params'])
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check if checkpoint contains 'generator' key
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        return model