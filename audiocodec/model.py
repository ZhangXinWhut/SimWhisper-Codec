# -*- coding: utf-8 -*-
import os
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


from audiocodec.nn.feature_extractor import MelFeatureExtractor
from audiocodec.nn.modules import OmniAudioEncoder, OmniAudioDecoder, FrameStackDownConv, FrameStackUpConv, Vocos
from audiocodec.nn.quantizer import GroupFiniteScalarQuantizer
from utils.weight_init import load_whisper_weights

class AudioCodec(nn.Module):
    def __init__(self, generator_params):
        super().__init__()
        # Basic parameters
        self.input_sample_rate = generator_params['input_sample_rate']
        self.output_sample_rate = generator_params['output_sample_rate']
        self.max_audio_seconds = 30
        self.encoder_downsample_rate = generator_params['encoder_downsample_rate']
        self.decoder_upsample_rate = generator_params['decoder_upsample_rate']
        

        # GroupFSQ parameters
        fsq_params = generator_params['quantizer']
        self.num_groups = fsq_params['num_groups']
        self.codebook_dim_per_group = len(fsq_params['num_levels_per_group'])
        
        ## Codec part

        ## Acoustic channel
        # Extract acoustic_encoder params and separate out the 'freeze' option
        acoustic_encoder_params = generator_params['acoustic_encoder'].copy()
        self.freeze_acoustic_encoder_flag = acoustic_encoder_params.pop('freeze', False)
        # 提取 Whisper 初始化相关配置
        self.whisper_model_path = acoustic_encoder_params.pop('whisper_model_path', None)
        self.init_from_whisper = acoustic_encoder_params.pop('init_from_whisper', False)
        self.acoustic_encoder = OmniAudioEncoder(**acoustic_encoder_params)

        self.acoustic_encoder = OmniAudioEncoder(**acoustic_encoder_params)

        self.downsample = FrameStackDownConv(**generator_params['downsample'])

        self.quantizer = GroupFiniteScalarQuantizer(**generator_params['quantizer'])

        ## Acoustic channel
        # self.upsample = UpConv(**generator_params['upsample'])
        self.upsample = FrameStackUpConv(**generator_params['upsample'])

        self.acoustic_decoder = OmniAudioDecoder(**generator_params['acoustic_decoder'])

        self.vocos = Vocos(**generator_params['vocos'])

        ## Feature extractor
        self.feature_extractor = MelFeatureExtractor(**generator_params['feature_extractor'])
    
    def _init_whisper_weights(self):
        """
        从预训练的 Whisper 模型加载 encoder 权重
        
        注意：此方法应该在 _apply_freezing_logic() 之前调用
        """
        if not self.init_from_whisper:
            return
        
        if self.whisper_model_path is None:
            logging.warning("init_from_whisper=True 但未指定 whisper_model_path，跳过权重初始化")
            return
        
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            logging.info(f"🔄 正在从 Whisper 模型加载 encoder 权重: {self.whisper_model_path}")
        
        try:
            # 调用权重加载函数
            load_whisper_weights(
                encoder=self.acoustic_encoder,
                whisper_model_name=self.whisper_model_path,
                verbose=(rank == 0),
                is_acoustic=True
            )
            if rank == 0:
                logging.info("✅ Whisper encoder 权重初始化完成")
        except Exception as e:
            logging.error(f"❌ Whisper 权重加载失败: {e}")
            raise

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
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
    
    def forward(self, batch):
        """
        Training forward pass

        Data flow:
        mel_features -> acoustic encoder -> downsample -> FSQ quantization -> upsample -> acoustic decoder -> Vocos -> reconstructed audio

        Args:
            mel_features: Mel features (B, D, T)
            mel_lengths: Mel feature lengths (B,)

        Returns:
            dict: contains reconstructed audio and length information
        """
        # 获取设备信息
        device = batch['mel_features'].device

        # 1. Mel特征（已经由Dataset提取好了）
        input_mel = batch['mel_features']  # (B, n_mels, T) - 已经填充到批次最大长度
        mel_lens = batch['mel_lens']  # (B,) - 实际Mel特征长度      
        
        # Acoustic channel - obtain hidden states from all layers (B, D, T)
        acoustic_encoder_output, acoustic_encoder_output_length = self.acoustic_encoder(
            input_mel, mel_lens
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
    
    @torch.inference_mode()
    def inference_tokenize(self, x, input_lengths):
        """
            Input:
                x: Waveform tensor # (B, 1, T), T <= 30s * sample_rate
                input_lengths: Valid length for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "zq": Quantized embeddings # (B, D, T)
                    "codes": Quantization codes # (num_groups, B, T)
                    "codes_lengths": Quantization code lengths # (B,)
        """
        # 1. Extract mel features
        list_x = [xi[:, :x_len].reshape(-1).cpu().numpy() for xi, x_len in zip(x, input_lengths)]
        features = self.feature_extractor(
            list_x,
            sampling_rate=self.input_sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_mel = features['input_features'].to(x.device).to(x.dtype) # (B, n_mels, T_mel)
        audio_attention_mask = features['attention_mask'].to(x.device) # (B, T_mel)
        
        # Get mel feature lengths
        mel_lens = torch.sum(audio_attention_mask, dim=-1).long() # (B,)
        
        # 2. Acoustic encoder
        acoustic_encoder_output, acoustic_encoder_output_length = self.acoustic_encoder(
            input_mel, mel_lens
        )
        
        # 3. Downsample
        downsample_output, downsample_output_length = self.downsample(
            acoustic_encoder_output, acoustic_encoder_output_length
        )
        
        # 4. FSQ quantization
        quantized_output, codes = self.quantizer(downsample_output, downsample_output_length)

        return {
            "zq": quantized_output, # (B, D, T)
            "codes": codes, # (num_groups, B, T)
            "codes_lengths": downsample_output_length # (B,)
        }
      
    @torch.inference_mode()  
    def inference_detokenize(self, codes, codes_lengths):
        """
            Input:
                codes: Quantization codes # (num_groups, B, T)
                codes_lengths: Quantization code lengths for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "y": Synthesized audio waveform # (B, 1, T)
                    "output_length": Output lengths # (B,)
        """
        # 1. Decode FSQ codes to quantized embeddings
        zq = self.quantizer.decode(codes, codes_lengths) # (B, D, T)

        # 2. Upsample
        upsample_output, upsample_output_length = self.upsample(
            zq, codes_lengths
        )

        # 3. Acoustic decoder
        acoustic_decoder_output, acoustic_decoder_output_length = self.acoustic_decoder(
            upsample_output, upsample_output_length
        )

        # 4. Vocos generator
        y, vocos_output_length = self.vocos(acoustic_decoder_output, acoustic_decoder_output_length) # (B, 1, T)
        
        return {
            "y": y, # (B, 1, T)
            "output_length": vocos_output_length, # (B,)
        }
        
    @torch.inference_mode()
    def encode(self, wav_list, overlap_seconds=10, device=torch.device("cuda")):
        """
            Input:
                wav_list: List of audio waveforms, each with potentially different length, may exceed max_audio_seconds # B * (T,)
                overlap_seconds: Overlap in seconds, process max_audio_seconds at a time, keeping (max_audio_seconds - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "codes_list": List of quantization codes # B * (num_groups, T)
        """
        duration_seconds = self.max_audio_seconds - overlap_seconds
        chunk_size = int(self.max_audio_seconds * self.input_sample_rate) # Maximum samples per chunk
        duration_size = int(duration_seconds * self.input_sample_rate) # Valid output samples per chunk
        # encoder_downsample_rate 应该计算从原始音频到量化码的总下采样倍率
        # mel: 16000/160=100Hz, encoder+downsample: 100Hz -> 12.5Hz (下采样8倍), 所以总倍率是 160*8=1280
        code_duration_length = duration_size // self.encoder_downsample_rate # Valid code length per chunk

        # Get maximum waveform length
        max_length = max(len(wav) for wav in wav_list)
        batch_size = len(wav_list)
        wav_tensor = torch.zeros(batch_size, 1, max_length, device=device)
        input_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, wav in enumerate(wav_list):
            wav_tensor[i, 0, :len(wav)] = wav
            input_lengths[i] = len(wav) # (B,)

        # Calculate number of chunks needed
        max_chunks = (max_length + duration_size - 1) // duration_size
        codes_list = []

        # Process the entire batch in chunks
        for chunk_idx in range(max_chunks):
            start = chunk_idx * duration_size
            end = min(start + chunk_size, max_length)
            chunk = wav_tensor[:, :, start:end] # (B, 1, T')
            chunk_lengths = torch.clamp(input_lengths - start, 0, end - start) # (B,)

            # Skip empty chunks
            if chunk_lengths.max() == 0:
                continue

            # Encode
            result = self.inference_tokenize(chunk, chunk_lengths) # {"zq": (B, D, T'), "codes": (nq, B, T'), "codes_lengths": (B,)}
            chunk_codes = result["codes"] # (nq, B, T')
            chunk_code_lengths = result["codes_lengths"] # (B,)

            # Extract valid portion
            valid_code_lengths = torch.clamp(chunk_code_lengths, 0, code_duration_length) # (B,)
            valid_chunk_codes = torch.zeros(self.num_groups, batch_size, code_duration_length, device=device, dtype=chunk_codes.dtype)
            for b in range(batch_size):
                if valid_code_lengths[b] > 0:
                    valid_chunk_codes[:, b, :valid_code_lengths[b]] = chunk_codes[:, b, :valid_code_lengths[b]] # (num_groups, B, valid_code_length)

            codes_list.append(valid_chunk_codes) # (num_groups, B, valid_code_length)

        # Concatenate all chunks
        if codes_list:
            codes_tensor = torch.cat(codes_list, dim=-1) # (num_groups, B, T_total)
            codes_list = [codes_tensor[:, i, :input_lengths[i] // self.encoder_downsample_rate] for i in range(batch_size)] # B * (num_groups, T)
        else:
            codes_list = [torch.zeros(self.num_groups, 0, device=device, dtype=torch.long) for _ in range(batch_size)] # B * (num_groups, 0)

        return {
            "codes_list": codes_list # B * (num_groups, T)
        }
        
    @torch.inference_mode()
    def decode(self, codes_list, overlap_seconds=10, device=torch.device("cuda")):
        """
            Input:
                codes_list: List of quantization codes # B * (num_groups, T)
                overlap_seconds: Overlap in seconds, process max_audio_seconds at a time, keeping (max_audio_seconds - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "syn_wav_list": List of synthesized audio waveforms # B * (T,)
        """
        duration_seconds = self.max_audio_seconds - overlap_seconds
        chunk_code_length = int(self.max_audio_seconds * self.input_sample_rate // self.encoder_downsample_rate) # Maximum code length per chunk
        duration_code_length = int(duration_seconds * self.input_sample_rate // self.encoder_downsample_rate) # Valid code length per chunk
        # decoder_upsample_rate 应该等于 encoder_downsample_rate，从量化码到输出音频
        duration_wav_length = duration_code_length * self.decoder_upsample_rate # Valid waveform length per chunk

        # Get maximum code length
        max_code_length = max(codes.shape[-1] for codes in codes_list)
        batch_size = len(codes_list)
        codes_tensor = torch.zeros(self.num_groups, batch_size, max_code_length, device=device, dtype=torch.long)
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, codes in enumerate(codes_list):
            codes_tensor[:, i, :codes.shape[-1]] = codes.to(device)
            code_lengths[i] = codes.shape[-1] # (B,)

        # Calculate number of chunks needed
        max_chunks = (max_code_length + duration_code_length - 1) // duration_code_length
        wav_list = []

        # Process the entire batch in chunks
        for chunk_idx in range(max_chunks):
            start = chunk_idx * duration_code_length
            end = min(start + chunk_code_length, max_code_length)
            chunk_codes = codes_tensor[:, :, start:end] # (num_groups, B, T')
            chunk_code_lengths = torch.clamp(code_lengths - start, 0, end - start) # (B,)

            # Skip empty chunks
            if chunk_code_lengths.max() == 0:
                continue

            # Decode
            result = self.inference_detokenize(chunk_codes, chunk_code_lengths) # {"y": (B, 1, T'), "output_length": (B,)}
            chunk_wav = result["y"] # (B, 1, T')
            chunk_wav_lengths = result["output_length"] # (B,)

            # Extract valid portion
            valid_wav_lengths = torch.clamp(chunk_wav_lengths, 0, duration_wav_length) # (B,)
            valid_chunk_wav = torch.zeros(batch_size, 1, duration_wav_length, device=device)
            for b in range(batch_size):
                if valid_wav_lengths[b] > 0:
                    valid_chunk_wav[b, :, :valid_wav_lengths[b]] = chunk_wav[b, :, :valid_wav_lengths[b]] # (B, 1, valid_wav_length)

            wav_list.append(valid_chunk_wav) # (B, 1, valid_wav_length)

        # Concatenate all chunks
        if wav_list:
            wav_tensor = torch.cat(wav_list, dim=-1) # (B, 1, T_total)
            syn_wav_list = [wav_tensor[i, 0, :code_lengths[i] * self.decoder_upsample_rate] for i in range(batch_size)] # B * (T,)
        else:
            syn_wav_list = [torch.zeros(0, device=device) for _ in range(batch_size)] # B * (0,)
            
        return {
            "syn_wav_list": syn_wav_list # B * (T,)
        }
    
    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        # Load model from configuration file and checkpoint
        logging.info(f"Loading model from {config_path} and {ckpt_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model instance (skip Whisper initialization since we'll load from checkpoint)
        model = cls(config['generator_params'])
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check if checkpoint contains 'model' key
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        return model