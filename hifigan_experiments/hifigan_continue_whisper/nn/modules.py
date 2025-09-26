import torch
import numpy as np
import logging
import math
import copy
import numpy as np
import scipy
import torch
import librosa

from typing import Optional, Tuple
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN


# Define function to generate positional embeddings using sine and cosine functions to represent sequence position information
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoidal waves for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# Generate sequence mask to distinguish valid sequence and padding parts
def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    return sequence_mask

# Define RMSNorm layer for normalizing hidden states and stabilizing training process
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

# Modified variable-length attention mechanism, supporting FP32 with unified interface
class VarLenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False, dropout=0.0):
        """
        Initialize variable-length attention module.

        Parameters:
            embed_dim (int): Embedding dimension (model's hidden dimension)
            num_heads (int): Number of attention heads
            causal (bool): Whether to enable causal attention (only attend to current and previous positions)
            dropout (float): Attention dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5  # Scaling factor

        # Linear projection layers for Q, K, V and output
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _create_attention_mask(self, seq_len, max_len, device, dtype):
        """
        Create attention mask supporting variable-length sequences and causality.

        Parameters:
            seq_len (torch.Tensor): Sequence length for each sample, shape [bsz]
            max_len (int): Maximum sequence length in the batch
            device: Device for tensor creation
            dtype: Data type for mask values

        Returns:
            mask (torch.Tensor): Attention mask, shape [bsz, 1, max_len, max_len], invalid positions set to minimum value
        """
        bsz = seq_len.size(0)
        # Initialize mask as 1 (valid positions)
        mask = torch.ones(bsz, 1, max_len, max_len, device=device, dtype=dtype)

        # Generate sequence indices
        seq_indices = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
        seq_len_expanded = seq_len.unsqueeze(1)  # [bsz, 1]

        # Mark valid positions (less than seq_len)
        valid_mask = seq_indices < seq_len_expanded.unsqueeze(-1)  # [bsz, 1, max_len]
        mask = mask * (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(3)).to(dtype)  # [bsz, 1, max_len, max_len]

        # If causal attention, add upper triangular mask
        if self.causal:
            causal_mask = torch.triu(torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1)
            mask = mask * (~causal_mask.unsqueeze(0).unsqueeze(1)).to(dtype)  # Keep only lower triangular part

        # Set invalid positions (0) to dtype's minimum value
        mask = mask + (1.0 - mask) * torch.finfo(dtype).min  # Valid positions unchanged, invalid positions to minimum value
        return mask

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation, input and output are [bsz, max_len, embed_dim].

        Parameters:
            hidden_states (torch.Tensor): Input hidden states, shape [bsz, max_len, embed_dim]
            seq_len (torch.Tensor): Sequence length for each sample, shape [bsz]

        Returns:
            attn_output (torch.Tensor): Attention output, shape [bsz, max_len, embed_dim]
        """
        bsz, max_len, _ = hidden_states.size()

        # Project to Q, K, V
        query = self.q_proj(hidden_states) * self.scaling  # [bsz, max_len, embed_dim]
        key = self.k_proj(hidden_states)                  # [bsz, max_len, embed_dim]
        value = self.v_proj(hidden_states)                # [bsz, max_len, embed_dim]

        # Reshape to multi-head form
        query = query.view(bsz, max_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, max_len, head_dim]
        key = key.view(bsz, max_len, self.num_heads, self.head_dim).transpose(1, 2)      # [bsz, num_heads, max_len, head_dim]
        value = value.view(bsz, max_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, max_len, head_dim]

        # Calculate attention scores
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # [bsz, num_heads, max_len, max_len]

        # Generate attention mask
        attn_mask = self._create_attention_mask(seq_len, max_len, hidden_states.device, attn_scores.dtype)  # [bsz, 1, max_len, max_len]
        # Apply mask (additive form, consistent with HubertEncoder)
        attn_scores = attn_scores + attn_mask  # Invalid positions set to very small value

        # Softmax calculate attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bsz, num_heads, max_len, max_len]
        attn_weights = self.dropout(attn_weights)

        # Calculate attention output
        attn_output = torch.matmul(attn_weights, value)  # [bsz, num_heads, max_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, max_len, self.embed_dim)  # [bsz, max_len, embed_dim]

        # Output projection
        attn_output = self.out_proj(attn_output)  # [bsz, max_len, embed_dim]

        return attn_output

# Define Transformer layer containing attention mechanism and feedforward network for feature extraction and transformation
class OmniWhisperTransformerLayer(nn.Module):
    def __init__(self, activation_function="gelu", d_model=1280, attention_heads=20, ffn_dim=5120, causal=False, ln_type="LayerNorm", attn_type="varlen"):
        super().__init__()
        self.embed_dim = d_model
        # Only keep varlen attention mechanism
        if attn_type != "varlen":
            raise ValueError(f"Unknown attn_type: {attn_type}. Only 'varlen' is supported.")
        self.self_attn = VarLenAttention(self.embed_dim, attention_heads, causal)
        if ln_type == "LayerNorm":
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.self_attn_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        if ln_type == "LayerNorm":
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        elif ln_type == "RMSNorm":
            self.final_layer_norm = RMSNorm(self.embed_dim)
        else:
            raise ValueError(f"Unknown ln_type: {ln_type}")

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        residual = hidden_states  # [bsz, max_len, embed_dim]
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # from torch.cuda.amp import autocast
        # print(f"{residual.dtype = }")
        # print(f"Autocast enabled: {torch.is_autocast_enabled():}")
        # print(f"after layernorm {hidden_states.dtype = }")
        hidden_states = self.self_attn(hidden_states, seq_len)  # [bsz, max_len, embed_dim]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if (hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16) and \
           (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states

# Define audio encoder to convert input audio features to hidden state representation
# Acousitc encoder architecture omits sinusoidal positional encodings (Vaswani et al., 2017) and uses unactivated convolutional operations
class OmniAudioEncoder(nn.Module):
    def __init__(
            self, 
            num_mel_bins=128,  # Input feature Mel band number, usually the dimension of Mel spectrogram
            sampling_rate=16000,  # Audio sampling rate, unit Hz
            hop_length=160,  # Frame shift length (sample number) when calculating Mel spectrogram
            stride_size=2,  # Convolution layer step, used for downsampling
            kernel_size=3,  # Convolution kernel size, controlling receptive field
            d_model=1280,  # Model's hidden state dimension (embedding dimension)
            scale_embedding=True,  # Whether to scale embedding (usually used for stabilizing training)
            max_audio_seconds=30,  # Maximum audio duration supported (seconds)
            encoder_layers=32,  # Transformer encoder layer number
            encoder_attention_heads=20,  # Attention head number for each Transformer layer
            encoder_ffn_dim=5120,  # Intermediate dimension for feedforward network
            activation_function="gelu",  # Activation function type, default GELU
            attn_type="varlen",  # New parameter, select attention mechanism type
            is_acoustic=False # New parameter, select encoder type
        ):
        super().__init__()
        # Calculate maximum sequence length: Convert sampling rate to frame number after considering downsampling step
        self.max_source_positions = (max_audio_seconds * sampling_rate // hop_length) // stride_size
        # Embedding scaling factor, if enabled sqrt(d_model), otherwise 1.0
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.num_mel_bins = num_mel_bins  # Save Mel band number
        self.d_model = d_model  # Save hidden state dimension
        self.stride_size = stride_size
        self.is_acoustic = is_acoustic
        
        # First convolution layer: Convert Mel spectrogram features (num_mel_bins) to hidden dimension (d_model)
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=kernel_size, padding=1)
        # Second convolution layer: Apply downsampling with stride_size
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride_size, padding=1)
        
        # Register positional embedding buffer, using sine function to generate, shape (max_source_positions, d_model)
        self.register_buffer("positional_embedding", sinusoids(self.max_source_positions, d_model))
        
        # Create Transformer encoder layer list, each layer contains attention mechanism and feedforward network
        self.layers = nn.ModuleList([
            OmniWhisperTransformerLayer(
                activation_function=activation_function, 
                d_model=d_model, 
                attention_heads=encoder_attention_heads, 
                ffn_dim=encoder_ffn_dim, 
                causal=False,  # Encoder does not need causal attention
                attn_type=attn_type  # Pass attention type
            ) for _ in range(encoder_layers)
        ])
        
        # Last layer normalization for stable output
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_features, input_length, output_hidden_states=False):
        """
        Forward propagation function to convert input audio features to hidden state representation
        
        Parameters:
            input_features (torch.Tensor): Input Mel spectrogram features, shape [bsz, num_mel_bins, seq_len]
            input_length (torch.Tensor): Input sequence length for each sample, shape [bsz]
            output_hidden_states (bool, optional): Whether to return hidden states for each layer, default False
        
        Returns:
            if output_hidden_states is False:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, d_model, tgt_len]
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
            else:
                hidden_states (torch.Tensor): Encoded hidden states, shape [bsz, d_model, tgt_len]
                output_length (torch.Tensor): Output sequence length for each sample, shape [bsz]
                hidden_states_all_layers (tuple): Tuple containing hidden states for each layer, including initial input
        """
        # Ensure input feature data type consistent with convolution layer weights
        input_features = input_features.to(self.conv1.weight.dtype)  # (B, D, T)
        
        if not self.is_acoustic:
            # First layer convolution + GELU activation, Convert Mel spectrogram to hidden states
            inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (B, D, T)
            
            # Second layer convolution + GELU activation, Apply downsampling with stride_size
            inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (B, D, T)
        else:
            # First layer convolution without GELU activation, Convert Mel spectrogram to hidden states
            inputs_embeds = self.conv1(input_features)  # (B, D, T)
            
            # Second layer convolution without GELU activation, Apply downsampling with stride_size
            inputs_embeds = self.conv2(inputs_embeds)  # (B, D, T)
        
        # Calculate output length: Result after downsampling with stride_size
        output_length = (input_length // self.stride_size).long()  # (B,)
        
        # Adjust dimension order to [bsz, seq_len, d_model] for Transformer input
        hidden_states = inputs_embeds.permute(0, 2, 1)  # (B, T, D)
        
        # Get batch size and target sequence length
        bsz, tgt_len, _ = hidden_states.size()
        
        if not self.is_acoustic:
            # According to current sequence length, take or use complete positional embedding
            if tgt_len < self.positional_embedding.shape[0]:
                current_positional_embedding = self.positional_embedding[:tgt_len]
            else:
                current_positional_embedding = self.positional_embedding
            
            # Add input embedding to positional embedding, convert to float to avoid precision issues
            hidden_states = (hidden_states.to(torch.float32) + current_positional_embedding).to(hidden_states.dtype)
        
        # Generate sequence mask for processing variable-length sequence
        attention_mask = get_sequence_mask(hidden_states, output_length)  # [bsz, tgt_len, 1]
        
        # Initialize hidden states list for storing output for each layer (if needed)
        hidden_states_all_layers = () if output_hidden_states else None
        
        # Process hidden states through Transformer encoder layer by layer
        for encoder_layer in self.layers:
            if output_hidden_states:
                hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)
            hidden_states = encoder_layer(hidden_states, output_length)  # [bsz, tgt_len, d_model]
        
        # Normalize hidden states
        hidden_states = self.layer_norm(hidden_states)  # [bsz, tgt_len, d_model]
        if output_hidden_states:
            hidden_states_all_layers = hidden_states_all_layers + (hidden_states,)
        
        # Use mask to zero out padding parts and ensure output only retains valid data
        hidden_states = torch.where(attention_mask, hidden_states, 0)  # [bsz, tgt_len, d_model]
        
        # Apply mask and transpose to all hidden states layers if requested
        if output_hidden_states:
            # Apply mask and transpose to each layer's hidden states
            masked_hidden_states_all_layers = ()
            for layer_hidden_states in hidden_states_all_layers:
                # Apply mask to each layer
                masked_layer_states = torch.where(attention_mask, layer_hidden_states, 0)  # [bsz, tgt_len, d_model]
                masked_hidden_states_all_layers = masked_hidden_states_all_layers + (masked_layer_states,)
            hidden_states_all_layers = masked_hidden_states_all_layers
        
        if not output_hidden_states:
            return hidden_states, output_length  
        else:
            return hidden_states, output_length, hidden_states_all_layers

        