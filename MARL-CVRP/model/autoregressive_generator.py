"""
Implementation of the Autoregressive Generator Model for sequence-based demand generation.
This model uses a Transformer Decoder architecture to autoregressively generate demands.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AutoregressiveGeneratorModel(nn.Module):
    """
    A Transformer Decoder-based model for generating sequences of demands.
    Uses numerically stable operations for distribution parameters.
    """
    def __init__(self, d_model=128, nhead=4, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # --- Layers ---
        # 1. Input Embedding for demand properties: (x, y, qty, arrival_time, deadline_delay)
        self.input_embedder = nn.Linear(5, d_model)
        
        # 2. Special [SOS] token embedding
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 5. Output Heads (using separate heads for each property)
        # Location (x, y): each with mean and log_std
        self.loc_head = nn.Linear(d_model, 4)
        # Quantity: mean and log_std
        self.qty_head = nn.Linear(d_model, 2)
        # Arrival Delta & Deadline Delay: each with mean and log_std
        self.time_head = nn.Linear(d_model, 4)

    def forward(self, tgt, memory=None):
        """
        Forward pass for one step of generation.
        
        Args:
            tgt (Tensor): Sequence of previously generated demand embeddings.
                          Shape: (batch_size, seq_len, d_model)
            memory (Tensor, optional): Memory from encoder for encoder-decoder attention.
                                       Not used in this model.
        
        Returns:
            Dictionary of distributions for the next demand's properties.
        """
        # Generate causal mask to prevent attending to future tokens
        seq_len = tgt.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Add positional encoding
        tgt_pos = self.pos_encoder(tgt)
        
        # Decoder forward pass
        output = self.transformer_decoder(tgt_pos, memory=memory, tgt_mask=tgt_mask)
        
        # We only care about the output for the *last* token
        last_step_output = output[:, -1:, :]
        
        # --- Get distribution parameters from heads ---
        # Location parameters (x, y)
        loc_params = self.loc_head(last_step_output)
        loc_mu = torch.sigmoid(loc_params[:, :, :2])  # Bound in [0, 1]
        loc_std = F.softplus(loc_params[:, :, 2:]) + 1e-4  # Always positive
        
        # Quantity parameters
        qty_params = self.qty_head(last_step_output)
        qty_mu = torch.sigmoid(qty_params[:, :, 0:1])  # Bound in [0, 1]
        qty_std = F.softplus(qty_params[:, :, 1:2]) + 1e-4
        
        # Time parameters (arrival_delta, deadline_delay)
        time_params = self.time_head(last_step_output)
        arrival_mu = F.softplus(time_params[:, :, 0:1]) + 0.5  # Ensure positive
        arrival_std = F.softplus(time_params[:, :, 1:2]) + 1e-4
        deadline_mu = F.softplus(time_params[:, :, 2:3]) + 5.0  # Min deadline delay is 5
        deadline_std = F.softplus(time_params[:, :, 3:4]) + 1e-4
        
        return {
            'loc_mu': loc_mu.squeeze(1),
            'loc_std': loc_std.squeeze(1),
            'qty_mu': qty_mu.squeeze(1),
            'qty_std': qty_std.squeeze(1),
            'arrival_mu': arrival_mu.squeeze(1),
            'arrival_std': arrival_std.squeeze(1),
            'deadline_mu': deadline_mu.squeeze(1),
            'deadline_std': deadline_std.squeeze(1)
        }


def generate_square_subsequent_mask(sz: int):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformers."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)