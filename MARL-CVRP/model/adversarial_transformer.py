"""
Implementation of the Adversarial Transformer model for the Generator agent.
This model observes the full state and decides where, how much, and how urgent
the next demand should be.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from utils.data_structures import GlobalState, Hotspot
from typing import List

class AdversarialTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=3):
        """
        Initializes the Transformer model components.
        - Embedders for different entity types (vehicle, demand, hotspot).
        - Transformer Encoder to process the sequence of entities.
        - Output heads to produce parameters for action distributions.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Embedding layers for different entity types
        self.vehicle_embed = nn.Linear(4, embed_dim) # Example: [loc_x, loc_y, capacity, plan_len]
        self.demand_embed = nn.Linear(5, embed_dim) # Example: [loc_x, loc_y, qty, deadline, status]
        self.hotspot_embed = nn.Linear(2, embed_dim) # Example: [loc_x, loc_y]
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action Heads
        # 1. Location Head: Chooses a hotspot
        self.location_head = nn.Linear(embed_dim, 1)
        
        # 2. Quantity Head: Predicts mean and std for demand quantity
        self.quantity_head = nn.Linear(embed_dim, 2)

        # 3. Urgency Head: Predicts mean and std for deadline delay
        self.urgency_head = nn.Linear(embed_dim, 2)

    def forward(self, state: GlobalState, hotspots: List[Hotspot]):
        """
        Processes the environment state to produce action distributions.
        1. Tokenization: Convert all entities (vehicles, demands, hotspots) into a sequence of embedding vectors.
        2. Encoding: Pass the sequence through the Transformer Encoder.
        3. Decoding: Use the output embeddings to feed the action heads.
        
        Returns:
            A tuple of torch.distributions: (loc_dist, qty_dist, urgency_dist)
        """
        # --- 1. Tokenization ---
        # This is a simplified example. A real implementation would need to handle
        # variable numbers of entities via padding or other mechanisms.
        # Also, converting dataclasses to tensors needs to be implemented.
        
        # vehicle_tokens = ... # Convert state.vehicles to tensor and pass through self.vehicle_embed
        # demand_tokens = ... # Convert state.pending_demands to tensor and pass through self.demand_embed
        # hotspot_tokens = ... # Convert hotspots to tensor and pass through self.hotspot_embed
        
        # Dummy tensors for shape demonstration
        batch_size = 1 # Assuming batch processing
        vehicle_tokens = torch.randn(batch_size, len(state.vehicles), self.embed_dim)
        demand_tokens = torch.randn(batch_size, len(state.pending_demands), self.embed_dim)
        hotspot_tokens = torch.randn(batch_size, len(hotspots), self.embed_dim)
        
        # Prepend CLS token and concatenate all tokens
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        input_sequence = torch.cat([cls_token, vehicle_tokens, demand_tokens, hotspot_tokens], dim=1)

        # --- 2. Encoding ---
        output_sequence = self.transformer_encoder(input_sequence)

        # --- 3. Decoding ---
        # Separate the encoded tokens
        cls_output = output_sequence[:, 0]
        # The start index for hotspot outputs depends on the number of other tokens
        hotspot_start_idx = 1 + vehicle_tokens.size(1) + demand_tokens.size(1)
        encoded_hotspots = output_sequence[:, hotspot_start_idx:]

        # Location Head -> Categorical distribution over hotspots
        loc_logits = self.location_head(encoded_hotspots).squeeze(-1)
        loc_dist = Categorical(logits=loc_logits)

        # Quantity Head -> Normal distribution for quantity
        qty_params = self.quantity_head(cls_output)
        qty_mu, qty_sigma = qty_params.chunk(2, dim=-1)
        qty_dist = Normal(qty_mu.squeeze(), torch.exp(qty_sigma).squeeze()) # Use exp for positive std

        # Urgency Head -> Normal distribution for deadline delay
        urgency_params = self.urgency_head(cls_output)
        urgency_mu, urgency_sigma = urgency_params.chunk(2, dim=-1)
        urgency_dist = Normal(urgency_mu.squeeze(), torch.exp(urgency_sigma).squeeze())

        return loc_dist, qty_dist, urgency_dist