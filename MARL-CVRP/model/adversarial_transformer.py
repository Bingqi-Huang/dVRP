"""
Implementation of the Adversarial Transformer model for the Generator agent.
This model observes the full state and decides where, how much, and how urgent
the next demand should be.

COMPLETE VERSION with all fixes:
- Batch processing support
- Attention masking for variable-length sequences
- Feature normalization
- Positional and type embeddings
- Output constraints for valid actions
- Proper device handling
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from typing import List, Tuple
import math

from utils.data_structures import GlobalState, Hotspot


class AdversarialTransformer(nn.Module):
    """
    Transformer-based model for generating adversarial demand scenarios.
    
    Architecture:
    1. Tokenization: Converts state entities (vehicles, demands, hotspots) to embeddings
    2. Encoding: Multi-head self-attention to capture relationships
    3. Decoding: Separate heads output action distributions
    
    Output Actions:
    - Location: Which hotspot to place new demand (Categorical)
    - Quantity: Demand size in [0, max_capacity] (Bounded Normal)
    - Urgency: Deadline delay in [min_delay, max_delay] (Bounded Normal)
    """
    
    def __init__(
        self, 
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_capacity: float = 100.0,
        min_deadline_delay: float = 5.0,
        max_deadline_delay: float = 50.0,
        max_sequence_len: int = 200
    ):
        """
        Args:
            embed_dim: Dimension of embedding vectors
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout probability
            max_capacity: Maximum vehicle capacity (for quantity bounds)
            min_deadline_delay: Minimum time until deadline
            max_deadline_delay: Maximum time until deadline
            max_sequence_len: Maximum total tokens in sequence
        """
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.max_capacity = max_capacity
        self.min_deadline_delay = min_deadline_delay
        self.max_deadline_delay = max_deadline_delay
        self.max_sequence_len = max_sequence_len
        
        # ============================================
        # 1. SPECIAL TOKENS
        # ============================================
        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # ============================================
        # 2. FEATURE NORMALIZATION
        # ============================================
        # Normalize inputs to stabilize training
        self.vehicle_norm = nn.LayerNorm(4)   # [loc_x, loc_y, capacity, plan_len]
        self.demand_norm = nn.LayerNorm(5)    # [loc_x, loc_y, qty, time_left, status]
        self.hotspot_norm = nn.LayerNorm(2)   # [loc_x, loc_y]
        
        # ============================================
        # 3. EMBEDDING LAYERS
        # ============================================
        # Project normalized features to embedding space
        self.vehicle_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.demand_embed = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.hotspot_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # ============================================
        # 4. POSITIONAL ENCODING
        # ============================================
        # Learned positional embeddings
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_sequence_len, embed_dim)
        )
        
        # ============================================
        # 5. TYPE EMBEDDINGS
        # ============================================
        # Distinguish entity types (like BERT segment embeddings)
        # 0: CLS, 1: Vehicle, 2: Demand, 3: Hotspot
        self.type_embedding = nn.Embedding(4, embed_dim)
        
        # ============================================
        # 6. TRANSFORMER ENCODER
        # ============================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # ============================================
        # 7. OUTPUT HEADS
        # ============================================
        
        # Location Head: Scores each hotspot
        self.location_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Quantity Head: Predicts mean and log_std for demand quantity
        self.quantity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)  # [mean, log_std]
        )
        
        # Urgency Head: Predicts mean and log_std for deadline delay
        self.urgency_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)  # [mean, log_std]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _tokenize_state(
        self, 
        states: List[GlobalState], 
        hotspots: List[List[Hotspot]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert list of states into padded token sequences.
        
        Returns:
            tokens: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] - True for real tokens, False for padding
            hotspot_mask: [batch_size, max_hotspots] - Mask for hotspot tokens only
        """
        batch_size = len(states)
        device = self.cls_token.device
        
        all_tokens = []
        all_masks = []
        all_hotspot_masks = []
        
        for state, hs_list in zip(states, hotspots):
            # ====================================
            # Process Vehicles
            # ====================================
            vehicle_features = torch.tensor(
                [
                    [
                        v.location[0], 
                        v.location[1], 
                        v.remaining_capacity / self.max_capacity,  # Normalize to [0,1]
                        min(len(v.current_plan) / 10.0, 1.0)  # Normalize, cap at 1
                    ] 
                    for v in state.vehicles
                ],
                dtype=torch.float32,
                device=device
            )
            vehicle_features = self.vehicle_norm(vehicle_features)
            vehicle_tokens = self.vehicle_embed(vehicle_features)
            num_vehicles = vehicle_tokens.size(0)
            
            # ====================================
            # Process Demands
            # ====================================
            if state.pending_demands:
                demand_features = torch.tensor(
                    [
                        [
                            d.location[0],
                            d.location[1],
                            d.quantity / self.max_capacity,  # Normalize
                            max(d.deadline - state.current_time, 0) / self.max_deadline_delay,  # Normalize
                            1.0 if d.status == 'pending' else 0.0
                        ]
                        for d in state.pending_demands
                    ],
                    dtype=torch.float32,
                    device=device
                )
                demand_features = self.demand_norm(demand_features)
                demand_tokens = self.demand_embed(demand_features)
                num_demands = demand_tokens.size(0)
            else:
                demand_tokens = torch.empty(0, self.embed_dim, device=device)
                num_demands = 0
            
            # ====================================
            # Process Hotspots
            # ====================================
            hotspot_features = torch.tensor(
                [[h.location[0], h.location[1]] for h in hs_list],
                dtype=torch.float32,
                device=device
            )
            hotspot_features = self.hotspot_norm(hotspot_features)
            hotspot_tokens = self.hotspot_embed(hotspot_features)
            num_hotspots = hotspot_tokens.size(0)
            
            # ====================================
            # Concatenate with CLS Token
            # ====================================
            cls_token = self.cls_token.squeeze(0)  # [1, embed_dim]
            
            sequence = torch.cat([
                cls_token,
                vehicle_tokens,
                demand_tokens,
                hotspot_tokens
            ], dim=0)  # [seq_len, embed_dim]
            
            # ====================================
            # Create Masks
            # ====================================
            seq_len = sequence.size(0)
            mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            
            # Hotspot mask (for location head)
            hotspot_mask = torch.ones(num_hotspots, dtype=torch.bool, device=device)
            
            all_tokens.append(sequence)
            all_masks.append(mask)
            all_hotspot_masks.append(hotspot_mask)
        
        # ====================================
        # Pad to Same Length
        # ====================================
        max_len = max(t.size(0) for t in all_tokens)
        max_len = min(max_len, self.max_sequence_len)
        
        padded_tokens = torch.zeros(
            batch_size, max_len, self.embed_dim,
            device=device
        )
        padded_masks = torch.zeros(
            batch_size, max_len,
            dtype=torch.bool,
            device=device
        )
        
        for i, (tokens, mask) in enumerate(zip(all_tokens, all_masks)):
            length = min(tokens.size(0), max_len)
            padded_tokens[i, :length] = tokens[:length]
            padded_masks[i, :length] = mask[:length]
        
        # Pad hotspot masks
        max_hotspots = max(m.size(0) for m in all_hotspot_masks)
        padded_hotspot_masks = torch.zeros(
            batch_size, max_hotspots,
            dtype=torch.bool,
            device=device
        )
        for i, mask in enumerate(all_hotspot_masks):
            padded_hotspot_masks[i, :mask.size(0)] = mask
        
        return padded_tokens, padded_masks, padded_hotspot_masks
    
    def _add_positional_and_type_embeddings(
        self,
        tokens: torch.Tensor,
        states: List[GlobalState],
        hotspots: List[List[Hotspot]]
    ) -> torch.Tensor:
        """
        Add positional and type embeddings to tokens.
        
        Args:
            tokens: [batch_size, seq_len, embed_dim]
            states: List of GlobalState objects
            hotspots: List of hotspot lists
            
        Returns:
            tokens with added embeddings
        """
        batch_size, seq_len, _ = tokens.shape
        device = tokens.device
        
        # ====================================
        # Positional Encoding
        # ====================================
        pos_embed = self.pos_encoding[:, :seq_len, :]
        tokens = tokens + pos_embed
        
        # ====================================
        # Type Embeddings
        # ====================================
        # Create type IDs for each token
        # 0: CLS, 1: Vehicle, 2: Demand, 3: Hotspot
        type_ids_list = []
        
        for state, hs_list in zip(states, hotspots):
            num_vehicles = len(state.vehicles)
            num_demands = len(state.pending_demands)
            num_hotspots = len(hs_list)
            
            type_ids = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),  # CLS
                torch.ones(num_vehicles, dtype=torch.long, device=device),  # Vehicles
                torch.full((num_demands,), 2, dtype=torch.long, device=device),  # Demands
                torch.full((num_hotspots,), 3, dtype=torch.long, device=device),  # Hotspots
            ])
            
            # Pad to seq_len
            if type_ids.size(0) < seq_len:
                padding = torch.zeros(
                    seq_len - type_ids.size(0),
                    dtype=torch.long,
                    device=device
                )
                type_ids = torch.cat([type_ids, padding])
            else:
                type_ids = type_ids[:seq_len]
            
            type_ids_list.append(type_ids)
        
        type_ids = torch.stack(type_ids_list)  # [batch_size, seq_len]
        type_embed = self.type_embedding(type_ids)
        tokens = tokens + type_embed
        
        return tokens
    
    def forward(
        self,
        states: List[GlobalState],
        hotspots: List[List[Hotspot]]
    ) -> Tuple[Categorical, TransformedDistribution, TransformedDistribution]:
        """
        Forward pass through the model.
        
        Args:
            states: List of GlobalState objects (batch)
            hotspots: List of hotspot lists for each state
            
        Returns:
            loc_dist: Categorical distribution over hotspots [batch_size, num_hotspots]
            qty_dist: Bounded Normal distribution for quantity [batch_size]
            urgency_dist: Bounded Normal distribution for deadline [batch_size]
        """
        batch_size = len(states)
        device = self.cls_token.device
        
        # ====================================
        # 1. TOKENIZATION
        # ====================================
        tokens, mask, hotspot_mask = self._tokenize_state(states, hotspots)
        # tokens: [batch_size, seq_len, embed_dim]
        # mask: [batch_size, seq_len]
        
        # ====================================
        # 2. ADD POSITIONAL & TYPE EMBEDDINGS
        # ====================================
        tokens = self._add_positional_and_type_embeddings(tokens, states, hotspots)
        
        # ====================================
        # 3. TRANSFORMER ENCODING
        # ====================================
        # Create attention mask (True = padding, False = real token)
        # PyTorch uses inverted convention
        padding_mask = ~mask
        
        encoded = self.transformer_encoder(
            tokens,
            src_key_padding_mask=padding_mask
        )
        # encoded: [batch_size, seq_len, embed_dim]
        
        # ====================================
        # 4. EXTRACT REPRESENTATIONS
        # ====================================
        # CLS token for global decisions (quantity, urgency)
        cls_output = encoded[:, 0, :]  # [batch_size, embed_dim]
        
        # Hotspot tokens for location decision
        # Find hotspot indices dynamically
        hotspot_outputs_list = []
        for i, (state, hs_list) in enumerate(zip(states, hotspots)):
            num_vehicles = len(state.vehicles)
            num_demands = len(state.pending_demands)
            hotspot_start = 1 + num_vehicles + num_demands
            hotspot_end = hotspot_start + len(hs_list)
            
            hotspot_outputs = encoded[i, hotspot_start:hotspot_end, :]
            hotspot_outputs_list.append(hotspot_outputs)
        
        # Pad hotspot outputs
        max_hotspots = max(h.size(0) for h in hotspot_outputs_list)
        padded_hotspot_outputs = torch.zeros(
            batch_size, max_hotspots, self.embed_dim,
            device=device
        )
        for i, h in enumerate(hotspot_outputs_list):
            padded_hotspot_outputs[i, :h.size(0), :] = h
        
        # ====================================
        # 5. LOCATION HEAD (Categorical)
        # ====================================
        loc_logits = self.location_head(padded_hotspot_outputs).squeeze(-1)
        # loc_logits: [batch_size, max_hotspots]
        
        # Mask invalid hotspots
        loc_logits = loc_logits.masked_fill(~hotspot_mask, float('-inf'))
        
        loc_dist = Categorical(logits=loc_logits)
        
        # ====================================
        # 6. QUANTITY HEAD (Bounded Normal)
        # ====================================
        qty_params = self.quantity_head(cls_output)
        qty_mean_raw, qty_log_std = qty_params.chunk(2, dim=-1)
        
        # Use sigmoid to bound mean to [0, 1], then scale to [0, max_capacity]
        qty_mean = torch.sigmoid(qty_mean_raw) * self.max_capacity
        qty_std = torch.exp(qty_log_std.clamp(-5, 2))  # Clamp for stability
        
        # Create bounded distribution using transforms
        base_dist = Normal(qty_mean.squeeze(-1), qty_std.squeeze(-1))
        # Add small epsilon to avoid boundary issues
        qty_dist = self._create_bounded_normal(
            base_dist, 
            low=0.1,  # Minimum demand
            high=self.max_capacity
        )
        
        # ====================================
        # 7. URGENCY HEAD (Bounded Normal)
        # ====================================
        urgency_params = self.urgency_head(cls_output)
        urgency_mean_raw, urgency_log_std = urgency_params.chunk(2, dim=-1)
        
        # Map to [min_delay, max_delay]
        deadline_range = self.max_deadline_delay - self.min_deadline_delay
        urgency_mean = (
            torch.sigmoid(urgency_mean_raw) * deadline_range + self.min_deadline_delay
        )
        urgency_std = torch.exp(urgency_log_std.clamp(-5, 2))
        
        base_dist = Normal(urgency_mean.squeeze(-1), urgency_std.squeeze(-1))
        urgency_dist = self._create_bounded_normal(
            base_dist,
            low=self.min_deadline_delay,
            high=self.max_deadline_delay
        )
        
        return loc_dist, qty_dist, urgency_dist
    
    def _create_bounded_normal(
        self,
        base_dist: Normal,
        low: float,
        high: float
    ) -> TransformedDistribution:
        """
        Create a bounded normal distribution using sigmoid + affine transform.
        
        This ensures samples are always in [low, high] range.
        """
        # Transform: x -> sigmoid(x) -> scale to [low, high]
        transforms = [
            SigmoidTransform(),
            AffineTransform(loc=low, scale=high - low)
        ]
        return TransformedDistribution(base_dist, transforms)
    
    def sample_action(
        self,
        states: List[GlobalState],
        hotspots: List[List[Hotspot]],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the model's output distributions.
        
        Args:
            states: List of states
            hotspots: List of hotspot lists
            deterministic: If True, use mode/mean instead of sampling
            
        Returns:
            hotspot_indices: [batch_size] - Selected hotspot for each state
            quantities: [batch_size] - Demand quantities
            deadlines: [batch_size] - Deadline delays
            log_probs: [batch_size] - Total log probability of actions
        """
        loc_dist, qty_dist, urgency_dist = self.forward(states, hotspots)
        
        if deterministic:
            # Use mode for categorical, mean for continuous
            hotspot_indices = loc_dist.probs.argmax(dim=-1)
            quantities = qty_dist.mean
            deadlines = urgency_dist.mean
        else:
            # Sample from distributions
            hotspot_indices = loc_dist.sample()
            quantities = qty_dist.sample()
            deadlines = urgency_dist.sample()
        
        # Calculate total log probability
        log_probs = (
            loc_dist.log_prob(hotspot_indices) +
            qty_dist.log_prob(quantities) +
            urgency_dist.log_prob(deadlines)
        )
        
        return hotspot_indices, quantities, deadlines, log_probs


# ============================================
# UTILITY FUNCTIONS FOR TESTING
# ============================================

def test_model():
    """Test the model with dummy data."""
    from utils.data_structures import Vehicle, Demand
    
    print("Testing AdversarialTransformer...")
    
    # Create model
    model = AdversarialTransformer(
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        max_capacity=100.0
    )
    model.eval()
    
    # Create dummy states
    states = [
        GlobalState(
            current_time=5,
            vehicles=[
                Vehicle(id=0, location=(0.5, 0.5), remaining_capacity=80.0, current_plan=[]),
                Vehicle(id=1, location=(0.3, 0.7), remaining_capacity=50.0, current_plan=[1])
            ],
            pending_demands=[
                Demand(
                    id=1,
                    location=(0.8, 0.2),
                    quantity=20.0,
                    arrival_time=3,
                    deadline=15,
                    status='pending'
                )
            ],
            serviced_demands=[],
            failed_demands=[]
        ),
        GlobalState(
            current_time=10,
            vehicles=[
                Vehicle(id=0, location=(0.1, 0.9), remaining_capacity=100.0, current_plan=[])
            ],
            pending_demands=[],
            serviced_demands=[],
            failed_demands=[]
        )
    ]
    
    hotspots = [
        [
            Hotspot(location=(0.2, 0.3)),
            Hotspot(location=(0.7, 0.8)),
            Hotspot(location=(0.9, 0.1))
        ],
        [
            Hotspot(location=(0.5, 0.5)),
            Hotspot(location=(0.1, 0.1))
        ]
    ]
    
    # Forward pass
    with torch.no_grad():
        loc_dist, qty_dist, urgency_dist = model(states, hotspots)
        
        print(f"\nLocation distribution shape: {loc_dist.logits.shape}")
        print(f"Location probs (state 0): {loc_dist.probs[0]}")
        
        print(f"\nQuantity distribution mean: {qty_dist.mean}")
        print(f"Quantity distribution std: {qty_dist.stddev}")
        
        print(f"\nUrgency distribution mean: {urgency_dist.mean}")
        print(f"Urgency distribution std: {urgency_dist.stddev}")
        
        # Sample actions
        indices, qtys, deadlines, log_probs = model.sample_action(states, hotspots)
        print(f"\nSampled actions:")
        print(f"  Hotspot indices: {indices}")
        print(f"  Quantities: {qtys}")
        print(f"  Deadlines: {deadlines}")
        print(f"  Log probs: {log_probs}")
    
    print("\n✅ Model test passed!")


if __name__ == "__main__":
    test_model()