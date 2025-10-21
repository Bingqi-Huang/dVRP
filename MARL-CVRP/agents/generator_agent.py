"""
Generator Agent: Creates adversarial demand scenarios.
Uses AdversarialTransformer to decide where, how much, and when to place demands.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np

from model.adversarial_transformer import AdversarialTransformer
from utils.data_structures import GlobalState, GeneratorAction, Demand, Hotspot


class GeneratorAgent:
    """
    The adversarial agent that generates challenging demand scenarios.
    Goal: Maximize the regret (difficulty) for the Planner.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        learning_rate: float = 1e-4,
        max_capacity: float = 100.0,
        min_deadline_delay: float = 5.0,
        max_deadline_delay: float = 50.0,
        max_demands_per_step: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Optimizer learning rate
            max_capacity: Maximum vehicle capacity
            min_deadline_delay: Minimum time until deadline
            max_deadline_delay: Maximum time until deadline
            max_demands_per_step: Maximum demands to generate per timestep
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.max_demands_per_step = max_demands_per_step
        self.max_capacity = max_capacity
        self.min_deadline_delay = min_deadline_delay
        self.max_deadline_delay = max_deadline_delay
        
        # Initialize model
        self.model = AdversarialTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_capacity=max_capacity,
            min_deadline_delay=min_deadline_delay,
            max_deadline_delay=max_deadline_delay
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Episode memory for REINFORCE
        self.saved_log_probs = []
        self.rewards = []
        
        # Next demand ID counter
        self.next_demand_id = 0
    
    def _generate_hotspots(self, state: GlobalState, num_hotspots: int = 10) -> List[Hotspot]:
        """
        Generate candidate hotspot locations for placing demands.
        
        Strategy:
        - Mix of random locations
        - Locations far from current vehicles (exploitation)
        - Locations near pending demands (clustering)
        
        Args:
            state: Current global state
            num_hotspots: Number of candidate locations
            
        Returns:
            List of Hotspot objects
        """
        hotspots = []
        
        # Strategy 1: Random locations (50%)
        num_random = num_hotspots // 2
        for _ in range(num_random):
            loc = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            hotspots.append(Hotspot(location=loc))
        
        # Strategy 2: Far from vehicles (25%)
        if state.vehicles:
            num_far = num_hotspots // 4
            vehicle_locs = [v.location for v in state.vehicles]
            for _ in range(num_far):
                # Sample random location and keep if far from vehicles
                best_loc = None
                best_dist = 0
                for _ in range(5):  # Try 5 candidates
                    loc = (np.random.uniform(0, 1), np.random.uniform(0, 1))
                    min_dist = min(
                        np.sqrt((loc[0] - v[0])**2 + (loc[1] - v[1])**2)
                        for v in vehicle_locs
                    )
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_loc = loc
                if best_loc:
                    hotspots.append(Hotspot(location=best_loc))
        
        # Strategy 3: Near pending demands (25%)
        if state.pending_demands:
            num_cluster = num_hotspots - len(hotspots)
            demand_locs = [d.location for d in state.pending_demands]
            for _ in range(num_cluster):
                # Pick random demand and add noise
                base_loc = demand_locs[np.random.randint(len(demand_locs))]
                noise = np.random.normal(0, 0.1, 2)
                loc = (
                    np.clip(base_loc[0] + noise[0], 0, 1),
                    np.clip(base_loc[1] + noise[1], 0, 1)
                )
                hotspots.append(Hotspot(location=loc))
        
        # Fill remaining with random
        while len(hotspots) < num_hotspots:
            loc = (np.random.uniform(0, 1), np.random.uniform(0, 1))
            hotspots.append(Hotspot(location=loc))
        
        return hotspots[:num_hotspots]
    
    def choose_action(
        self, 
        state: GlobalState, 
        deterministic: bool = False
    ) -> Tuple[GeneratorAction, torch.Tensor]:
        """
        Generate new demands for the current timestep.
        
        Args:
            state: Current global state
            deterministic: If True, use greedy action selection
            
        Returns:
            action: GeneratorAction with list of new demands
            log_prob: Log probability of the action (for REINFORCE)
        """
        # Generate hotspot candidates
        hotspots = self._generate_hotspots(state, num_hotspots=10)
        
        # Decide how many demands to generate (random for now)
        # TODO: Could make this learnable too
        num_demands = np.random.randint(1, self.max_demands_per_step + 1)
        
        new_demands = []
        total_log_prob = torch.tensor(0.0, device=self.device)
        
        for _ in range(num_demands):
            # Sample action from model
            hotspot_idx_t, quantity_t, deadline_delay_t, log_prob_t = self.model.sample_action(
                states=[state],
                hotspots=[hotspots],
                deterministic=deterministic
            )
            
            # Extract scalar values for creating the Demand object
            hotspot_idx = hotspot_idx_t.item()
            quantity = quantity_t.item()
            deadline_delay = deadline_delay_t.item()
            
            # Create demand
            selected_hotspot = hotspots[hotspot_idx]
            demand = Demand(
                id=self.next_demand_id,
                location=selected_hotspot.location,
                quantity=quantity,
                arrival_time=state.current_time,
                deadline=state.current_time + int(deadline_delay),
                status='pending'
            )
            
            new_demands.append(demand)
            # FIXED: Accumulate the log_prob TENSOR to maintain the computation graph
            total_log_prob = total_log_prob + log_prob_t.squeeze()
            self.next_demand_id += 1
        
        action = GeneratorAction(new_demands=new_demands)
        
        # Store for REINFORCE update
        if not deterministic:
            self.saved_log_probs.append(total_log_prob)
        
        return action, total_log_prob
    
    def store_reward(self, reward: float):
        """
        Store reward for the last action (called by training loop).
        
        Args:
            reward: Reward value (typically +delta_regret for Generator)
        """
        self.rewards.append(reward)
    
    def update(self):
        """
        Update policy using REINFORCE algorithm.
        Should be called at end of episode.
        """
        if not self.saved_log_probs or not self.rewards:
            return 0.0
        
        # Convert to tensors
        rewards = torch.tensor(self.rewards, device=self.device)
        log_probs = torch.stack(self.saved_log_probs)
        
        # Compute returns (discounted cumulative rewards)
        # For simplicity, using sum of rewards (no discounting)
        # TODO: Add discount factor gamma
        returns = rewards.sum()
        
        # Normalize returns (optional, helps stability)
        if len(self.rewards) > 1:
            returns = (returns - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = -(log_probs * returns).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear episode memory
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'next_demand_id': self.next_demand_id
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.next_demand_id = checkpoint['next_demand_id']