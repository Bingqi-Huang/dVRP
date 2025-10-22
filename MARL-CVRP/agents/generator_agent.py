"""
Generator Agent: Creates adversarial demand scenarios.
In this refactored version, it generates a complete instance offline using
an autoregressive neural network model.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
import random

from utils.data_structures import Demand
from model.autoregressive_generator import AutoregressiveGeneratorModel


class GeneratorAgent:
    """
    The adversarial agent that generates challenging demand scenarios.
    This version uses a learned autoregressive model for generation.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_decoder_layers: int = 2,
        learning_rate: float = 1e-4,
        min_deadline_delay: float = 5.0,
        max_deadline_delay: float = 50.0,
        baseline_alpha: float = 0.9,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.min_deadline_delay = min_deadline_delay
        self.max_deadline_delay = max_deadline_delay
        self.next_demand_id = 0
        
        # Initialize the autoregressive model
        print(f"Initializing model on device: {device}")
        self.model = AutoregressiveGeneratorModel(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Baseline for REINFORCE
        self.reward_baseline = 0.0
        self.baseline_alpha = baseline_alpha
        
        # Store for REINFORCE updates
        self.saved_log_probs = []

    def generate_instance(
        self, 
        num_demands: int, 
        episode_length: int, 
        max_quantity: float, 
        map_size: Tuple[float, float] = (1.0, 1.0),
        random_seed: int = None
    ) -> List[Demand]:
        """
        Generates a complete instance using the autoregressive model.
        Falls back to random generation for testing or if the model fails.
        
        Args:
            num_demands: Number of demands to generate
            episode_length: Maximum episode length
            max_quantity: Maximum demand quantity
            map_size: Environment dimensions
            random_seed: Optional seed for reproducibility
        """
        # Clear the saved log probs from any previous instance
        self.saved_log_probs = []
        
        # If a random seed is provided, set it
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
        
        try:
            return self._generate_with_model(num_demands, episode_length, max_quantity, map_size)
        except Exception as e:
            print(f"Neural model generation failed with: {e}")
            print("Falling back to random generation...")
            return self._generate_randomly(num_demands, episode_length, max_quantity, map_size)

    def _generate_with_model(
        self, 
        num_demands: int, 
        episode_length: int, 
        max_quantity: float, 
        map_size: Tuple[float, float]
    ) -> List[Demand]:
        """Generate using the neural model with careful numerical handling."""
        demand_schedule = []
        
        # Start with the [SOS] token
        current_sequence = self.model.sos_token.clone().to(self.device)
        
        # Track cumulative arrival time
        last_arrival_time = 0.0
        
        # Enable evaluation mode during generation
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_demands):
                # Get distribution parameters for the next demand
                outputs = self.model(current_sequence)
                
                # --- Sample values using the normal distribution ---
                # Location (x, y)
                loc_dist = torch.distributions.Normal(
                    outputs['loc_mu'], outputs['loc_std']
                )
                loc = loc_dist.sample().clamp(0.0, 1.0)
                
                # Quantity
                qty_dist = torch.distributions.Normal(
                    outputs['qty_mu'], outputs['qty_std']
                )
                qty = qty_dist.sample().clamp(0.0, 1.0)
                
                # Arrival delta (time between demands)
                arrival_dist = torch.distributions.Normal(
                    outputs['arrival_mu'], outputs['arrival_std']
                )
                arrival_delta = arrival_dist.sample().clamp(0.1, 10.0)
                
                # Deadline delay
                deadline_dist = torch.distributions.Normal(
                    outputs['deadline_mu'], outputs['deadline_std']
                )
                deadline_delay = deadline_dist.sample().clamp(
                    self.min_deadline_delay, self.max_deadline_delay
                )
                
                # --- Calculate log probabilities ---
                log_prob = (
                    loc_dist.log_prob(loc).sum() +
                    qty_dist.log_prob(qty).sum() +
                    arrival_dist.log_prob(arrival_delta).sum() +
                    deadline_dist.log_prob(deadline_delay).sum()
                )
                self.saved_log_probs.append(log_prob)
                
                # --- Create the demand object ---
                current_arrival_time = last_arrival_time + arrival_delta.item()
                current_arrival_time = round(current_arrival_time)
                
                # Stop if we've exceeded the episode length
                if current_arrival_time >= episode_length:
                    break
                
                # Scale quantity to desired range
                scaled_qty = 1.0 + qty.item() * (max_quantity - 1.0)
                
                # Calculate deadline
                current_deadline = min(
                    int(current_arrival_time + deadline_delay.item()),
                    episode_length
                )
                
                # Create demand
                demand = Demand(
                    id=self.next_demand_id,
                    location=(
                        loc[0, 0].item() * map_size[0], 
                        loc[0, 1].item() * map_size[1]
                    ),
                    quantity=scaled_qty,
                    arrival_time=current_arrival_time,
                    deadline=current_deadline,
                    status='scheduled'
                )
                
                demand_schedule.append(demand)
                self.next_demand_id += 1
                last_arrival_time = current_arrival_time
                
                # --- Prepare the next input for the model ---
                # Normalize and embed the demand properties
                next_input = torch.tensor([[
                    loc[0, 0].item(),
                    loc[0, 1].item(),
                    qty.item(),
                    arrival_delta.item() / 10.0,  # Scale for numerical stability
                    deadline_delay.item() / 50.0  # Scale for numerical stability
                ]], device=self.device)
                
                next_input_embedded = self.model.input_embedder(next_input).unsqueeze(0)
                current_sequence = torch.cat([current_sequence, next_input_embedded], dim=1)
        
        # Re-enable training mode
        self.model.train()
        
        return demand_schedule

    def _generate_randomly(
        self, 
        num_demands: int, 
        episode_length: int, 
        max_quantity: float, 
        map_size: Tuple[float, float]
    ) -> List[Demand]:
        """Fallback random generator for testing and robustness."""
        demand_schedule = []
        last_arrival_time = 0
        
        for i in range(num_demands):
            # Add a random delta to the previous arrival time
            arrival_delta = random.randint(0, 5)
            arrival_time = last_arrival_time + arrival_delta
            
            # Stop if we exceed the episode length
            if arrival_time >= episode_length:
                break
                
            # Generate deadline as arrival_time + a random delay
            deadline_delay = random.uniform(self.min_deadline_delay, self.max_deadline_delay)
            deadline = min(int(arrival_time + deadline_delay), episode_length)
            
            # Create the demand
            demand = Demand(
                id=self.next_demand_id,
                location=(random.uniform(0, map_size[0]), random.uniform(0, map_size[1])),
                quantity=random.uniform(1.0, max_quantity),
                arrival_time=arrival_time,
                deadline=deadline,
                status='scheduled'
            )
            
            demand_schedule.append(demand)
            self.next_demand_id += 1
            last_arrival_time = arrival_time
        
        # Clear any saved log probs since we're not using the model
        self.saved_log_probs = []
        
        return demand_schedule

    def update(self, final_planner_performance: float) -> float:
        """
        Update policy using REINFORCE with a baseline.
        Falls back gracefully if no valid log probs are available.
        
        Args:
            final_planner_performance: Score where higher means the instance was harder
        
        Returns:
            The calculated loss or 0.0 if no update was performed
        """
        # If we have no log probs (used random generation), just update the baseline
        if not self.saved_log_probs:
            self.reward_baseline = (self.baseline_alpha * self.reward_baseline + 
                                  (1 - self.baseline_alpha) * final_planner_performance)
            print(f"Generator received score: {final_planner_performance:.2f} (no update)")
            return 0.0
            
        try:
            # Calculate advantage
            advantage = final_planner_performance - self.reward_baseline
            
            # Calculate policy gradient loss
            loss = 0
            for log_prob in self.saved_log_probs:
                loss -= log_prob * advantage
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update the baseline
            self.reward_baseline = (self.baseline_alpha * self.reward_baseline + 
                                  (1 - self.baseline_alpha) * final_planner_performance)
            
            # Clear saved log probs
            self.saved_log_probs = []
            
            print(f"Generator updated with score: {final_planner_performance:.2f}, loss: {loss.item():.4f}")
            return loss.item()
            
        except Exception as e:
            print(f"Update failed: {e}")
            self.saved_log_probs = []
            return 0.0
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'next_demand_id': self.next_demand_id,
            'reward_baseline': self.reward_baseline
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.next_demand_id = checkpoint.get('next_demand_id', 0)
        self.reward_baseline = checkpoint.get('reward_baseline', 0.0)