"""
The Generator Agent implementation.
This agent uses the AdversarialTransformer model to generate challenging demands,
and updates its policy using reinforcement learning to maximize the Planner's regret.
"""
import torch
import torch.optim as optim
from typing import List

from model.adversarial_transformer import AdversarialTransformer
from utils.data_structures import GlobalState, Hotspot, GeneratorAction, DemandState

class GeneratorAgent:
    def __init__(self, embed_dim=128, learning_rate=1e-4):
        """
        Initializes the agent, its model, and the optimizer.
        """
        self.model = AdversarialTransformer(embed_dim=embed_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def choose_action(self, state: GlobalState, hotspots: List[Hotspot], constraints: dict) -> (GeneratorAction, float):
        """
        Chooses a composite action based on the current state.
        1. Gets action distributions from the model.
        2. Samples an action from each distribution.
        3. Applies real-world constraints to the sampled values.
        4. Constructs a GeneratorAction object.
        5. Calculates the log probability of the chosen action.
        
        Returns:
            A tuple containing the GeneratorAction and its total log probability.
        """
        loc_dist, qty_dist, urgency_dist = self.model(state, hotspots)

        # Sample actions
        loc_idx = loc_dist.sample()
        quantity = qty_dist.sample()
        deadline_delay = urgency_dist.sample()

        # [KEY] Apply constraints
        # These values come from the environment's budget rules
        q_min, q_max = constraints.get("quantity_range", (1, 10))
        l_min, l_max = constraints.get("deadline_delay_range", (10, 100))
        
        final_loc = hotspots[loc_idx.item()].location
        final_qty = torch.clamp(quantity, q_min, q_max).item()
        final_delay = torch.clamp(deadline_delay, l_min, l_max).item()
        
        # Calculate log probabilities
        log_prob = loc_dist.log_prob(loc_idx) + \
                   qty_dist.log_prob(quantity) + \
                   urgency_dist.log_prob(deadline_delay)

        # Create the final action object
        new_demand = DemandState(
            id=-1, # Will be assigned by the env
            location=final_loc,
            quantity=final_qty,
            arrival_time=state.current_time,
            deadline=state.current_time + int(final_delay),
            status='pending'
        )
        action = GeneratorAction(new_demands=[new_demand])
        
        return action, log_prob

    def update(self, reward: float, log_prob: float):
        """
        Updates the model's policy using a simple policy gradient method (REINFORCE).
        The loss is the negative log probability of the action multiplied by the reward.
        The goal is to MAXIMIZE reward, so we do gradient ASCENT (minimize -reward).
        """
        loss = -log_prob * reward
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()