import torch
import torch.optim as optim
from torch.distributions import Categorical # Example distribution

from model.planner_model import PlannerModel
from utils.replay_buffer import ReplayBuffer
from utils.config import PlannerConfig, ModelConfig
from utils.data_structures import State

class PlannerAgent:
    """
    TODO: This is the RL agent logic based on our planner model.
    Implements the RL agent logic (e.g., PPO). It uses the PlannerModel
    to make decisions and learns by updating the model's weights.
    """
    def __init__(self, planner_config: PlannerConfig, model_config: ModelConfig):
        self.config = planner_config
        
        # Initialize the model and optimizer
        self.model = PlannerModel(model_config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Initialize the replay buffer for on-policy learning
        self.buffer = ReplayBuffer()

    def choose_action(self, state: State):
        """
        Selects an action based on the current state observation.
        """
        # Note: In a real implementation, the 'state' object would be
        # pre-processed into a tensor before being passed to the model.
        
        with torch.no_grad():
            action_logits, state_value = self.model(state)
        
        # Create a probability distribution from the logits.
        # This is a placeholder for a more complex action space.
        dist = Categorical(logits=action_logits)
        
        # Sample an action from the distribution
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        # The returned 'action' here is just a tensor. It will need to be
        # decoded into a meaningful plan for the environment's step function.
        return action, action_log_prob, state_value

    def store_transition(self, state, action_log_prob, reward, done, value):
        """Stores an experience tuple in the replay buffer."""
        self.buffer.store(state, action_log_prob, reward, done, value)

    def update_policy(self):
        """
        Updates the policy using the experiences collected in the buffer.
        This is a placeholder for the actual PPO update logic.
        """
        if len(self.buffer) == 0:
            print("Buffer is empty, skipping policy update.")
            return

        print(f"Updating policy with {len(self.buffer)} transitions...")
        
        # In a real PPO implementation, you would:
        # 1. Get data from the buffer
        # 2. Calculate advantages (e.g., GAE)
        # 3. Loop for PPO_EPOCHS and update the model
        
        # For now, we just clear the buffer as is required for on-policy algorithms.
        self.buffer.clear()
        print("Policy update placeholder finished and buffer cleared.")

    def save_model(self, path: str):
        """Saves the model weights."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Loads the model weights."""
        self.model.load_state_dict(torch.load(path))