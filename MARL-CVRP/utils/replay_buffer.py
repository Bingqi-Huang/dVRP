from typing import List, Any
import numpy as np

class ReplayBuffer:
    """
    TODO: Following is only a template of replay buffer for RL algorithms like PPO.
    """
    def __init__(self):
        self.states: List[Any] = []
        self.action_log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def store(self, state, action_log_prob, reward, done, value):
        """Stores a single transition in the buffer."""
        self.states.append(state)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_data(self):
        """
        Returns all stored data as numpy arrays.
        This is typically called before a policy update.
        """
        return {
            'states': np.array(self.states, dtype=object),
            'action_log_probs': np.array(self.action_log_probs),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values)
        }

    def clear(self):
        """
        Clears the buffer. Must be called after each policy update for on-policy algorithms.
        """
        self.states.clear()
        self.action_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)