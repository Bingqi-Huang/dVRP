"""
A simple policy pool for the PSRO training paradigm.
Stores the weights of trained policies and allows sampling from them.
"""
import random
from collections import deque

class PolicyPool:
    def __init__(self, max_size=10):
        """
        Initializes a policy pool with a maximum size.
        """
        self.policies = deque(maxlen=max_size)

    def add(self, policy_weights):
        """
        Adds a new set of policy weights to the pool.
        """
        self.policies.append(policy_weights)

    def sample_latest(self):
        """
        Returns the most recently added policy.
        """
        if not self.policies:
            return None
        return self.policies[-1]

    def sample(self):
        """
        Returns a random policy from the pool.
        """
        if not self.policies:
            return None
        return random.choice(self.policies)