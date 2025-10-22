from typing import List
from utils.data_structures import State, Demand

class RegretOracle:
    """
    TODO: Regret calculation class.
    Placeholder for the Regret Oracle. Its final goal is to calculate step-wise
    rewards based on the difference between the agent's performance and a
    theoretical optimum.
    """
    def __init__(self):
        # In a real implementation, this might initialize a CVRP solver like LKH-3 or OR-Tools.
        self.previous_cost = 0.0

    def calculate_step_reward(self, state: State, newly_serviced: List[Demand]) -> float:
        """
        Calculates the reward for the current timestep.
        
        For now, this is a placeholder. A simple reward could be:
        - A large positive reward for each serviced demand.
        - A small negative reward for the passage of time (encouraging speed).
        """
        # Placeholder logic: +10 for each serviced demand, -0.1 for each time step.
        reward = 0.0
        reward += len(newly_serviced) * 10.0
        reward -= 0.1 # Cost for time passing
        
        # A true regret-based reward would be much more complex:
        # current_actual_cost = self._get_actual_cost(state)
        # current_ideal_cost = self._get_ideal_optimal_cost(state)
        # current_regret = current_actual_cost - current_ideal_cost
        # delta_regret = current_regret - self.previous_regret
        # self.previous_regret = current_regret
        # reward = -delta_regret
        
        return reward

    def reset(self):
        """Resets any internal state for a new episode."""
        self.previous_cost = 0.0