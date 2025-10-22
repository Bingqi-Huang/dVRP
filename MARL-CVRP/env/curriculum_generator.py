import numpy as np
from typing import List, Dict, Any

from utils.data_structures import DemandScriptEntry
from utils.config import GeneratorConfig, EnvConfig

class CurriculumGenerator:
    """
    Generates demand scripts for the environment based on a difficulty level.
    This is a non-learning component for now.
    TODO: The definition of difficulty level should be carefully designed.
    TODO: In future, this can be replaced with a learning-based generator.
    """
    def __init__(self, gen_config: GeneratorConfig, env_config: EnvConfig):
        self.gen_config = gen_config
        self.env_config = env_config
        self.current_difficulty = self.gen_config.initial_difficulty.copy()
        # TODO: Define a proper threshold based on planner's performance metrics.
        self.acl_threshold = self.gen_config.acl_threshold

    def get_difficulty(self) -> Dict[str, Any]:
        """Returns the current difficulty parameters."""
        return self.current_difficulty

    def update_difficulty(self, planner_performance: float):
        """
        Updates the difficulty if the planner's performance exceeds a threshold.
        This is the core of the outer loop (ACL).
        """
        # TODO: This is a dummy example of updating difficulty. Need to changed based on actual planner output metrics.
        if planner_performance < self.acl_threshold:
            # Example of increasing difficulty: increase demand arrival rate.
            self.current_difficulty['lambda'] *= 1.1
            print(f"Planner performance threshold met. New difficulty: {self.current_difficulty}")
        else:
            print("Planner performance did not meet threshold. Difficulty remains the same.")

    def generate_script_batch(self, batch_size: int) -> List[List[DemandScriptEntry]]:
        """Generates a batch of demand scripts containing batch_size number of episodes."""
        return [self._generate_single_script() for _ in range(batch_size)]


    # TODO: Modify the generate policy on new difficulty parameters.
    def _generate_single_script(self) -> List[DemandScriptEntry]:
        """
        Generates a single demand script episode using a random process based on current difficulty.
        """
        script = []
        # TODO: Use difficulty parameters to influence demand generation, please update as here is only a dummy example.
        demand_rate = self.current_difficulty['lambda']
        
        # Generate arrival times using a Poisson process
        current_time = 0
        while current_time < self.env_config.sim_duration:
            inter_arrival_time = np.random.exponential(1.0 / demand_rate)
            current_time += inter_arrival_time
            
            if current_time < self.env_config.sim_duration:
                arrival_time = int(round(current_time))
                
                # Generate other demand properties
                location = tuple(np.random.rand(2) * self.env_config.map_size) # Random location within map
                quantity = np.random.randint(1, 10) # TODO: make this config parameter writtin in config.yaml and read here.

                script.append(DemandScriptEntry(
                    arrival_time=arrival_time,
                    location=location,
                    quantity=quantity
                ))
        
        return sorted(script, key=lambda d: d.arrival_time)