from typing import List, Tuple

from utils.data_structures import Demand, Vehicle, State, DemandScriptEntry
from utils.config import EnvConfig
# from env.regret_oracle import RegretOracle # To be implemented in Step 5

class DvrpEnv:
    """
    The core DVRP simulation environment. It executes a static demand script
    provided by the CurriculumGenerator.
    """
    def __init__(self, config: EnvConfig): # oracle: RegretOracle
        self.config = config
        # self.oracle = oracle # TODO: To be used for reward calculation and saving for the generator. 
        
        self.current_time: int = 0
        self.vehicles: List[Vehicle] = []
        self.pending_demands: List[Demand] = []
        self.serviced_demands: List[Demand] = []
        
        self._full_demand_script: List[DemandScriptEntry] = []
        self._demand_idx_counter = 0

    def reset(self, demand_script: List[DemandScriptEntry]) -> State:
        """
        Resets the environment with a new demand script for ONE new episode.
        This reset initializes vehicles, and obtain the demands from generator for THIS episode.
        """
        self.current_time = 0
        self._full_demand_script = demand_script
        self._demand_idx_counter = 0
        
        # TODO： Need to determine how and where is the central depot for vehicles. For now, assume center of the map.
        depot_location = (self.config.map_size / 2, self.config.map_size / 2)
        self.vehicles = [
            Vehicle(
                id=i,
                location=depot_location,
                capacity=self.config.vehicle_capacity
            ) for i in range(self.config.n_vehicles)
        ]
        
        self.pending_demands = []
        self.serviced_demands = []
        
        self._reveal_new_demands() 
        return self._get_current_state()

    def step(self, action) -> Tuple[State, float, bool]:
        """
        Advances the environment by one time step.
        """
        if self.current_time >= self.config.sim_duration:
            raise Exception("Cannot step beyond simulation duration. Please reset.")

        self.current_time += 1
        
        # 1. Update vehicle states based on the planner's action
        self._update_vehicles(action)
        
        # 2. Reveal new demands that arrive at the current time
        self._reveal_new_demands()
        
        # 3. Calculate reward using the oracle
        # TODO: This is a placeholder, write a cal_reward function yourself in regret_oracle.py
        # For now, a placeholder reward. This will need to be updated to reflect
        # objectives like minimizing travel distance or waiting time.
        # reward = self.oracle.calculate_step_reward(...)
        reward = 0.0 # Simple placeholder reward

        # 4. Check if the episode is done
        done = self.current_time >= self.config.sim_duration
        
        next_state = self._get_current_state()
        
        return next_state, reward, done

    def _get_current_state(self) -> State:
        """Constructs the State object from the current environment data."""
        return State(
            current_time=self.current_time,
            vehicles=self.vehicles,
            pending_demands=self.pending_demands
        )

    def _reveal_new_demands(self):
        """
        Demand for THIS episode were pre-generated and stored in self._full_demand_script.
        Checks the demand script and adds any new demands that have arrived.
        """
        newly_revealed = []
        for i, entry in enumerate(self._full_demand_script):
            if entry.arrival_time == self.current_time:
                demand = Demand(
                    id=self._demand_idx_counter,
                    location=entry.location,
                    quantity=entry.quantity,
                    arrival_time=entry.arrival_time
                )
                newly_revealed.append(demand)
                self._demand_idx_counter += 1
        
        if newly_revealed:
            self.pending_demands.extend(newly_revealed)
            # Remove revealed demands from pre generated script for this episode to avoid re-processing in next step.
            self._full_demand_script = [d for d in self._full_demand_script if d.arrival_time > self.current_time]

    def _update_vehicles(self, action):
        """Placeholder for vehicle movement and servicing logic."""
        # TODO: This is depending on how actions from planner are defined.
        # e.g., update vehicle locations, handle loading/unloading, etc.
        pass