"""
Dynamic CVRP Environment with time-dependent demands.
"""
import numpy as np
from typing import Tuple, List
import copy

from utils.data_structures import (
    GlobalState, Vehicle, Demand, PlannerAction
)


class DynamicCVRPEnv:
    """
    Environment for Dynamic Capacitated Vehicle Routing Problem.
    
    Dynamics:
    1. The environment is initialized with a full schedule of demands.
    2. Demands are revealed from the schedule as time progresses.
    3. Planner assigns vehicles to serve pending demands.
    4. Vehicles move toward assigned demands.
    5. Demands expire if not serviced by deadline.
    """
    
    def __init__(
        self,
        num_vehicles: int = 5,
        vehicle_capacity: float = 100.0,
        depot_location: Tuple[float, float] = (0.5, 0.5),
        vehicle_speed: float = 0.1,  # Units per timestep
        episode_length: int = 100,
        max_demands: int = 50
    ):
        """
        Args:
            num_vehicles: Number of delivery vehicles
            vehicle_capacity: Maximum capacity per vehicle
            depot_location: (x, y) coordinates of depot
            vehicle_speed: Distance traveled per timestep
            episode_length: Maximum timesteps per episode
            max_demands: Maximum total demands in episode (now informational)
        """
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.vehicle_speed = vehicle_speed
        self.episode_length = episode_length
        self.max_demands = max_demands
        
        self.state = None
        self.demand_schedule: List[Demand] = []
    
    def reset(self, demand_schedule: List[Demand]) -> GlobalState:
        """
        Reset environment to initial state with a given demand schedule.
        
        Args:
            demand_schedule: A complete list of demands for the episode.

        Returns:
            Initial GlobalState
        """
        # Initialize vehicles at depot
        vehicles = [
            Vehicle(
                id=i,
                location=self.depot_location,
                remaining_capacity=self.vehicle_capacity,
                current_plan=[]
            )
            for i in range(self.num_vehicles)
        ]
        
        self.state = GlobalState(
            current_time=0,
            vehicles=vehicles,
            pending_demands=[],
            serviced_demands=[],
            failed_demands=[]
        )
        
        # Set and sort the new demand schedule
        self.demand_schedule = sorted(demand_schedule, key=lambda d: d.arrival_time)
        
        # Reveal any demands that appear at t=0
        self._reveal_new_demands()
        
        return copy.deepcopy(self.state)
    
    def step(
        self,
        action_P: PlannerAction
    ) -> Tuple[GlobalState, float, bool, dict]:
        """
        Execute one timestep of the environment.
        
        Args:
            action_P: Planner's routing decisions
            
        Returns:
            next_state: Updated global state
            reward: Reward for the planner for this step
            done: Whether episode is finished
            info: Auxiliary diagnostic information
        """
        # 1. Update vehicle plans from Planner
        for vehicle in self.state.vehicles:
            if vehicle.id in action_P.vehicle_plans:
                vehicle.current_plan = action_P.vehicle_plans[vehicle.id]
        
        # 2. Move vehicles and service demands
        self._move_vehicles()
        
        # 3. Check for expired demands
        self._check_deadlines()
        
        # 4. Calculate reward for the planner
        reward = self._calculate_reward()
        
        # 5. Advance time
        self.state.current_time += 1
        
        # 6. Reveal new demands that have arrived
        self._reveal_new_demands()
        
        # 7. Check termination
        # Episode ends if time is up.
        done = self.state.current_time >= self.episode_length
        
        info = {} # Empty info dict for now

        return copy.deepcopy(self.state), reward, done, info
    
    def _reveal_new_demands(self):
        """
        Moves demands from the schedule to the pending list if their arrival time has come.
        Assumes self.demand_schedule is sorted by arrival_time.
        """
        while self.demand_schedule and self.demand_schedule[0].arrival_time <= self.state.current_time:
            demand = self.demand_schedule.pop(0)
            demand.status = 'pending'
            self.state.pending_demands.append(demand)
        
        # Optional: sort pending demands for planner convenience (e.g., by deadline)
        self.state.pending_demands.sort(key=lambda d: d.deadline)

    def _move_vehicles(self):
        """Move vehicles toward their next target and service demands."""
        for vehicle in self.state.vehicles:
            if not vehicle.current_plan:
                continue
            
            # Get next demand
            next_demand_id = vehicle.current_plan[0]
            demand = self._find_demand_by_id(next_demand_id)
            
            if demand is None:
                # Demand no longer exists, remove from plan
                vehicle.current_plan.pop(0)
                continue
            
            # Calculate distance to demand
            dist = self._euclidean_distance(vehicle.location, demand.location)
            
            if dist <= self.vehicle_speed:
                # Service the demand
                if vehicle.remaining_capacity >= demand.quantity:
                    vehicle.remaining_capacity -= demand.quantity
                    demand.status = 'serviced'
                    self.state.serviced_demands.append(demand)
                    self.state.pending_demands.remove(demand)
                    vehicle.current_plan.pop(0)
                    vehicle.location = demand.location
                else:
                    # Not enough capacity, skip
                    vehicle.current_plan.pop(0)
            else:
                # Move toward demand
                direction = (
                    (demand.location[0] - vehicle.location[0]) / dist,
                    (demand.location[1] - vehicle.location[1]) / dist
                )
                vehicle.location = (
                    vehicle.location[0] + direction[0] * self.vehicle_speed,
                    vehicle.location[1] + direction[1] * self.vehicle_speed
                )
    
    def _check_deadlines(self):
        """Mark demands as failed if deadline exceeded."""
        expired = []
        for demand in self.state.pending_demands:
            if self.state.current_time >= demand.deadline:
                demand.status = 'failed'
                expired.append(demand)
        
        for demand in expired:
            self.state.pending_demands.remove(demand)
            self.state.failed_demands.append(demand)
    
    def _calculate_reward(self) -> float:
        """
        Calculate instantaneous reward for the planner.
        
        Regret = (Planner's cost) - (Optimal cost)
        
        For now, using simple heuristic as a cost:
        - Cost = distance traveled + penalty for failed demands
        - Optimal = 0 (placeholder)
        
        The reward is the negative of this cost.
        TODO: Implement proper reward calculation
        """
        # Penalty for failed demands
        failed_penalty = len(self.state.failed_demands) * 10.0
        
        # Distance cost (simplified: just count pending demands)
        pending_penalty = len(self.state.pending_demands) * 0.1
        
        cost = failed_penalty + pending_penalty
        
        return -cost
    
    def _find_demand_by_id(self, demand_id: int) -> Demand:
        """Find demand in pending list by ID."""
        for demand in self.state.pending_demands:
            if demand.id == demand_id:
                return demand
        return None
    
    def _euclidean_distance(
        self, 
        loc1: Tuple[float, float], 
        loc2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two locations."""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)