"""
Dynamic CVRP Environment with time-dependent demands.
"""
import numpy as np
from typing import Tuple, List
import copy

from utils.data_structures import (
    GlobalState, Vehicle, Demand, GeneratorAction, PlannerAction
)


class DynamicCVRPEnv:
    """
    Environment for Dynamic Capacitated Vehicle Routing Problem.
    
    Dynamics:
    1. Generator creates new demands at each timestep
    2. Planner assigns vehicles to serve demands
    3. Vehicles move toward assigned demands
    4. Demands expire if not serviced by deadline
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
            max_demands: Maximum total demands in episode
        """
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_location = depot_location
        self.vehicle_speed = vehicle_speed
        self.episode_length = episode_length
        self.max_demands = max_demands
        
        self.state = None
        self.total_demands_created = 0
    
    def reset(self) -> GlobalState:
        """
        Reset environment to initial state.
        
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
        
        self.total_demands_created = 0
        
        return copy.deepcopy(self.state)
    
    def step(
        self,
        action_P: PlannerAction,
        action_G: GeneratorAction
    ) -> Tuple[GlobalState, float, bool]:
        """
        Execute one timestep of the environment.
        
        Args:
            action_P: Planner's routing decisions
            action_G: Generator's new demands
            
        Returns:
            next_state: Updated global state
            delta_regret: Instantaneous regret (cost - optimal_cost)
            done: Whether episode is finished
        """
        # 1. Add new demands from Generator
        for demand in action_G.new_demands:
            if self.total_demands_created < self.max_demands:
                self.state.pending_demands.append(demand)
                self.total_demands_created += 1
        
        # 2. Update vehicle plans from Planner
        for vehicle in self.state.vehicles:
            if vehicle.id in action_P.vehicle_plans:
                vehicle.current_plan = action_P.vehicle_plans[vehicle.id]
        
        # 3. Move vehicles and service demands
        self._move_vehicles()
        
        # 4. Check for expired demands
        self._check_deadlines()
        
        # 5. Calculate regret (placeholder)
        delta_regret = self._calculate_regret()
        
        # 6. Advance time
        self.state.current_time += 1
        
        # 7. Check termination
        done = (
            self.state.current_time >= self.episode_length or
            self.total_demands_created >= self.max_demands
        )
        
        return copy.deepcopy(self.state), delta_regret, done
    
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
    
    def _calculate_regret(self) -> float:
        """
        Calculate instantaneous regret.
        
        Regret = (Planner's cost) - (Optimal cost)
        
        For now, using simple heuristic:
        - Cost = distance traveled + penalty for failed demands
        - Optimal = 0 (placeholder)
        
        TODO: Implement proper regret calculation
        """
        # Penalty for failed demands
        failed_penalty = len(self.state.failed_demands) * 10.0
        
        # Distance cost (simplified: just count pending demands)
        pending_penalty = len(self.state.pending_demands) * 0.1
        
        regret = failed_penalty + pending_penalty
        
        return regret
    
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