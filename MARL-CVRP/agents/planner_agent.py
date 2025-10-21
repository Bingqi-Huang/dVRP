"""
Planner Agent: Routes vehicles to serve demands.
Currently uses a simple greedy heuristic.
TODO: Replace with learned policy (e.g., attention-based model).
"""
import numpy as np
import math
from typing import Dict, List

from utils.data_structures import GlobalState, PlannerAction


class PlannerAgent:
    """
    The planner agent that decides vehicle routes.
    Goal: Minimize operational cost and failed demands.
    
    Current Implementation: Greedy nearest-neighbor heuristic
    Future: Replace with neural network policy (AM/POMO style)
    """
    
    def __init__(self):
        """Initialize the planner agent."""
        # TODO: The actual model (e.g., Dynamic Encoder + Rolling Decoder) would be defined here.
        self.model = None 
        self.optimizer = None

    # TODO: Get responses from a planner model(network), here is a dummy implementation. 
    # We are supposed to have multiple planner other than our own, used as baseline.
    def choose_action(self, state: GlobalState) -> PlannerAction:
        """
        Choose routing action for all vehicles.
        
        Uses greedy heuristic:
        - Each vehicle serves its nearest unassigned demand
        - If vehicle is full, return to depot first
        
        Args:
            state: Current global state
            
        Returns:
            PlannerAction with vehicle plans
        """
        # Simple greedy assignment: nearest demand to each vehicle
        routes = {}
        assigned_demands = set()
        
        for vehicle in state.vehicles:
            if vehicle.remaining_capacity <= 0:
                # Vehicle is full, needs to return to depot
                routes[vehicle.id] = []  # Empty plan = return to depot
                continue
            
            # Find nearest unassigned demand
            nearest_demand = None
            nearest_dist = float('inf')
            
            for demand in state.pending_demands:
                if demand.id in assigned_demands:
                    continue
                
                # Check if vehicle can handle this demand
                if demand.quantity > vehicle.remaining_capacity:
                    continue
                
                # Calculate distance
                dist = self._euclidean_distance(vehicle.location, demand.location)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_demand = demand
            
            # Assign nearest demand to vehicle
            if nearest_demand is not None:
                routes[vehicle.id] = [nearest_demand.id]
                assigned_demands.add(nearest_demand.id)
            else:
                routes[vehicle.id] = []
        
        # FIXED: Use 'vehicle_plans' instead of 'vehicle_routes'
        action = PlannerAction(vehicle_plans=routes)
        return action
    
    def _euclidean_distance(self, loc1, loc2):
        """Calculate Euclidean distance between two locations."""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def save(self, filepath: str):
        """
        Save planner policy.
        
        TODO: Implement when neural network is added
        """
        # Placeholder for future neural network implementation
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'type': 'greedy'}, f)
    
    def load(self, filepath: str):
        """
        Load planner policy.
        
        TODO: Implement when neural network is added
        """
        # Placeholder for future neural network implementation
        pass

    def update(self, reward: float, log_prob: float):
        """
        Updates the model's policy. The goal is to MINIMIZE regret, which means
        minimizing the reward signal from the environment (-delta_regret).
        """
        # Standard policy gradient update, e.g., loss = -log_prob * reward
        pass