"""
Contains all core data structures for the project based on the guide.
These structures define the state and action spaces for the agents and environment.
"""
from dataclasses import dataclass
from typing import List, Dict

# --- Entities ---
@dataclass
class VehicleState:
    id: int
    location: tuple  # (x, y)
    remaining_capacity: float
    current_plan: List[int]  # Next H-step demand IDs

@dataclass
class DemandState:
    id: int
    location: tuple  # (x, y)
    quantity: float
    arrival_time: int
    deadline: int
    status: str  # 'pending', 'serviced', 'failed'

@dataclass
class Hotspot:
    id: int
    location: tuple  # GMM center (x, y)

# --- State & Actions ---
@dataclass
class GlobalState:
    current_time: int
    vehicles: List[VehicleState]
    pending_demands: List[DemandState]
    # Note: Hotspots are considered static and passed separately to the agent

@dataclass
class PlannerAction:
    # Key is vehicle_id, value is the H-step route for that vehicle
    vehicle_routes: Dict[int, List[int]]

@dataclass
class GeneratorAction:
    # List of new demands to be added to the environment
    new_demands: List[DemandState]