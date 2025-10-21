"""
Data structures for the Dynamic CVRP environment.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Vehicle:
    """Represents a delivery vehicle."""
    id: int
    location: Tuple[float, float]  # (x, y) coordinates in [0, 1]
    remaining_capacity: float
    current_plan: List[int] = field(default_factory=list)  # List of demand IDs to visit

@dataclass
class Demand:
    """Represents a customer demand."""
    id: int
    location: Tuple[float, float]  # (x, y) coordinates in [0, 1]
    quantity: float
    arrival_time: int  # Timestep when demand appeared
    deadline: int      # Timestep by which demand must be serviced
    status: str = 'pending'  # 'pending', 'serviced', 'failed'

@dataclass
class Hotspot:
    """Represents a potential location for new demands (used by Generator)."""
    location: Tuple[float, float]  # (x, y) coordinates in [0, 1]

@dataclass
class GlobalState:
    """Complete state of the DVRP system at a timestep."""
    current_time: int
    vehicles: List[Vehicle]
    pending_demands: List[Demand]
    serviced_demands: List[Demand] = field(default_factory=list)
    failed_demands: List[Demand] = field(default_factory=list)

@dataclass
class GeneratorAction:
    """Action taken by the Generator agent."""
    new_demands: List[Demand]  # List of demands to introduce this timestep

@dataclass
class PlannerAction:
    """Action taken by the Planner agent."""
    vehicle_plans: dict  # {vehicle_id: [demand_id_1, demand_id_2, ...]}