from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Demand:
    """Represents a single customer demand."""
    id: int
    location: Tuple[float, float]
    quantity: int
    arrival_time: int
    status: str = "pending"  # e.g., pending, serviced

@dataclass
class Vehicle:
    """Represents a single vehicle."""
    id: int
    location: Tuple[float, float]
    capacity: int
    route: List[int] = field(default_factory=list)  # List of demand IDs

@dataclass
class State:
    """Represents the observation for the Planner at a given timestep."""
    current_time: int
    vehicles: List[Vehicle]
    pending_demands: List[Demand]

@dataclass
class DemandScriptEntry:
    """Represents a single line in the offline generated demand script."""
    arrival_time: int
    location: Tuple[float, float]
    quantity: int