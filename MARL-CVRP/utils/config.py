import yaml
from dataclasses import dataclass
from typing import Dict

@dataclass
class EnvConfig:
    sim_duration: int
    n_vehicles: int
    vehicle_capacity: int
    map_size: int

@dataclass
class GeneratorConfig:
    acl_threshold: float
    initial_difficulty: Dict[str, float]
    min_demand_quantity: int
    max_demand_quantity: int


@dataclass
class PlannerConfig:
    lr: float
    gamma: float
    gae_lambda: float
    ppo_clip: float
    ppo_epochs: int
    batch_size: int

@dataclass
class ModelConfig:
    embedding_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int

@dataclass
class TrainingConfig:
    num_acl_updates: int
    script_batch_size: int
    log_interval: int

@dataclass
class Config:
    seed: int
    device: str
    env: EnvConfig
    generator: GeneratorConfig
    planner: PlannerConfig
    model: ModelConfig
    training: TrainingConfig

def load_config(path: str) -> Config:
    """Loads a YAML config file and parses it into a Config object."""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(
        seed=config_dict['seed'],
        device=config_dict['device'],
        env=EnvConfig(**config_dict['env']),
        generator=GeneratorConfig(**config_dict['generator']),
        planner=PlannerConfig(**config_dict['planner']),
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training'])
    )