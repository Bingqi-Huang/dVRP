# MARL-CVRP Project

## Overview
The MARL-CVRP project aims to develop a robust planner for solving the Dynamic Vehicle Routing Problem (DVRP) using Reinforcement Learning. The project employs an Automated Curriculum Learning (ACL) framework where a "Generator" (teacher) adaptively creates harder problems for a "Planner" (student) agent as it learns.

## Project Structure
The project is organized into several key components:

- **`agents/`**: Contains the implementation of the `PlannerAgent`, which is responsible for making decisions in the DVRP environment.
- **`configs/`**: Holds configuration files for the project, primarily `base_config.yaml`.
- **`env/`**: Implements the simulation environment, including the `CurriculumGenerator` and the `DvrpEnv`.
- **`model/`**: Contains the neural network architecture (`PlannerModel`) for the planner.
- **`scripts/`**: Includes executable scripts for training (`train.py`) and testing components (`test_generator.py`).
- **`utils/`**: Provides utility functions and core data structures used throughout the project.
- **`pyproject.toml`**: The project definition file used for installation.

## Setup Instructions

This project uses `uv` for environment and package management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MARL-CVRP
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # This creates a .venv folder in your project directory
    uv venv
    
    # Activate the environment (on Linux/macOS)
    source .venv/bin/activate
    ```

3.  **Install the project and its dependencies:**
    This command installs the project in "editable" mode, meaning any changes you make to the source code are immediately available. It also installs required libraries like `numpy`, `torch`, `pyyaml`, and `matplotlib` (for development).
    ```bash
    uv pip install -e .
    ```

## Usage

Ensure your virtual environment is activated before running any scripts.

**To run the main training loop:**
```bash
python scripts/train.py
```

**To test the curriculum generator and visualize its output:**
This script is useful for verifying that the problem generation logic is working correctly.
```bash
python scripts/test_generator.py
```

## Next Steps: TODO Summary

This section outlines the key areas requiring further implementation and design.

### [DONE]  1. Configuration (`configs/base_config.yaml` & `utils/config.py`)
*   **`acl_threshold`**: Design and tune this parameter based on actual planner performance metrics to determine when to increase difficulty.
*   **`initial_difficulty`**: Refine the representation and design of difficulty parameters based on the actual environment dynamics.
*   **Demand Quantity**: Make the `quantity` of demands (currently `np.random.randint(1, 10)`) a configurable parameter in `base_config.yaml`.

### 2. Environment (`env/`)
*   **`DvrpEnv`**:
    *   **Depot Location**: Determine how and where the central depot for vehicles is defined (currently assumed center of the map).
    *   **Vehicle Updates (`_update_vehicles`)**: Implement the logic for vehicle movement, demand servicing (loading/unloading), and route management based on the planner's actions. This method should return a list of `newly_serviced` demands.
*   **`CurriculumGenerator`**:
    *   **Difficulty Definition**: Carefully design the definition of difficulty levels beyond just `lambda`.
    *   **Difficulty Update Logic**: Refine the `update_difficulty` method based on actual planner output metrics.
    *   **Future Enhancement**: Consider replacing the current non-learning generator with a learning-based generator.
*   **`RegretOracle`**:
    *   **Reward Calculation**: Implement a meaningful `calculate_step_reward` function. This will likely involve integrating a traditional CVRP solver (e.g., LKH-3, OR-Tools) to compute optimal or near-optimal costs for regret-based rewards.

### 3. Model (`model/planner_model.py`)
*   **`PlannerModel`**: Implement the core neural network architecture for the DVRP solver. This will be the "brain" of the planner.
*   **Encoder/Decoder Components**:
    *   **`VehicleEncoder`**: Define layers and forward pass for encoding vehicle states.
    *   **`DemandEncoder`**: Define layers and forward pass for encoding demand information.
    *   **`AttentionDecoder`**: Implement attention-based mechanisms to generate action logits from encoded features.
    *   **`ValueHead`**: Implement the value network to estimate the state value.

### 4. Agent (`agents/planner_agent.py` & `utils/replay_buffer.py`)
*   **`PlannerAgent`**: Implement the full Reinforcement Learning agent logic, specifically the PPO update mechanism, including:
    *   Calculating advantages (e.g., Generalized Advantage Estimation - GAE).
    *   Iterating for `ppo_epochs` to update the model.
*   **`ReplayBuffer`**: The current buffer is a template; ensure it fully supports the requirements of the chosen RL algorithm (e.g., PPO).

