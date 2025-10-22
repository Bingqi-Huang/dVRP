# MARL-CVRP Project

## Overview
The MARL-CVRP project aims to develop a robust planner for solving the Dynamic Vehicle Routing Problem (DVRP) using Multi-Agent Reinforcement Learning (MARL) techniques. The project employs an Automated Curriculum Learning (ACL) framework to enhance the learning process of the planner agent.

## Project Structure
The project is organized into several key components:

- **agents/**: Contains the implementation of the planner agent, which is responsible for making decisions in the DVRP environment.
  - `planner_agent.py`: Defines the `PlannerAgent` class with methods for action selection, transition storage, and policy updates.

- **configs/**: Holds configuration files for the project.
  - `base_config.yaml`: Contains base configuration settings, including hyperparameters and environment settings.

- **env/**: Implements the environment for the DVRP, including the curriculum generator and regret oracle.
  - `curriculum_generator.py`: Defines the `CurriculumGenerator` class for generating training scripts based on difficulty levels.
  - `dvrp_env.py`: Implements the `DvrpEnv` class, simulating the DVRP environment.
  - `regret_oracle.py`: Defines the `RegretOracle` class for calculating regret-based rewards.

- **model/**: Contains the neural network architecture for the planner.
  - `planner_model.py`: Implements the `PlannerModel` class for the planner's neural network.

- **scripts/**: Includes executable scripts for training and evaluating the planner.
  - `train.py`: The main entry point for training the planner.

- **utils/**: Provides utility functions and data structures used throughout the project.
  - `config.py`: Functions for loading and managing configuration settings.
  - `data_structures.py`: Defines core data structures like `Demand`, `Vehicle`, `State`, and `DemandScriptEntry`.
  - `replay_buffer.py`: Implements the `ReplayBuffer` class for storing experiences.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd MARL-CVRP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the settings in `configs/base_config.yaml` as needed.

## Usage
To train the planner, run the following command:
```
python scripts/train.py
```

This will initiate the training process, utilizing the defined environment, agent, and curriculum generator.

## Contribution
Contributions to the project are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.