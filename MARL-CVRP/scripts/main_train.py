"""
Main training script for MARL-CVRP.
This script implements the offline adversarial training loop:
1. GENERATE: The Generator creates a complete, challenging CVRP instance.
2. SOLVE: The Planner attempts to solve this instance as it unfolds dynamically.
3. UPDATE: The Generator is updated based on the Planner's final performance.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from env.dynamic_cvrp_env import DynamicCVRPEnv
from agents.generator_agent import GeneratorAgent
from agents.planner_agent import PlannerAgent
from utils.data_structures import GlobalState

# --- Constants ---
NUM_EPISODES = 2000
NUM_DEMANDS_PER_INSTANCE = 50
MAX_QUANTITY = 20.0
CHECKPOINT_INTERVAL = 100

# --- Hyperparameters ---
# (These will be more relevant when the learning models are integrated)
LEARNING_RATE_G = 1e-4
LEARNING_RATE_P = 1e-4


class MetricsTracker:
    """Track and visualize training metrics for the new loop."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log(self, **kwargs):
        """Log metrics for the current episode."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot(self, save_path='training_curves.png'):
        """Plot all tracked metrics."""
        num_metrics = len(self.metrics)
        if num_metrics == 0: return
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(8, num_metrics * 4))
        if num_metrics == 1: axes = [axes] # Make it iterable
        fig.suptitle('Adversarial Training Metrics', fontsize=16)
        
        for ax, (key, values) in zip(axes, self.metrics.items()):
            ax.plot(values, label=key)
            ax.set_xlabel('Episode')
            ax.set_ylabel(key.replace('_', ' ').title())
            ax.set_title(f'{key.replace("_", " ").title()} over Time')
            ax.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        print(f"📊 Training curves saved to: {save_path}")
        plt.close()


def calculate_final_performance(final_state: GlobalState) -> float:
    """
    Calculates a performance score for the planner at the end of an episode.
    A higher score means WORSE performance. This is the reward signal for the generator.
    
    Args:
        final_state: The GlobalState at the end of the episode.
        
    Returns:
        A float representing the total cost/poor performance of the planner.
    """
    # High penalty for each demand that was not serviced in time.
    failure_penalty = len(final_state.failed_demands) * 100.0
    
    # Penalty for demands that were revealed but not yet serviced.
    pending_penalty = len(final_state.pending_demands) * 10.0
    
    # TODO: Add total distance traveled by vehicles as a cost component
    # total_distance = sum(v.distance_traveled for v in final_state.vehicles)
    
    return failure_penalty + pending_penalty


def main():
    """
    Main training loop following the Generate -> Solve -> Update paradigm.
    """
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(os.path.join(checkpoint_dir, 'generator'), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'planner'), exist_ok=True)

    env = DynamicCVRPEnv()
    generator_agent = GeneratorAgent(learning_rate=LEARNING_RATE_G, device=device)
    planner_agent = PlannerAgent() # Currently a non-learning heuristic agent
    tracker = MetricsTracker()

    print("Starting adversarial training...")
    # --- Main Training Loop ---
    for episode in tqdm(range(NUM_EPISODES), desc="Training Episodes"):
        
        # 1. GENERATE: Generator creates the entire problem instance.
        demand_schedule = generator_agent.generate_instance(
            num_demands=NUM_DEMANDS_PER_INSTANCE,
            episode_length=env.episode_length,
            max_quantity=MAX_QUANTITY,
            map_size=(1.0, 1.0)
        )

        # 2. SOLVE: Planner attempts to solve the instance dynamically.
        state = env.reset(demand_schedule=demand_schedule)
        done = False
        
        while not done:
            # Planner acts based on the current state.
            # NOTE: The current PlannerAgent is a simple heuristic.
            planner_action = planner_agent.choose_action(state)
            
            # Environment steps forward, revealing new demands from the schedule.
            next_state, reward, done, info = env.step(planner_action)
            
            # For a learning planner, this is where it would store experience.
            # planner_agent.store_transition(state, planner_action, reward, next_state, done)
            state = next_state

        # 3. UPDATE: Agents are updated based on the episode's outcome.
        final_state = env.state
        
        # Calculate a final performance score for the planner.
        # This score is the "reward" for the generator.
        generator_reward = calculate_final_performance(final_state)

        # Update the generator (placeholder for now).
        generator_loss = generator_agent.update(generator_reward)
        
        # Update the planner (placeholder for now, as it's not a learning agent).
        # planner_loss = planner_agent.update()

        # --- Logging and Checkpointing ---
        serviced_count = len(final_state.serviced_demands)
        failed_count = len(final_state.failed_demands)
        total_demands = serviced_count + failed_count + len(final_state.pending_demands)
        service_rate = (serviced_count / total_demands) * 100 if total_demands > 0 else 100.0

        tracker.log(
            generator_reward=generator_reward,
            service_rate_percent=service_rate,
            failed_demands=failed_count
        )

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            print(f"\n--- Episode {episode + 1} Checkpoint ---")
            print(f"  Generator Reward (Planner Cost): {generator_reward:.2f}")
            print(f"  Service Rate: {service_rate:.2f}%")
            print(f"  Failed Demands: {failed_count}")
            
            # Save models and plot metrics
            gen_path = os.path.join(checkpoint_dir, 'generator', f'policy_{episode+1}.pt')
            generator_agent.save(gen_path)
            tracker.plot(save_path=os.path.join(checkpoint_dir, f'training_curves_ep{episode+1}.png'))

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    
    # Final plot
    tracker.plot(save_path=os.path.join(checkpoint_dir, 'final_training_curves.png'))


if __name__ == "__main__":
    main()