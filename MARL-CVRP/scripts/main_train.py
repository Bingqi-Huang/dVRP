"""
Main training script for MARL-CVRP using PSRO.
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
from utils.policy_pool import PolicyPool
from utils.data_structures import Hotspot

# --- Constants ---
NUM_ITERATIONS = 1000
TRAINING_EPOCHS_P = 100
TRAINING_EPOCHS_G = 100
T_MAX = 200

class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log(self, **kwargs):
        """Log metrics for current iteration."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot(self, save_path='training_curves.png'):
        """Plot all metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('MARL-CVRP Training Metrics', fontsize=16)
        
        # Plot 1: Generator Reward
        if 'generator_reward' in self.metrics:
            axes[0, 0].plot(self.metrics['generator_reward'], label='Generator')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Avg Reward')
            axes[0, 0].set_title('Generator Reward (Regret)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Planner Reward
        if 'planner_reward' in self.metrics:
            axes[0, 1].plot(self.metrics['planner_reward'], label='Planner', color='orange')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].set_title('Planner Reward (-Regret)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Service Rate
        if 'service_rate' in self.metrics:
            axes[0, 2].plot(self.metrics['service_rate'], label='Service Rate', color='green')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Rate (%)')
            axes[0, 2].set_title('Demand Service Rate')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Plot 4: Failed Demands
        if 'failed_demands' in self.metrics:
            axes[1, 0].plot(self.metrics['failed_demands'], label='Failed', color='red')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Failed Demands per Episode')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 5: Total Demands Generated
        if 'total_demands' in self.metrics:
            axes[1, 1].plot(self.metrics['total_demands'], label='Total Demands', color='purple')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Total Demands Generated')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Plot 6: Nash Gap (if both rewards tracked)
        if 'generator_reward' in self.metrics and 'planner_reward' in self.metrics:
            # Nash gap: sum of regrets (should converge to stable value)
            nash_gap = [g + p for g, p in zip(self.metrics['generator_reward'], 
                                               self.metrics['planner_reward'])]
            axes[1, 2].plot(nash_gap, label='Nash Gap', color='brown')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Gap')
            axes[1, 2].set_title('Nash Equilibrium Gap')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📊 Training curves saved to: {save_path}")
        plt.close()


def main():
    """
    Main PSRO training loop.
    """
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create checkpoint directories
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment and policy pools
    env = DynamicCVRPEnv()
    policy_pool_P = PolicyPool(save_dir=os.path.join(checkpoint_dir, 'planner'))
    policy_pool_G = PolicyPool(save_dir=os.path.join(checkpoint_dir, 'generator'))

    # Initialize metrics tracker
    tracker = MetricsTracker()

    # Initialize agents
    initial_planner = PlannerAgent()
    initial_generator = GeneratorAgent(device=device)
    
    # Add initial generator to pool
    policy_pool_G.add(initial_generator, metadata={'iteration': 0, 'avg_reward': 0.0})

    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*70}")
        print(f"PSRO Iteration {iteration+1}/{NUM_ITERATIONS}")
        print(f"{'='*70}")

        # ====================================================
        # PHASE 1: Train Planner vs. Fixed Generator
        # ====================================================
        print("\n[Phase 1] Training Planner...")
        
        # Load fixed generator
        if policy_pool_G.size() > 0:
            fixed_policy_G = GeneratorAgent(device=device)
            fixed_policy_G.load(policy_pool_G.sample_latest())
        else:
            fixed_policy_G = GeneratorAgent(device=device)
        
        current_policy_P = PlannerAgent()

        planner_rewards = []
        for ep in range(TRAINING_EPOCHS_P):
            state = env.reset()
            episode_reward = 0
            
            for t in range(T_MAX):
                action_P = current_policy_P.choose_action(state)
                
                with torch.no_grad():
                    action_G, _ = fixed_policy_G.choose_action(state, deterministic=False)
                
                next_state, delta_regret, done = env.step(action_P, action_G)
                
                reward_P = -delta_regret
                episode_reward += reward_P
                
                state = next_state
                if done: break
            
            planner_rewards.append(episode_reward)
        
        avg_planner_reward = np.mean(planner_rewards)
        policy_pool_P.add(current_policy_P, metadata={
            'iteration': iteration + 1, 
            'avg_reward': avg_planner_reward
        })
        print(f"  ✓ Planner Avg Reward: {avg_planner_reward:.2f}")

        # ====================================================
        # PHASE 2: Train Generator vs. Fixed Planner
        # ====================================================
        print("\n[Phase 2] Training Generator...")
        
        if policy_pool_P.size() > 0:
            fixed_policy_P = current_policy_P  # Use just-trained planner
        else:
            fixed_policy_P = PlannerAgent()

        current_policy_G = GeneratorAgent(device=device)

        generator_rewards = []
        total_failed_demands = []
        total_demands_created = []
        service_rates = []
        
        for ep in range(TRAINING_EPOCHS_G):
            state = env.reset()
            
            # Reset generator memory
            current_policy_G.saved_log_probs = []
            current_policy_G.rewards = []
            
            episode_cost = 0
            demands_created = 0
            
            for t in range(T_MAX):
                with torch.no_grad():
                    action_P = fixed_policy_P.choose_action(state)
                
                action_G, log_prob_G = current_policy_G.choose_action(state, deterministic=False)
                
                demands_created += len(action_G.new_demands)

                next_state, delta_regret, done = env.step(action_P, action_G)
                
                episode_cost += delta_regret
                current_policy_G.store_reward(delta_regret)
                
                state = next_state
                if done: break
            
            # Episode metrics
            failed_count = len(state.failed_demands)
            serviced_count = len(state.serviced_demands)
            service_rate = serviced_count / demands_created if demands_created > 0 else 1.0
            
            generator_rewards.append(episode_cost)
            total_failed_demands.append(failed_count)
            total_demands_created.append(demands_created)
            service_rates.append(service_rate)
            
            # Update generator
            loss = current_policy_G.update()

        avg_generator_reward = np.mean(generator_rewards)
        avg_failed = np.mean(total_failed_demands)
        avg_service_rate = np.mean(service_rates) * 100
        avg_demands = np.mean(total_demands_created)
        
        policy_pool_G.add(current_policy_G, metadata={
            'iteration': iteration + 1,
            'avg_reward': avg_generator_reward
        })
        
        print(f"  ✓ Generator Avg Reward: {avg_generator_reward:.2f}")
        print(f"  ✓ Avg Failed Demands: {avg_failed:.1f}")
        print(f"  ✓ Avg Service Rate: {avg_service_rate:.1f}%")
        print(f"  ✓ Avg Total Demands: {avg_demands:.1f}")

        # ====================================================
        # Log Metrics
        # ====================================================
        tracker.log(
            generator_reward=avg_generator_reward,
            planner_reward=avg_planner_reward,
            failed_demands=avg_failed,
            service_rate=avg_service_rate,
            total_demands=avg_demands
        )

        # ====================================================
        # Periodic Evaluation & Plotting
        # ====================================================
        if (iteration + 1) % 10 == 0:
            print(f"\n{'='*70}")
            print(f"📈 Checkpoint at Iteration {iteration+1}")
            print(f"{'='*70}")
            print(f"  Generator Reward Trend: {tracker.metrics['generator_reward'][-10:]}")
            print(f"  Planner Reward Trend: {tracker.metrics['planner_reward'][-10:]}")
            print(f"  Service Rate Trend: {[f'{x:.1f}%' for x in tracker.metrics['service_rate'][-10:]]}")
            
            # Plot training curves
            tracker.plot(save_path=os.path.join(checkpoint_dir, f'training_curves_iter{iteration+1}.png'))

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    
    # Final plot
    tracker.plot(save_path=os.path.join(checkpoint_dir, 'final_training_curves.png'))


if __name__ == "__main__":
    main()