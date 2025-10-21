"""
Main training script for MARL-CVRP using PSRO.
"""
import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import torch
import numpy as np
from tqdm import tqdm

from env.dynamic_cvrp_env import DynamicCVRPEnv
from agents.generator_agent import GeneratorAgent
from agents.planner_agent import PlannerAgent
from utils.policy_pool import PolicyPool
from utils.data_structures import Hotspot

# --- Constants ---
NUM_ITERATIONS = 1000
TRAINING_EPOCHS_P = 100
TRAINING_EPOCHS_G = 100
T_MAX = 200 # Max steps per episode

def main():
    """
    Main PSRO training loop.
    """
    # --- START GPU/CPU DEVICE SETUP ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    # --- END DEVICE SETUP ---

    # Create checkpoint directories
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment and policy pools
    env = DynamicCVRPEnv()
    policy_pool_P = PolicyPool(save_dir=os.path.join(checkpoint_dir, 'planner'))
    policy_pool_G = PolicyPool(save_dir=os.path.join(checkpoint_dir, 'generator'))

    # Initialize agents with starting policies
    initial_planner = PlannerAgent()
    initial_generator = GeneratorAgent(device=device)
    
    # Add initial generator to pool
    policy_pool_G.add(initial_generator, metadata={'iteration': 0, 'avg_reward': 0.0})
    
    # Define static hotspots for the generator (convert to proper coordinates in [0,1])
    hotspots = [
        Hotspot(location=(0.1, 0.1)),
        Hotspot(location=(0.9, 0.9)),
        Hotspot(location=(0.1, 0.9)),
        Hotspot(location=(0.9, 0.1)),
        Hotspot(location=(0.5, 0.5))
    ]


    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- PSRO Iteration {iteration+1}/{NUM_ITERATIONS} ---")

        # --- 1. Train Planner vs. Fixed Generator ---
        print("--- Training Planner (Agent P) ---")
        
        # Load fixed generator from pool
        if policy_pool_G.size() > 0:
            fixed_policy_G = GeneratorAgent(device=device)
            fixed_policy_G.load(policy_pool_G.sample_latest())
        else:
            fixed_policy_G = GeneratorAgent(device=device)
        
        current_policy_P = PlannerAgent() # Train a new planner from scratch

        planner_rewards = []
        for ep in range(TRAINING_EPOCHS_P):
            state = env.reset()
            episode_reward = 0
            
            for t in range(T_MAX):
                action_P = current_policy_P.choose_action(state)
                
                with torch.no_grad():
                    action_G, _ = fixed_policy_G.choose_action(state, deterministic=False)
                
                next_state, delta_regret, done = env.step(action_P, action_G)
                
                # Planner's goal is to MINIMIZE regret, so its reward is -delta_regret
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
        print(f"Planner training complete. Avg reward: {avg_planner_reward:.2f}")

        # --- 2. Train Generator vs. Fixed Planner ---
        print("--- Training Generator (Agent G) ---")
        
        # Load fixed planner from policy pool
        if policy_pool_P.size() > 0:
            fixed_policy_P = PlannerAgent()
            # Note: PlannerAgent doesn't have load() yet, so using latest trained one
            fixed_policy_P = current_policy_P  # Use the one we just trained
        else:
            fixed_policy_P = PlannerAgent()

        current_policy_G = GeneratorAgent(device=device)

        # --- LOGGING FOR GENERATOR TRAINING ---
        print("  Epoch | Avg Episode Cost | Avg Failed Demands | Avg Service Rate")
        generator_rewards = []
        
        for ep in range(TRAINING_EPOCHS_G):
            state = env.reset()
            
            # Reset generator's episode memory
            current_policy_G.saved_log_probs = []
            current_policy_G.rewards = []
            
            # --- METRIC TRACKING ---
            total_episode_cost = 0
            total_demands_generated = 0
            
            for t in range(T_MAX):
                with torch.no_grad():
                    action_P = fixed_policy_P.choose_action(state)
                
                action_G, log_prob_G = current_policy_G.choose_action(state, deterministic=False)
                
                # Track generated demands
                total_demands_generated += len(action_G.new_demands)

                next_state, delta_regret, done = env.step(action_P, action_G)
                
                # Accumulate cost
                total_episode_cost += delta_regret

                # Generator's goal is to MAXIMIZE regret, so its reward is +delta_regret
                current_policy_G.store_reward(delta_regret)
                
                state = next_state
                if done: break
            
            # Update generator at end of episode
            loss = current_policy_G.update()
            generator_rewards.append(total_episode_cost)
            
            # --- CALCULATE AND LOG END-OF-EPISODE METRICS ---
            # Count failed demands at the end of the episode
            failed_demands_count = len(state.failed_demands)
            serviced_count = len(state.serviced_demands)
            
            service_rate = (serviced_count / total_demands_generated) if total_demands_generated > 0 else 1.0

            if (ep + 1) % 10 == 0: # Log every 10 epochs
                print(f"  {ep+1:^5} | {total_episode_cost:^18.2f} | {failed_demands_count:^20} | {service_rate:^18.2%}")

        avg_generator_reward = np.mean(generator_rewards)
        policy_pool_G.add(current_policy_G, metadata={
            'iteration': iteration + 1,
            'avg_reward': avg_generator_reward
        })
        print(f"Generator training complete. Avg reward: {avg_generator_reward:.2f}")


if __name__ == "__main__":
    main()