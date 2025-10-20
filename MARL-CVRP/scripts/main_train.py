"""
Main training script implementing the PSRO (Policy-Space Response Oracle) loop.
It alternates between training the Planner and the Generator.
"""
import torch

from agents.planner_agent import PlannerAgent
from agents.generator_agent import GeneratorAgent
from env.dynamic_cvrp_env import DynamicCVRPEnv
from utils.policy_pool import PolicyPool
from utils.data_structures import Hotspot

# --- Constants ---
NUM_ITERATIONS = 100
TRAINING_EPOCHS_P = 50
TRAINING_EPOCHS_G = 50
T_MAX = 200 # Max steps per episode

def main():
    """
    Main PSRO training loop.
    """
    # Initialize environment and policy pools
    env = DynamicCVRPEnv()
    policy_pool_P = PolicyPool()
    policy_pool_G = PolicyPool()

    # Initialize agents with starting policies
    initial_planner = PlannerAgent()
    initial_generator = GeneratorAgent()
    # policy_pool_P.add(initial_planner.model.state_dict())
    # policy_pool_G.add(initial_generator.model.state_dict())
    
    # Define static hotspots for the generator
    hotspots = [Hotspot(id=i, location=(x,y)) for i, (x,y) in enumerate([(10,10), (90,90), (10,90), (90,10)])]
    constraints = {"quantity_range": (5, 20), "deadline_delay_range": (20, 50)}


    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- PSRO Iteration {iteration+1}/{NUM_ITERATIONS} ---")

        # --- 1. Train Planner vs. Fixed Generator ---
        print("--- Training Planner (Agent P) ---")
        # fixed_policy_G_weights = policy_pool_G.sample_latest()
        fixed_policy_G = GeneratorAgent() # In a real run, load weights
        # if fixed_policy_G_weights:
        #     fixed_policy_G.model.load_state_dict(fixed_policy_G_weights)
        
        current_policy_P = PlannerAgent() # Train a new planner from scratch or fine-tune

        for ep in range(TRAINING_EPOCHS_P):
            state = env.reset()
            for t in range(T_MAX):
                action_P, log_prob_P = current_policy_P.choose_action(state)
                
                with torch.no_grad():
                    action_G, _ = fixed_policy_G.choose_action(state, hotspots, constraints)
                
                next_state, delta_regret, done = env.step(action_P, action_G)
                
                # Planner's goal is to MINIMIZE regret, so its reward is -delta_regret
                current_policy_P.update(reward=-delta_regret, log_prob=log_prob_P)
                
                state = next_state
                if done: break
        
        # policy_pool_P.add(current_policy_P.model.state_dict())
        print("Planner training complete.")

        # --- 2. Train Generator vs. Fixed Planner ---
        print("--- Training Generator (Agent G) ---")
        # fixed_policy_P_weights = policy_pool_P.sample_latest()
        fixed_policy_P = PlannerAgent() # In a real run, load weights
        # if fixed_policy_P_weights:
        #     fixed_policy_P.model.load_state_dict(fixed_policy_P_weights)

        current_policy_G = GeneratorAgent()

        for ep in range(TRAINING_EPOCHS_G):
            state = env.reset()
            for t in range(T_MAX):
                with torch.no_grad():
                    action_P, _ = fixed_policy_P.choose_action(state)
                
                action_G, log_prob_G = current_policy_G.choose_action(state, hotspots, constraints)
                
                next_state, delta_regret, done = env.step(action_P, action_G)
                
                # Generator's goal is to MAXIMIZE regret, so its reward is +delta_regret
                current_policy_G.update(reward=delta_regret, log_prob=log_prob_G)
                
                state = next_state
                if done: break
        
        # policy_pool_G.add(current_policy_G.model.state_dict())
        print("Generator training complete.")


if __name__ == "__main__":
    main()