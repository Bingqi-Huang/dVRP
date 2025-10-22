import os
import yaml
import numpy as np

from utils.config import load_config
from env.curriculum_generator import CurriculumGenerator
from env.dvrp_env import DvrpEnv
from env.regret_oracle import RegretOracle
from agents.planner_agent import PlannerAgent

def main():
    """
    Main function to run the training pipeline.
    """
    print("--- Starting DVRP Planner Training ---")
    
    # 1. Load Configuration
    config = load_config('configs/base_config.yaml')
    np.random.seed(config.seed)
    
    # 2. Initialize Components
    oracle = RegretOracle()
    generator = CurriculumGenerator(config.generator, config.env)
    env = DvrpEnv(config.env, oracle)
    agent = PlannerAgent(config.planner, config.model)
    
    # --- OUTER LOOP (Automated Curriculum Learning) ---
    # This loop adjusts the difficulty of the problems.
    num_acl_updates = config.training.num_acl_updates
    
    for acl_step in range(num_acl_updates):
        print(f"\n--- ACL Step {acl_step + 1}/{num_acl_updates} ---")
        print(f"Current Difficulty: {generator.get_difficulty()}")
        
        # 3. Generate a batch of training instances (scripts)
        script_batch_size = config.training.script_batch_size
        script_batch = generator.generate_script_batch(script_batch_size)
        
        episode_rewards = []
        
        # --- INNER LOOP (RL Agent Training) ---
        # This loop trains the agent on the generated batch of scripts.
        for i, episode_script in enumerate(script_batch):
            
            # 4. Run one episode
            state = env.reset(episode_script)
            oracle.reset()
            done = False
            total_episode_reward = 0
            
            while not done:
                # a. Agent chooses an action
                action, log_prob, value = agent.choose_action(state)
                
                # b. Environment steps forward
                # Note: The placeholder 'action' is not yet used by env.step()
                next_state, reward, done = env.step(action)
                
                # c. Agent stores the transition
                agent.store_transition(state, log_prob, reward, done, value)
                
                state = next_state
                total_episode_reward += reward
            
            # 5. Update policy after the episode
            # For on-policy algorithms like PPO, updates often happen after collecting
            # a certain number of steps or full episodes.
            agent.update_policy()
            
            episode_rewards.append(total_episode_reward)
            if (i + 1) % config.training.log_interval == 0:
                print(f"  Episode {i + 1}/{script_batch_size} | Total Reward: {total_episode_reward:.2f}")

        # 6. Evaluate planner performance on this batch
        avg_performance = np.mean(episode_rewards)
        print(f"--- End of ACL Step {acl_step + 1} ---")
        print(f"Average performance on this batch: {avg_performance:.2f}")
        
        # 7. Update curriculum difficulty based on performance
        generator.update_difficulty(avg_performance)
        
        # Optional: Save a model checkpoint
        # agent.save_model(f"planner_model_acl_{acl_step}.pth")

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()