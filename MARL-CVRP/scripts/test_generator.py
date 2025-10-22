import numpy as np
import matplotlib.pyplot as plt
from typing import List

from utils.config import load_config, EnvConfig
from env.curriculum_generator import CurriculumGenerator
from utils.data_structures import DemandScriptEntry

def visualize_script(script: List[DemandScriptEntry], env_config: EnvConfig, title: str):
    """
    Visualizes the demands from a single script on a 2D plot.
    The color and label of each point indicate its arrival order.
    """
    if not script:
        print("Script is empty, skipping visualization.")
        return

    locations = np.array([d.location for d in script])
    arrival_times = np.array([d.arrival_time for d in script])
    
    plt.figure(figsize=(10, 10))
    
    # Plot depot at the center
    depot_loc = (env_config.map_size / 2, env_config.map_size / 2)
    plt.scatter(depot_loc[0], depot_loc[1], c='red', marker='*', s=250, label='Depot', zorder=5)
    
    # Plot demands with color representing arrival time
    scatter = plt.scatter(
        locations[:, 0], 
        locations[:, 1], 
        c=arrival_times, 
        cmap='cividis', 
        s=60, 
        label='Demands'
    )
    
    # Add text labels for the order of arrival and quantity
    for i, loc in enumerate(locations):
        # Display format: "Order (Quantity)"
        plt.text(loc[0] + 1, loc[1] + 1, f"{i + 1} ({script[i].quantity})", fontsize=9, ha='center')
        
    # Add a colorbar to show the time scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Arrival Time')
    
    # Formatting
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, env_config.map_size)
    plt.ylim(0, env_config.map_size)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def test_generator():
    """
    A standalone script to test the CurriculumGenerator's functionality.
    """
    print("--- Testing Curriculum Generator ---")

    # 1. Initialize the generator
    config = load_config('configs/base_config.yaml')
    generator = CurriculumGenerator(config.generator, config.env)
    
    print(f"Initial difficulty: {generator.get_difficulty()}")

    # 2. Test initial script generation
    print("\n1. Generating a batch with initial difficulty...")
    initial_batch = generator.generate_script_batch(batch_size=100)
    
    # Calculate and print a simple metric: average number of demands
    avg_demands_initial = np.mean([len(script) for script in initial_batch])
    print(f"-> Average number of demands per script: {avg_demands_initial:.2f}")
    assert len(initial_batch) == 100

    # Visualize one of the "easy" scripts
    visualize_script(initial_batch[0], config.env, "Example Episode (Initial Difficulty)")

    # 3. Test the difficulty update mechanism with a "good" performance score
    # The threshold is 0.1, and the logic is `performance < threshold`.
    # So, a large negative number represents very good performance.
    print("\n2. Simulating GOOD planner performance to trigger a difficulty update...")
    good_performance = -50.0 
    print(f"   (Using dummy performance score: {good_performance})")
    generator.update_difficulty(good_performance)
    
    print(f"Updated difficulty: {generator.get_difficulty()}")

    # 4. Test script generation with the NEW, harder difficulty
    print("\n3. Generating a batch with the new, harder difficulty...")
    harder_batch = generator.generate_script_batch(batch_size=100)
    avg_demands_harder = np.mean([len(script) for script in harder_batch])
    print(f"-> Average number of demands per script: {avg_demands_harder:.2f}")

    # Visualize one of the "harder" scripts
    visualize_script(harder_batch[0], config.env, "Example Episode (Increased Difficulty)")

    # We expect the average number of demands to increase with a higher lambda
    assert avg_demands_harder > avg_demands_initial
    print("-> CHECK PASSED: Difficulty increase resulted in more demands.")

    # 5. Test the difficulty update mechanism with a "bad" performance score
    print("\n4. Simulating BAD planner performance to check if difficulty stays the same...")
    bad_performance = 20.0 # This is greater than the 0.1 threshold
    print(f"   (Using dummy performance score: {bad_performance})")
    difficulty_before_bad_update = generator.get_difficulty().copy()
    generator.update_difficulty(bad_performance)
    difficulty_after_bad_update = generator.get_difficulty()
    
    assert difficulty_before_bad_update == difficulty_after_bad_update
    print("-> CHECK PASSED: Difficulty did not change with bad performance.")

    print("\n--- Generator Test Finished Successfully ---")


if __name__ == '__main__':
    test_generator()