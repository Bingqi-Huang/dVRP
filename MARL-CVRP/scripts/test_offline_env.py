"""
Test Script for Verifying the Offline Generation and Dynamic Environment.

This script performs a single episode run to validate that:
1. The GeneratorAgent can produce a valid demand schedule.
2. The DynamicCVRPEnv can be reset with this schedule.
3. The environment correctly reveals demands from the schedule as time progresses
   during the `step` function calls.
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from env.dynamic_cvrp_env import DynamicCVRPEnv
from agents.generator_agent import GeneratorAgent
from agents.planner_agent import PlannerAgent

def run_verification_test():
    """
    Executes the verification test.
    """
    print("="*50)
    print("🚀 Starting Offline Environment Verification Test...")
    print("="*50)

    # --- 1. Initialization ---
    print("[1/4] Initializing agents and environment...")
    try:
        env = DynamicCVRPEnv()
        # Use the learning-based generator for the test
        generator_agent = GeneratorAgent() 
        # The planner is a simple heuristic, which is fine for this test
        planner_agent = PlannerAgent() 
        print("   ✅ Initialization successful.")
    except Exception as e:
        print(f"   ❌ ERROR during initialization: {e}")
        return

    # --- 2. Instance Generation ---
    print("\n[2/4] Generating a test instance (demand schedule)...")
    try:
        demand_schedule = generator_agent.generate_instance(
            num_demands=20,
            episode_length=env.episode_length,
            max_quantity=15.0
        )
        assert isinstance(demand_schedule, list) and len(demand_schedule) > 0
        print(f"   ✅ Generated a schedule with {len(demand_schedule)} demands.")
        # Optional: Print a few demands to inspect
        for i in range(min(3, len(demand_schedule))):
            d = demand_schedule[i]
            print(f"     - Demand {d.id}: Arrival={d.arrival_time}, Deadline={d.deadline}")

    except Exception as e:
        print(f"   ❌ ERROR during instance generation: {e}")
        return

    # --- 3. Environment Reset ---
    print("\n[3/4] Resetting environment with the generated schedule...")
    try:
        state = env.reset(demand_schedule=demand_schedule)
        print("   ✅ Environment reset successful.")
        print(f"   Initial pending demands at t=0: {len(state.pending_demands)}")
    except Exception as e:
        print(f"   ❌ ERROR during environment reset: {e}")
        return

    # --- 4. Episode Rollout and Verification ---
    print("\n[4/4] Running episode and verifying demand reveals at each step...")
    done = False
    step_count = 0
    try:
        while not done:
            current_time = state.current_time
            
            # Core Assertion: Check if all currently pending demands are valid
            for demand in state.pending_demands:
                assert demand.arrival_time <= current_time, \
                    f"VIOLATION at t={current_time}: Demand {demand.id} is pending but its arrival time is {demand.arrival_time}!"

            # Planner acts
            planner_action = planner_agent.choose_action(state)
            
            # Environment steps
            state, reward, done, info = env.step(planner_action)
            step_count += 1
        
        print(f"\n   ✅ Episode completed successfully in {step_count} steps.")
        print("   ✅ All pending demands were revealed at or before the correct timestep.")

    except AssertionError as e:
        print(f"\n   ❌ VERIFICATION FAILED: {e}")
        return
    except Exception as e:
        print(f"\n   ❌ ERROR during episode rollout: {e}")
        return
        
    print("\n" + "="*50)
    print("🎉 Verification Test Passed! The core logic is sound.")
    print("="*50)


if __name__ == "__main__":
    run_verification_test()