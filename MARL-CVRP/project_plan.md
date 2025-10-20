# Project Completion Plan: Adversarial Demand Generator for DVRP

This document outlines the step-by-step plan to build, integrate, and train the components for the MARL-CVRP project. Each major task is a checkbox to track progress.

---

### Phase 1: Implement the Core Simulation Engine

**Goal:** Make the `DynamicCVRPEnv` a fully functional simulation. This involves implementing the logic for how the state (vehicles, demands) changes over time.

**File to Modify:** `env/dynamic_cvrp_env.py`

- [ ] **Implement the `step()` method's core logic:**
    - [ ] **Vehicle Movement:**
        - For each vehicle, identify its next destination from its `current_plan`.
        - Create a helper method `_calculate_distance(pos1, pos2)` for Euclidean distance.
        - If a vehicle is within one time step of its destination, it "arrives".
        - If it's further away, update its `location` by moving it one unit of distance towards the destination.
    - [ ] **State Updates on Arrival:**
        - When a vehicle arrives at a demand location:
            - Update its `location` to match the demand's location exactly.
            - Decrease its `remaining_capacity` by the demand's `quantity`.
            - Change the demand's `status` to `'serviced'`.
            - Remove the serviced demand from the vehicle's `current_plan`.
    - [ ] **Demand State Management:**
        - After all vehicles have moved, iterate through the `pending_demands` list.
        - **Check for Failures:** If `current_time` > `demand.deadline`, change the demand's `status` to `'failed'`.
        - **Clean Up List:** Rebuild the `state.pending_demands` list to filter out any demands marked as `'serviced'`.

---

### Phase 2: Implement the Adversarial Generator Agent ($\mathcal{G}$)

**Goal:** Make the `AdversarialTransformer` process real state information from the environment instead of dummy data.

**File to Modify:** `model/adversarial_transformer.py`

- [ ] **Implement Real Tokenization in the `forward()` method:**
    - [ ] **Feature Engineering:** Define and extract feature vectors for each entity.
        - **Vehicle:** e.g., `[loc_x, loc_y, remaining_capacity, len(current_plan)]`
        - **Demand:** e.g., `[loc_x, loc_y, quantity, time_until_deadline]`
        - **Hotspot:** e.g., `[loc_x, loc_y]`
    - [ ] **Tensor Conversion:** Convert the lists of feature vectors into PyTorch tensors.
        - Handle the edge case where `state.pending_demands` is empty.
    - [ ] **Embedding:** Pass each feature tensor through its corresponding embedding layer (`self.vehicle_embed`, `self.demand_embed`, etc.).
    - [ ] **Sequence Assembly:** Concatenate the `[CLS]` token embedding with the resulting vehicle, demand, and hotspot embeddings to create the final input sequence for the Transformer Encoder.

---

### Phase 3: Implement a Baseline Opponent (Planner Agent $\mathcal{P}$)

**Goal:** Create a simple, rule-based Planner agent to serve as the initial opponent for the Generator. A non-learning heuristic is sufficient to start the training loop.

**File to Modify:** `agents/planner_agent.py`

- [ ] **Implement a Heuristic in the `choose_action()` method:**
    - [ ] **Algorithm:** Implement a greedy "nearest-neighbor" assignment logic.
    - [ ] **Process:**
        - For each vehicle, iterate through all currently unassigned demands.
        - Find the demand that is closest to the vehicle's current location and also satisfies its capacity constraint.
        - Assign this demand to the vehicle's route.
        - Repeat until no more valid demands can be assigned to that vehicle.
    - [ ] **Output:** Package the generated routes into the `PlannerAction` data structure. Return `0.0` for the `log_prob` as it's not used.

- [ ] **Implement the `update()` method:**
    - [ ] This agent does not learn, so the method body should simply be `pass`.

---

### Phase 4: Integrate and Implement the Training Loop

**Goal:** Connect all components, implement a concrete reward signal, and make the `main_train.py` script fully functional.

- [ ] **Implement the Reward Signal:**
    - **File:** `env/dynamic_cvrp_env.py`
    - **Method:** `_calculate_regret()`
    - [ ] **Task:** Replace the dummy return value with a meaningful cost function.
        - **Failure Penalty:** Add a large, constant penalty for every demand that newly transitions to the `'failed'` state in the current time step.
        - **Operational Cost:** Add a smaller cost to represent resource usage (e.g., total distance traveled by all vehicles, or a simpler proxy like a small cost per active vehicle).
        - The method should return the sum of these costs for the current step.

- [ ] **Activate the PSRO Training Logic:**
    - **File:** `scripts/main_train.py`
    - **Method:** `main()`
    - [ ] **Policy Pool Integration:**
        - Uncomment the lines for adding to and sampling from the policy pools.
        - In the "Train Planner" phase, load the latest Generator policy from `policy_pool_G` into a `fixed_policy_G` agent instance using `load_state_dict()`.
        - In the "Train Generator" phase, after the training epochs, save the newly trained policy weights into the pool using `policy_pool_G.add(current_policy_G.model.state_dict())`.
    - [ ] **Connect Rewards:**
        - Ensure the `delta_regret` value from `env.step()` is passed correctly to the agents' `update()` methods.
        - **Planner Reward:** `reward = -delta_regret` (its goal is to minimize cost).
        - **Generator Reward:** `reward = delta_regret` (its goal is to maximize cost).