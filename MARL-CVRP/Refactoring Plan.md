# Refactoring Plan: From Real-Time to Offline Instance Generation

**Project:** MARL-CVRP

**Date:** October 21, 2025

**Goal:** Modify the project to an offline adversarial training paradigm. The Generator will create a complete, dynamic CVRP instance (a schedule of demands over time) at the start of each episode. The Planner will then solve this pre-defined instance.

---

### **Strategic Rationale: The Adversarial "Gym"**

The core purpose of this refactoring is to create a more powerful and stable training environment for a sophisticated **Planner Agent**. The key distinction to understand is:

*   **Offline Generation (The Generator's Role):** The Generator acts like a "dungeon master," creating a complete, challenging scenario (the `demand_schedule`) *before* the episode begins. Its reward is based on the Planner's final performance, allowing it to learn what constitutes a holistically difficult problem.

*   **Online Experience (The Planner's Role):** The Planner agent experiences the problem in a fully dynamic, "online" fashion. It does **not** see the future. At each timestep, it only receives the current state (newly revealed demands, vehicle locations, etc.) and must make decisions with incomplete information.

This framework effectively turns the Generator into an intelligent curriculum designer. It will learn to find the Planner's weaknesses and generate targeted scenarios to train against them. This is essential for developing a robust, generalizable dVRP solver, such as the complex transformer-based model you plan to implement. The refactored environment is agnostic to the Planner's architecture; it simply provides the dynamic state at each step, making it a perfect "gym" to train and benchmark any Planner agent.

---

## Pre-flight Check: Data Structures

*   **Confirm `Demand` class:** Ensure the `Demand` class in `utils/data_structures.py` has a `status: str` field. This is critical for tracking a demand's lifecycle from `'scheduled'` to `'pending'`, `'serviced'` or `'failed'`. If not present, it must be added first.

---

## High-Level Plan

1.  **Refactor the Environment:** The environment will no longer interact with the Generator in its `step` function. It will be initialized with a full demand schedule and will "reveal" demands as time progresses.
2.  **Refactor the Generator Agent:** The Generator's role will change from making a single decision at each timestep to generating an entire sequence of demands in one call. The existing `AdversarialTransformer` will be temporarily disabled.
3.  **Update the Main Training Loop:** The training script will be restructured from the current PSRO loop to a simpler `Generate -> Solve -> Update` cycle for each episode.
4.  **Future Work - Redesign the Generator Model:** The current `AdversarialTransformer` is designed for single-step decisions. A new, sequence-generating model (like an RNN or an auto-regressive Transformer) will be needed to learn how to create challenging instances.

---

## Phase 1: Environment Refactoring (`env/dynamic_cvrp_env.py`)

**Objective:** Decouple the environment from the Generator and make it manage a pre-defined schedule of demands.

**Tasks:**

1.  **Modify `__init__`:**
    *   Add a new instance attribute: `self.demand_schedule: List[Demand] = []`.

2.  **Create a New Private Method `_reveal_new_demands(self)`:**
    *   This method should iterate through `self.demand_schedule` (which should be sorted by `arrival_time`).
    *   Any demand where `demand.arrival_time <= self.state.current_time` should be moved from `self.demand_schedule` to `self.state.pending_demands`.
    *   Update the status of the revealed demand from `'scheduled'` to `'pending'`.

3.  **Modify `reset(self, ...)`:**
    *   Change the signature from `reset(self) -> GlobalState:` to `reset(self, demand_schedule: List[Demand]) -> GlobalState:`.
    *   Inside `reset`, set `self.demand_schedule` to the provided `demand_schedule` and sort it by `arrival_time`.
    *   Initialize `self.state.pending_demands` as an empty list.
    *   Call `self._reveal_new_demands()` at the end of `reset` to activate any demands that start at `t=0`.

4.  **Modify `step(self, ...)`:**
    *   Change the signature from `step(self, action_P: PlannerAction, action_G: GeneratorAction)` to `step(self, planner_actions: PlannerAction)`.
    *   **Remove** the section of code that processes `action_G` and adds new demands.
    *   After advancing `self.state.current_time`, add a call to `self._reveal_new_demands()`.

---

## Phase 2: Generator Agent Refactoring (`agents/generator_agent.py`)

**Objective:** Change the Generator's core function to produce a complete problem instance, temporarily disabling the neural network.

**Tasks:**

1.  **Remove Obsolete Methods:** The step-by-step decision-making logic is no longer needed. Delete the following methods:
    *   `choose_action(self, ...)`
    *   `_generate_hotspots(self, ...)`
    *   `store_reward(self, ...)`

2.  **Remove Obsolete Attributes:** The REINFORCE-specific attributes are no longer needed. Remove these from `__init__`:
    *   `self.saved_log_probs`
    *   `self.rewards`

3.  **Create a New Primary Method `generate_instance(self, ...)`:**
    *   Define the signature: `generate_instance(self, num_demands: int, max_time: int, max_qty: float, map_size: tuple, ...) -> List[Demand]:`.
    *   **Initial Implementation (Placeholder):** Implement a simple, random generator. This version will **not** use `self.model`. For `num_demands`, create a `Demand` object with:
        *   Random location `(x, y)` within `map_size`.
        *   Random `quantity` up to `max_qty`.
        *   Random `arrival_time` between `0` and `max_time`.
        *   A `deadline` calculated from `arrival_time` + a random delay.
        *   A new status: `status='scheduled'`.
    *   Return the complete list of generated `Demand` objects.

4.  **Modify the `update(...)` method:**
    *   Change the signature from `update(self)` to `update(self, final_planner_performance: float)`.
    *   The update logic will now be based on the final outcome of an entire episode.
    *   **Important:** The existing REINFORCE logic is incompatible. For now, make this a placeholder. The actual update logic will be implemented in Phase 4 with the new sequence model.
    ```python
    # Placeholder update method
    def update(self, final_planner_performance: float):
        # This method will be implemented with a sequence-based policy gradient
        # in a future phase. The 'final_planner_performance' will serve as the
        # reward for the entire generated sequence.
        pass
    ```

---

## Phase 3: Training Loop Refactoring (`scripts/main_train.py`)

**Objective:** Restructure the main training loop from PSRO to a simpler paradigm for validating the new environment structure.

**Tasks:**

1.  **Define Performance Metric:** Create a helper function to evaluate the planner's performance on a completed instance. This score will be the reward signal for the Generator. A higher score should mean worse performance.
    ```python
    def calculate_final_performance(final_state: GlobalState) -> float:
        """Calculates a performance score. Higher is worse."""
        # Example: Combine total travel distance and a penalty for failed demands.
        total_distance = sum(calculate_route_distance(v.route) for v in final_state.vehicles)
        failure_penalty = len(final_state.failed_demands) * 100 # High penalty
        return total_distance + failure_penalty
    ```

2.  **Rewrite the main episode loop:** Replace the complex PSRO iteration logic with a straightforward episode loop.

    ```python
    # Conceptual Code for the new loop in main()
    for episode in range(num_episodes):
        # 1. GENERATE: Generator creates the entire problem instance.
        demand_schedule = generator_agent.generate_instance(...)

        # 2. SOLVE: Planner attempts to solve the instance.
        state = env.reset(demand_schedule=demand_schedule)
        done = False
        while not done:
            # Planner acts based on the current state.
            # This is where your future transformer-based solver will be called.
            planner_action = planner_agent.choose_action(state)
            
            # Environment steps forward, revealing demands from the schedule online.
            next_state, reward, done, info = env.step(planner_action)
            
            # For a learning planner, store experience for its own update
            # planner_agent.store_transition(state, planner_action, reward, next_state, done)
            state = next_state

        # 3. UPDATE: Both agents are updated based on the episode's outcome.
        
        # Calculate a final performance score for the planner on the instance.
        final_planner_performance = calculate_final_performance(env.state)

        # Update the planner using its collected experience (if it's a learning agent).
        # planner_agent.update()
        
        # Update the generator. Its "reward" is the planner's poor performance.
        generator_reward = final_planner_performance 
        generator_agent.update(generator_reward) 
    ```

---

## Phase 4: Future Work - Redesigning the Generator Model

**Objective:** Replace the random instance generator with a learned, auto-regressive Transformer model capable of generating sequences of challenging demands.

### Model Architecture: Auto-regressive Transformer Decoder

The model will generate the properties of one demand at a time, conditioning the generation of the next demand on all previously created ones. This allows it to learn complex relationships and structures that make an instance difficult.

*   **Core Idea:** The model is a sequence-to-sequence model that takes a "start" signal and outputs a sequence of demand specifications.
*   **Components:**
    1.  **Input Embedding:** A trainable embedding for a special `[SOS]` (Start of Sequence) token that kicks off generation. The output of each generation step (a demand's properties) is also passed through a linear layer to be embedded into the correct dimension for the next step.
    2.  **Transformer Decoder:** A stack of standard Transformer Decoder layers. Each layer uses masked self-attention to ensure that when predicting demand `i`, it can only attend to the previously generated demands `0` to `i-1`.
    3.  **Output Heads:** After the final decoder layer, the model uses separate linear layers ("heads") to predict the properties for the *next* demand in the sequence. Each head outputs the parameters of a distribution:
        *   **Location `(x, y)`:** A head that outputs the mean and standard deviation for a 2D Normal distribution. We can clip or transform the output to stay within the map boundaries.
        *   **Quantity:** A head that outputs parameters for a Bounded Normal or Log-Normal distribution to ensure positivity.
        *   **Arrival Time:** To ensure time is always moving forward, the model should predict a *delay* (`delta_t`) from the previous demand's arrival time. `arrival_time_i = arrival_time_{i-1} + delta_t_i`. The head predicts the distribution for `delta_t_i`.
        *   **Deadline:** Similarly, the model predicts a `deadline_delay` relative to the new demand's arrival time. `deadline_i = arrival_time_i + deadline_delay_i`.

*   **Generation Process (Auto-regressive Loop):**
    1.  Feed the `[SOS]` embedding into the decoder.
    2.  Generate the properties for `Demand_0` by sampling from the output distributions.
    3.  Embed the properties of `Demand_0` and feed it as the input for the next step.
    4.  The decoder now attends to `[SOS]` and `Demand_0` to generate `Demand_1`.
    5.  Repeat for the desired number of demands in the instance.

### Training the Auto-regressive Generator

**The Challenge:** We don't have a "correct" difficult instance to train against (like in supervised learning). We only know how well the planner performed on the generated instance *after* the episode is over.

**The Solution: Policy Gradient (REINFORCE):** This is a Reinforcement Learning problem. The Generator is an RL agent, its "action" is the entire generated sequence of demands, and its "reward" is the final planner performance score.

**Training Loop Steps:**

1.  **Sample (Rollout):**
    *   During the generation process described above, for each demand property, we sample from the distribution predicted by the model.
    *   Crucially, we must also calculate and store the **log-probability** of the values we sampled.
    *   The total log-probability for the entire instance is the sum of the log-probabilities of all the individual properties sampled: `log P(Instance) = Σ log p(property)`.

2.  **Evaluate:**
    *   The generated `demand_schedule` is passed to the environment.
    *   The Planner agent runs for a full episode to solve it.
    *   We calculate the `final_planner_performance` score. This single scalar value is our reward, `R`. Remember, higher score = worse planner performance = higher reward for the generator.

3.  **Calculate Loss:**
    *   The policy gradient loss is calculated as: `Loss = -log P(Instance) * Advantage`.
    *   The `Advantage` is `(R - b)`, where `b` is a **baseline**.
    *   A high positive advantage (meaning the instance was much harder than average) combined with the negative sign in the loss function will push the model to increase the probability of generating that kind of instance again.

4.  **Update:**
    *   Call `loss.backward()` to compute gradients.
    *   Call `optimizer.step()` to update the weights of the Transformer Decoder.

**Key Technique: Baseline for Variance Reduction**

*   The reward `R` can vary wildly between episodes, making training unstable. A baseline `b` helps stabilize this.
*   A simple and effective baseline is an **exponential moving average (EMA) of past rewards**.
*   Before calculating the loss, you update the baseline: `b = alpha * b + (1 - alpha) * R`.
*   This baseline represents the "average" difficulty the generator has been creating. The advantage `(R - b)` then measures if the most recent instance was surprisingly hard or surprisingly easy, which is a much better learning signal.

---

## Phase 5: Cleanup and Verification

**Objective:** Ensure the refactored code works and remove obsolete components.

**Tasks:**

1.  **Create a Test Script:** Write a simple script (`scripts/test_offline_env.py`) that:
    *   Initializes the `GeneratorAgent` (with the random generator).
    *   Calls `generate_instance()` to get a schedule.
    *   Initializes the `DynamicCVRPEnv`.
    *   Calls `env.reset()` with the generated schedule.
    *   Runs `env.step()` in a loop, asserting at each step `t` that any newly pending demand `d` satisfies `d.arrival_time <= t`.
2.  **Review and Remove:** Check for any helper functions or logic in `main_train.py` related to the old real-time PSRO loop that are no longer used and can be deleted.