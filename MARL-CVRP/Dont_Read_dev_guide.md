### **项目指南：基于自动课程学习的鲁棒 DVRP 规划器**

#### 1\. 顶层项目概述 (High-Level Project Overview)

**1.1. 项目目标 (The Goal)**

本项目的**唯一核心目标**是研发并训练一个高性能、高鲁棒性的**“规划器” (Planner) 智能体 $\mathcal{P}$**。

这个 `Planner` ($\mathcal{P}$) 是一个强化学习 (RL) 智能体，它必须学会在**动态车辆路径问题 (DVRP)** 中做出实时的、在线的决策（例如，在新订单到达时重新规划路线），以最小化总成本（如**行驶距离**或**客户等待时间**）。

**1.2. 核心方法论 (The "Big Idea")**

为了让 `Planner` ($\mathcal{P}$) 变得“鲁棒”，我们不能只用简单、随机的环境来训练它。

我们将采用一种**自动课程学习 (Automated Curriculum Learning, ACL)** 的框架。这是一种“**老师-学生**”模式：

  * **学生 (The Planner, $\mathcal{P}$):** 我们的 RL 智能体（**您的研究核心**）。它在模拟环境中进行“内循环”训练，以最小化“遗憾”。
  * **老师 (The Generator, $\mathcal{G}$):** 一个**非学习型**的“难度控制器”。它的工作是在“外循环”中，根据 $\mathcal{P}$ 的“考试成绩”（平均遗憾），**自动调整**下一批训练数据的“难度”。

**1.3. 关键组件的角色 (Key Roles)**

1.  **Planner ($\mathcal{P}$) (agents/):** “学生”。一个复杂的 RL 智能体，它在**运行时 (online)** 做决策，它**看不到**未来的需求。
2.  **Generator ($\mathcal{G}$) (env/):** “老师”。一个**非学习 (non-learning)** 的参数化脚本。它在**训练前 (offline)** 生成一批“需求脚本”（完整的需求列表和各自的出现时间）。
3.  **Environment ($\text{Env}$) (env/):** “教室/考场”。它**执行** `G` 提供的**一个**静态脚本，模拟时间的流逝，并只在正确的时间向 `P` “揭示”新需求。
4.  **Oracle ($\mathcal{O}$) (env/):** “评分者/裁判”。它在 `Env` 的每一步中，计算 `P` 的当前决策与“理论最优解”之间的差距，即\*\*“遗憾” (Regret)\*\*，并将其作为奖励信号（$r_t = -\Delta \text{Regret}_t$）反馈给 `P`。

**1.4. 训练流程：双层循环 (The Training Flow)**

这个架构清晰地分离了“出题”和“答题”：

  * **外循环 (ACL / `scripts/train.py`):**

    1.  `G`（老师）说：“这是当前的 `difficulty = 1`（例如：订单稀疏，时间充裕）”。
    2.  `G` 根据 `difficulty = 1` **生成** 1000 个“需求脚本”（1000 套“试卷”）。
    3.  `P`（学生）进入“**内循环**”，在这 1000 套试卷上训练和学习。
    4.  `P` 训练完毕。`G`（老师）来“评估” `P` 在 `difficulty = 1` 上的平均表现（`avg_regret`）。
    5.  `IF` `P` 的表现很好（`avg_regret` 低于阈值），`G`（老师）说：“你毕业了，现在提升到 `difficulty = 2`（例如：订单密集，时间紧急）”。
    6.  循环回到第 2 步。

  * **内循环 (RL / `scripts/train.py` 中的 `train_planner...`):**

    1.  `Env`（教室）从 `G` 提供的 1000 个脚本中拿**一个** `script`。
    2.  `Env.reset()`，时间 $t=0$。
    3.  `P`（学生）观察 $t=0$ 的状态（没有订单）。
    4.  `Env.step()`，时间 $t=1$。`Env` 检查 `script`，发现 $t=1$ 时没有新订单。`P` 获得 $r_1$。
    5.  ...
    6.  `Env.step()`，时间 $t=10$。`Env` 检查 `script`，发现一个订单在 $t=10$ 到达。`Env` 将这个订单加入 `state`。
    7.  `P`（学生）观察到 $t=10$ 的新 `state`（包含新订单），**它不知道 $t=11$ 是否还会有订单**。
    8.  `P` **在线**决策（例如，输出一个 H 步计划 `action`）。
    9.  `Env` 执行 `action`，并**调用** `Oracle` 计算 `P` 的奖励 $r_{10} = -\Delta \text{Regret}_{10}$。
    10. `P` 将这个 $(state, action, r_{10}, next\_state)$ 存储到经验池中。
    11. 循环直到 `Env` 模拟结束 (例如 $t=480$)。
    12. `P` 使用经验池中的数据，执行一次策略更新（例如 PPO Update）。
    13. 循环回到第 1 步，拿下一个 `script`。

-----

### **项目包（Package）结构指南**

```
robust_dvrp_project/
├── env/                          # (包) 仿真环境与博弈规则
│   ├── __init__.py
│   ├── dvrp_env.py
│   ├── regret_oracle.py
│   └── curriculum_generator.py
│
├── agents/                       # (包) 智能体 (策略与学习)
│   ├── __init__.py
│   └── planner_agent.py
│
├── model/                        # (包) 神经网络架构
│   ├── __init__.py
│   └── planner_model.py
│
├── utils/                        # (包) 共享工具
│   ├── __init__.py
│   ├── config.py
│   ├── data_structures.py
│   └── replay_buffer.py
│
├── scripts/                      # (文件夹) 可执行脚本
│   └── train.py
│
└── configs/                      # (文件夹) 配置文件
    └── base_config.yaml
```

-----

### **各模块详细功能与调用关系**

#### 2\. `utils/` (工具包)

此包提供项目共享的基础数据结构和配置。

  * **`utils/data_structures.py`**

      * **职责:** 定义核心数据对象（推荐使用 `dataclass`）。
      * **内容:**
          * `Demand`: 定义单个需求（ID, 位置, 需求量, 到达时间, 状态等）。
          * `Vehicle`: 定义车辆（ID, 位置, 剩余容量, 当前路线等）。
          * `State`: 定义 `Planner` 在每一步的观测（当前时间, 车辆列表, 待处理需求列表）。
          * `DemandScriptEntry`: 定义需求脚本中的条目（到达时间, 位置, 需求量）。
      * **调用关系:** 被 `env/`, `agents/`, `scripts/` 全局导入和使用。

  * **`utils/config.py`**

      * **职责:** 加载和管理所有配置。
      * **内容:**
          * `EnvConfig`, `PlannerConfig`, `GeneratorConfig`, `ModelConfig` (推荐使用 `dataclass` 或 `pydantic` 定义配置类)。
          * `load_config(path_to_yaml)`:
              * **功能:** 从 YAML 文件读取配置，并将其解析为上述的配置类对象。
      * **调用关系:** 被 `scripts/train.py` 在启动时**调用**。

  * **`utils/replay_buffer.py`**

      * **职责:** 为 on-policy 学习（如 PPO）存储经验。
      * **内容:** `class ReplayBuffer:`
          * `store(state, action_log_prob, reward, done, value)`: 存储一个时间步的经验。
          * `get_data()`: 返回缓冲区中存储的所有数据，用于策略更新。
          * `clear()`: 清空缓冲区（on-policy 算法在每次更新后必须调用）。
      * **调用关系:** 被 `agents/planner_agent.py` **实例化和调用**。

#### 3\. `model/` (模型包)

此包仅包含 `torch.nn.Module` 的网络结构定义，与 RL 逻辑解耦。

  * **`model/planner_model.py`**
      * **职责:** 定义 `Planner` Agent 的神经网络。
      * **内容:** `class PlannerModel(nn.Module):`
          * `__init__(config)`:
              * **功能:** 定义所有网络层。
              * **包含:**
                  * **动态编码器:** 用于处理 `State` 中的动态实体（例如，`VehicleEncoder` (MLP), `DemandEncoder` (GRU/Transformer)）。
                  * **滚动解码器:** 用于输出 H 步规划（例如，`AttentionDecoder` 或 `PointerNetwork`）。
                  * **价值头 (Value Head):** 一个 MLP，用于输出当前状态的价值（`value`），配合 PPO/A2C。
          * `forward(state_tensor)`:
              * **功能:** 执行模型的前向传播。
              * **输入:** 经过预处理的 `State` 张量。
              * **输出:** `action_logits` (动作的对数概率) 和 `state_value` (状态价值)。
      * **调用关系:** 被 `agents/planner_agent.py` **实例化和调用**。

#### 4\. `agents/` (智能体包)

此包实现 `Planner` Agent 的 RL 学习算法。

  * **`agents/planner_agent.py`**
      * **职责:** 实现 `Planner` ($\mathcal{P}$) 的策略逻辑和学习算法（例如 PPO）。
      * **内容:** `class PlannerAgent:`
          * `__init__(config)`:
              * **功能:** 初始化智能体。
              * **调用:**
                  * **实例化** `model/planner_model.py` 中的 `PlannerModel`。
                  * **实例化** `utils/replay_buffer.py` 中的 `ReplayBuffer`。
                  * 定义 `self.optimizer` (例如 `Adam`)。
          * `choose_action(state)`:
              * **功能:** 根据当前状态选择一个 H 步的滚动计划。
              * **调用:** **调用** `self.model.forward(state_tensor)` 获取 `logits` 和 `value`。
              * **逻辑:** 从 `logits` 中采样一个 `action` (动作)，并计算其 `log_prob` (对数概率)。
              * **返回:** `action_plan` (解码后的具体计划), `log_prob`, `value`。
          * `store_transition(state, action_log_prob, reward, done, value)`:
              * **功能:** 将 `train.py` 传来的单步经验存入缓冲区。
              * **调用:** **调用** `self.buffer.store(...)`。
          * `update_policy()`:
              * **功能:** 在一个 episode 结束后，执行 PPO (或 A2C) 策略更新。
              * **调用:**
                  * **调用** `self.buffer.get_data()`。
                  * **调用** `_calculate_gae()` (私有函数) 计算优势 (Advantage)。
                  * 执行 PPO 的损失计算和 `self.optimizer.step()`。
                  * **调用** `self.buffer.clear()`。
          * `_calculate_gae(rewards, values, dones)`:
              * **功能:** (私有) 计算广义优势估计 (Generalized Advantage Estimation)。
          * `save_model()` / `load_model()`:
              * **功能:** 保存和加载模型权重，用于 checkpoint。

#### 5\. `env/` (环境包)

此包定义了 DVRP 世界的运行规则和奖励机制。

  * **`env/curriculum_generator.py`**

      * **职责:** (ACL) 自动课程控制器（“老师”），**非学习型**。
      * **内容:** `class CurriculumGenerator:`
          * `__init__(config)`: 初始化 `self.current_difficulty` (一个字典，如 `{'lambda': 0.1, ...}`)。
          * `get_difficulty()`: 返回当前的难度参数。
          * `update_difficulty(planner_performance)`: **(外循环核心)**
              * **功能:** 比较 `planner_performance` (例如平均遗憾) 与 `acl_threshold`。
              * **逻辑:** 如果 `Planner` 表现达标，则提升 `self.current_difficulty` (例如，提高 `lambda`，降低 `L_mean`)。
          * `generate_script_batch(batch_size)`:
              * **功能:** 生成一批用于训练 `Planner` 的需求脚本（“试卷”）。
              * **调用:** 循环**调用** `_generate_single_script()`。
          * `_generate_single_script()`:
              * **功能:** (私有) 根据**当前难度参数**，使用随机过程（如泊松、高斯）生成一个完整的 `DemandScriptEntry` 列表。

  * **`env/regret_oracle.py`**

      * **职责:** (Critic) “遗憾”评估器（“评分者”），计算 `step`-wise 奖励。
      * **内容:** `class RegretOracle:`
          * `__init__(config)`: (可选) 初始化一个离线静态求解器（如 Gurobi 或 LKH）。
          * `calculate_step_reward(env_state, serviced_list, failed_list)`: **(内循环核心)**
              * **功能:** 计算 $r_t = -\Delta \text{Regret}_t$。
              * **调用:**
                  * **调用** `_get_actual_cost(env_state)` (私有) 获取实际成本。
                  * **调用** `_get_ideal_optimal_cost(all_known_demands)` (私有) 获取理论最优成本。
              * **逻辑:** 计算 $\text{Regret}(t) = \text{Actual}(t) - \text{Ideal}(t)$，然后 $\Delta \text{Regret}_t = \text{Regret}(t) - \text{Regret}(t-1)$。
              * **返回:** `reward = -delta_regret`。
          * `_get_ideal_optimal_cost(demands)`:
              * **功能:** (私有) **[项目难点]** 运行静态 CVRP 求解器，计算在 `t` 时刻所有已知需求的理论最优解。
          * `_get_actual_cost(env_state)`:
              * **功能:** (私有) 计算到目前为止的实际总成本（例如总行驶距离）。

  * **`env/dvrp_env.py`**

      * **职责:** 核心仿真器（“教室”），**执行**由 `Generator` 传入的**静态脚本**。
      * **内容:** `class DvrpEnv:`
          * `__init__(oracle, config)`: 存储 `oracle` 实例和配置。
          * `reset(demand_script)`:
              * **功能:** 重置环境以开始新 episode。
              * **逻辑:** 存储 `demand_script`，重置 `self.current_time = 0`，初始化车辆和空的需求列表。
              * **调用:** **调用** `_reveal_new_demands()` 来加载 $t=0$ 时的需求。
              * **返回:** 初始的 `State` 对象。
          * `step(planner_action)`:
              * **功能:** 在环境中推进一个时间步。
              * **逻辑:**
                1.  `self.current_time += 1`。
                2.  **调用** `_update_vehicles(planner_action)` (私有) 执行车辆动作。
                3.  **调用** `_reveal_new_demands()` (私有) 检查脚本，揭示新到达的需求。
                4.  **调用** `self.oracle.calculate_step_reward(...)` 获取 `reward`。
                5.  检查 `done` 条件 (是否达到 `sim_duration`)。
              * **返回:** `(next_state, reward, done)`。
          * `_reveal_new_demands()`: (私有) 检查 `self.full_demand_script`，将 `arrival_time == current_time` 的需求移入 `self.pending_demands`。
          * `_update_vehicles(action)`: (私有) 更新车辆位置、负载，处理服务。
          * `_get_current_state()`: (私有) 打包当前的车辆和待处理需求，返回 `State` 对象。

#### 6\. `scripts/` (可执行脚本)

此文件夹是项目的入口点，负责编排整个训练流程。

  * **`scripts/train.py`**
      * **职责:** 启动和管理“外循环”(ACL) 和“内循环”(RL)。
      * **内容:**
          * `main()`:
              * **功能:** 主函数。
              * **逻辑:**
                1.  **调用** `utils.config.load_config()` 加载配置。
                2.  **初始化**所有核心组件：`RegretOracle`, `CurriculumGenerator`, `DvrpEnv`, `PlannerAgent`。
                3.  **启动外循环 (ACL)** (`for outer_loop ...`)：
                    a. **调用** `generator.get_difficulty()`。
                    b. **调用** `generator.generate_script_batch()`。
                    c. **调用** `train_planner_on_batch()` (见下文) 并获取 `avg_performance`。
                    d. **调用** `generator.update_difficulty(avg_performance)`。
                    e. (可选) **调用** `planner.save_model()`。
          * `train_planner_on_batch(env, planner, script_batch)`:
              * **功能:** **(内循环 RL)** 在一个批次的脚本上训练 `Planner`。
              * **逻辑:**
                1.  `for script in script_batch:` (遍历批次中的每个脚本)
                2.  **调用** `env.reset(script)`。
                3.  `while not done:` (运行一个 episode)
                    a. **调用** `planner.choose_action(state)` 获取 `action`, `log_prob`, `value`。
                    b. **调用** `env.step(action)` 获取 `next_state`, `reward`, `done`。
                    c. **调用** `planner.store_transition(...)`。
                4.  (Episode 结束) **调用** `planner.update_policy()`。
              * **返回:** `Planner` 在这个 `script_batch` 上的平均性能（例如平均总遗憾）。
          * `evaluate_planner(...)`: (可选) 一个独立的评估函数，用于在不更新策略的情况下测试 `Planner` 性能。

-----

### **开发路线图 (Development Roadmap)**

**Step 1: Configuration and Utilities (`configs/` & `utils/`)**
*   **`configs/base_config.yaml`**: Define a YAML file with placeholders for all necessary parameters (e.g., environment settings, model hyperparameters, training options).
*   **`utils/config.py`**: Create a Python script to load, parse, and provide easy access to the parameters from the YAML file using dataclasses.
*   **`utils/data_structures.py`**: Define core data classes like `Demand`, `Vehicle`, and `State` to ensure consistent data representation across the project.

**Step 2: Environment Core (`env/`)**
*   **`env/dvrp_env.py`**: Implement the main `DvrpEnv` class following a standard RL environment structure (`reset`, `step`).
*   **`env/curriculum_generator.py`**: Create the class for generating demand scripts based on difficulty parameters.
*   **`utils/replay_buffer.py`**: Implement a basic replay buffer for on-policy learning.

**Step 3: Model and Agent Abstractions (`model/` & `agents/`)**
*   **`model/planner_model.py`**: Define a placeholder `PlannerModel` class using `torch.nn.Module`.
*   **`agents/planner_agent.py`**: Create the `PlannerAgent` class structure, including methods for action selection (`choose_action`), storing transitions, and updating the policy (`update_policy`).

**Step 4: Training Infrastructure (`scripts/`)**
*   **`scripts/train.py`**: Write the main training script to orchestrate the outer and inner training loops, connecting all the components.

**Step 5: Advanced Components (`env/`)**
*   **`env/regret_oracle.py`**: Create a placeholder for the `RegretOracle` to calculate step-wise rewards. This can be integrated later with a dedicated CVRP solver.

---

### **Solving Part of TODOs: Next Development Focus**

This section outlines the high-level plan for addressing the immediate TODOs related to demand generation and curriculum difficulty.

#### **[DONE]1. Demand Quantity Configuration**

**Goal:** To make the range for generating demand quantities configurable, rather than hardcoded.

**High-Level Guidance:**

1.  **Configuration File Update:** Add two new parameters, `min_demand_quantity` and `max_demand_quantity`, to the `generator` section within `configs/base_config.yaml`. These will define the lower and upper bounds for the quantity of a single demand.
2.  **Config Class Update:** Modify the `GeneratorConfig` dataclass in `utils/config.py` to include these new `min_demand_quantity` and `max_demand_quantity` fields. This ensures they are correctly loaded from the YAML.
3.  **Generator Logic Update:** In the `_generate_single_script` method of `env/curriculum_generator.py`, replace the hardcoded `np.random.randint(1, 10)` with a call that uses the newly configured `self.gen_config.min_demand_quantity` and `self.gen_config.max_demand_quantity`. Remember that `np.random.randint` is exclusive on the high end.

#### **[TO BE DETERMINED]2. CurriculumGenerator Difficulty Design & Update Logic**

**Goal:** To enhance the `CurriculumGenerator` by introducing more dimensions to problem difficulty and refining how these difficulties are updated based on planner performance.

**High-Level Guidance:**

1.  **Expand Difficulty Parameters:**
    *   **Identify New Dimensions:** Beyond just `lambda` (demand arrival rate), consider adding parameters that influence the spatial distribution of demands. A good candidate is `demand_spatial_spread`, which could control how clustered or dispersed demands are on the map.
    *   **Configuration File Update:** Add `demand_spatial_spread` (and any other chosen new parameters) to the `generator.initial_difficulty` section in `configs/base_config.yaml` with appropriate initial values.
    *   **Config Class Update:** Ensure the `initial_difficulty` dictionary in `GeneratorConfig` (within `utils/config.py`) is prepared to handle these new keys。

2.  **Utilize New Parameters in Script Generation (`_generate_single_script`):**
    *   **Location Generation:** Modify the demand `location` generation logic in `env/curriculum_generator.py`. Instead of a uniform random distribution, use a Gaussian (normal) distribution centered on the map, with its standard deviation controlled by the `demand_spatial_spread` parameter. Remember to clip the generated coordinates to stay within the `map_size` boundaries.
    *   *(Note: Parameters like `vehicle_capacity_factor` or `num_vehicles_factor` would typically influence the `DvrpEnv`'s initialization rather than script generation directly. These can be considered for later, more advanced difficulty control.)*

3.  **Refine Difficulty Update Logic (`update_difficulty`):**
    *   **Clarify Performance Metric:** Before refining, ensure the `planner_performance` metric (currently `np.mean(episode_rewards)`) is meaningful. It should represent a cost (e.g., average total travel distance) that the planner aims to minimize. This will depend on the actual reward function implemented in `RegretOracle`.
    *   **Redefine `acl_threshold`:** The `acl_threshold` should be interpreted as a target performance value. If the `planner_performance` (e.g., average total travel distance) falls *below* this threshold, it indicates good performance.
    *   **Design Multi-Parameter Update Strategy:** When the planner's performance meets the `acl_threshold`, adjust *multiple* difficulty parameters simultaneously. For example:
        *   Increase `lambda` (more frequent demands).
        *   Increase `demand_spatial_spread` (demands are more spread out).
        *   Ensure these adjustments are gradual and progressive to maintain a smooth learning curve.
    *   *(Important: The specific values for `acl_threshold` and the exact update rules will require tuning once the planner and reward function are more developed。）*