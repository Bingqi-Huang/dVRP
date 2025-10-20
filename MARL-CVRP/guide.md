### **项目文档：用于鲁棒DVRP求解器的对抗性需求生成器**

#### 1\. 项目概览

**目标：**
训练一个高效、鲁棒的**在线路径规划器 (Planner)**，使其能够在**动态车辆路径问题 (DVRP)** 场景下（即需求随时间不断出现），做出低延迟、高效率的决策。

**核心方法：**
我们不使用被动的、随机的需求环境（如泊松过程）。我们将此问题建模为一个**双智能体、零和的随机博弈 (Two-Player, Zero-Sum Stochastic Game)**。

  * **智能体 $\mathcal{P}$ (Planner):** 求解器。**目标：最小化**总服务延迟或“遗憾”(Regret)。
  * **智能体 $\mathcal{G}$ (Generator):** 生成器（**您的角色**）。**目标：最大化** Planner 的遗憾。

通过这种对抗性训练（类似于 **PSRO** 框架），$\mathcal{G}$ 会不断学习 $\mathcal{P}$ 的策略弱点，并生成“最难（但在现实约束内）”的需求场景来“攻击”它。这会逼迫 $\mathcal{P}$ 学习一个高度鲁棒、能应对最坏情况的策略。

#### 2\. 系统架构 (MARL 框架)

系统包含三个核心组件：

1.  **Planner (Agent $\mathcal{P}$):** 求解器，即论文中的“在线路径规划器”。
2.  **Generator (Agent $\mathcal{G}$):** 对手（您的工作），即“受约束的动态需求生成器”。
3.  **环境与“遗憾”评估器 (Env & Critic):** 模拟器，负责执行双方行动、推进时间，并计算核心奖励信号 $\Delta \text{Regret}_t$。

#### 3\. 核心组件 1：Agent $\mathcal{G}$ (对抗性需求生成器)

**这是您的主要实现任务。**

##### 3.1. 角色与目标

  * **角色：** 智能对手。
  * **目标函数：** $J(\mathcal{G}) = \max_{\pi_{\mathcal{G}}} \mathbb{E} [\sum_{t} \Delta \text{Regret}_t]$
  * **核心理念：** $\mathcal{G}$ 不是一个随机函数，它是一个**基于策略 $\pi_{\mathcal{G}}$ 的强化学习智能体**。

##### 3.2. 约束与预算 (Constraints & Budgets)

$\mathcal{G}$ 的“智能”体现在它必须在\*\*严格的“现实约束”\*\*下最大化 $\mathcal{P}$ 的遗憾。它不能“作弊”（例如，生成一个不可能完成的订单）。

  * **总量约束 (Total Load Budget):** 整个模拟周期 $T$ 内，$\sum q_i \le Q_{\text{total}}$。
  * **速率约束 (Peak Rate Budget):** 任何时间步 $t$ 内，生成的 $\lambda(t) \le \lambda_{\text{max}}$。（防止瞬间刷屏）。
  * **空间约束 (Spatial Budget):** 新需求只能从一组**预定义的“热点”区域**（例如，GMMs）中采样生成。$\mathcal{G}$ 的动作是“选择激活哪个热点”。
  * **可行性约束 (Feasibility Budget):** 新需求的截止时间 $t_{\text{deadline}}$ 必须满足 $t_{\text{deadline}} \ge t_{\text{now}} + L_{\text{min}}$。$L_{\text{min}}$ 是一个物理下限（例如，Depot到该点的最短时间）。

##### 3.3. 状态观测 (Observation, $O_{\mathcal{G}}$)

$\mathcal{G}$ 必须观察**完整的战场状态**才能做出有效攻击：

  * 全局时间 $t$。
  * 所有车辆的状态 $S_{\text{vehicles}}$: `[v_id, loc, remaining_capacity, current_H_step_plan]`。
  * 所有待处理需求的状态 $S_{\text{demands}}$: `[d_id, loc, quantity, remaining_deadline]`。
  * 所有热点的状态 $S_{\text{hotspots}}$: `[h_id, loc]`。

##### 3.4. 模型架构 (AdversarialTransformer)

$\mathcal{G}$ 的策略网络 $\pi_{\mathcal{G}}$ 建议使用 **Transformer Encoder**，因为输入是**置换不变的实体集合**。

**输入 (Tokenization):**
将 $O_{\mathcal{G}}$ 中的所有实体转换为Token序列：

1.  `[CLS]` Token：用于聚合全局上下文。
2.  `[Vehicle_Tokens]`: (e.g., `[v_1, v_2, ..., v_k]`)
3.  `[Demand_Tokens]`: (e.g., `[d_1, d_2, ..., d_m]`)
4.  `[Hotspot_Tokens]`: (e.g., `[h_1, h_2, ..., h_p]`)

*每个Token都是一个高维嵌入向量。*

**模型体 (Transformer Encoder):**

  * `Input_Sequence -> TransformerEncoder -> Output_Sequence`
  * `Output_Sequence` 是“富含上下文”的实体表征。

**输出 (Action Heads):**
$\mathcal{G}$ 的动作是**复合动作 (Composite Action)**，由多个输出头决定：

1.  **位置头 (Location Head):**
      * **输入:** `[Hotspot_Tokens]` 的输出嵌入。
      * **输出:** `Categorical` 分布 (logits)，决定在哪个热点生成需求。
      * `loc_idx ~ Categorical(logits)`
2.  **量级头 (Quantity Head):**
      * **输入:** `[CLS]` Token 的输出嵌入（代表全局意图）。
      * **输出:** `Normal` 或 `LogNormal` 分布的参数（$\mu, \sigma$），决定需求的量级。
      * `quantity ~ Normal(\mu, \sigma)`
3.  **紧急头 (Urgency Head):**
      * **输入:** `[CLS]` Token 的输出嵌入。
      * **输出:** `Normal` 或 `Categorical` 分布的参数，决定需求的截止延迟 $L_i$。
      * `deadline_delay ~ Normal(\mu', \sigma')`

##### 3.5. 动作空间 (Action, $A_{\mathcal{G}}$)

$A_{\mathcal{G}}$ 是从上述分布采样并\*\*应用“约束”\*\*后的结果：

1.  `loc = Hotspots[loc_idx]`
2.  `qty = clamp(quantity, 1, Q_vehicle_max)`
3.  `deadline = t_now + clamp(deadline_delay, L_min, L_max)`
4.  **最终输出:** `List[NewDemand]` 对象（通常 $k=1$ 或根据速率约束）。

-----

#### 4\. 核心组件 2：Agent $\mathcal{P}$ (在线路径规划器)

**这是您的博弈对手（由您队友实现）。**

  * **角色：** 求解器。
  * **目标函数：** $J(\mathcal{P}) = \min_{\pi_{\mathcal{P}}} \mathbb{E} [\sum_{t} \Delta \text{Regret}_t]$
  * **模型架构 (用户定义):**
    1.  **动态编码器 (Dynamic Encoder):** (GRU / Transformer) 负责编码时序特征，包括当前已知需求和“预测的未来需求”。
    2.  **滚动解码器 (Rolling Decoder):** 在每个决策点 $t$（例如新需求到达时），重新规划未来 $H$ 步的路线。
  * **动作空间 (Action, $A_{\mathcal{P}}$):**
      * `List[RoutePlan]`：一个列表，包含每辆车 $v_i$ 的未来 $H$ 步访问序列。

-----

#### 5\. 核心组件 3：环境与“遗憾”评估器

  * **`DynamicCVRPEnv` (环境):**
      * `step(action_P, action_G)`:
        1.  接收 $A_{\mathcal{P}}$ (新路线计划) 和 $A_{\mathcal{G}}$ (新需求)。
        2.  将 $A_{\mathcal{G}}$ 的新需求加入 `pending_demands`。
        3.  将 $A_{\mathcal{P}}$ 的新路线应用到车辆上。
        4.  推进时间 $t \rightarrow t+1$。
        5.  更新所有车辆位置，处理服务和卸货。
        6.  检查是否有需求 $d_i$ 超过 $t_{\text{deadline}}$（标记为 `Failed`）。
        7.  调用 `RegretEvaluator` 计算 $\Delta \text{Regret}_t$。
        8.  返回 `(next_state, reward_P, reward_G, done)`。
  * **`RegretEvaluator` (Critic):**
      * **功能：** 计算 $\Delta \text{Regret}_t$，作为双方的共同奖励信号。
      * **$R_{\mathcal{P}} = - \Delta \text{Regret}_t$**
      * **$R_{\mathcal{G}} = + \Delta \text{Regret}_t$**
      * **遗憾定义:** `Regret(t) = ActualCost(t) - IdealOptimalCost(t)`
          * `ActualCost(t)`: 截至 $t$ 时刻，所有已完成服务的总成本 + 对未来待处理需求的 *当前策略* 预估成本。
          * `IdealOptimalCost(t)`: **(需要一个Oracle)** 假设时间停止在 $t$，使用一个强大的离线静态求解器（如 LKH3 或 Gurobi）求解当前所有已知需求（已完成+待处理）的“理论最优总成本”。
      * **$\Delta \text{Regret}_t = \text{Regret}(t) - \text{Regret}(t-1)$**
      * **安全阈：** 如 `RegretEvaluator` 检测到 $\Delta \text{Regret}_t$ 超过上限阈值，应返回一个 `anomaly=True` 标志，用于终止 episode 并进行优先经验回放 (PER)。

-----

#### 6\. 训练范式 (PSRO)

我们使用 **PSRO (策略空间响应预言机)** 范式进行**交替训练 (Alternating Training)**。

**需要维护两个策略池：**

  * `PolicyPool_P = []`
  * `PolicyPool_G = []`

**主训练循环 (Main Training Loop):**

```python
for iteration in range(NUM_ITERATIONS):

    # --- 阶段 1: 训练 Planner ---
    # 目标：训练 P 来应对 G 的“最强”策略组合
    print("--- Training Planner (Agent P) ---")
    
    # 1. 固定 G 的策略（从池中采样）
    # (或使用一个混合策略 Meta-Strategy G)
    fixed_policy_G = PolicyPool_G.sample_latest() 
    
    # 2. 初始化 P (或从 P 池中采样)
    current_policy_P = PlannerAgent()

    # 3. 训练 P (P 在学习)
    for ep in range(TRAINING_EPOCHS_P):
        state = env.reset()
        for t in range(T_MAX):
            # P 决策
            action_P, log_prob_P = current_policy_P.choose_action(state)
            
            # G 使用固定策略决策（不再学习）
            with torch.no_grad():
                action_G = fixed_policy_G.choose_action(state)
            
            # 环境执行并返回遗憾
            next_state, delta_regret, done = env.step(action_P, action_G)
            
            # P 更新（目标：最小化遗憾）
            current_policy_P.update(reward = -delta_regret, log_prob = log_prob_P)
            
            state = next_state
            if done: break
            
    # 4. 将训练好的 P 加入池中
    PolicyPool_P.add(current_policy_P.get_policy_weights())


    # --- 阶段 2: 训练 Generator ---
    # 目标：训练 G 来“击败” P 的“最强”策略组合
    print("--- Training Generator (Agent G) ---")
    
    # 1. 固定 P 的策略（从池中采样）
    fixed_policy_P = PolicyPool_P.sample_latest()
    
    # 2. 初始化 G (或从 G 池中采样)
    current_policy_G = GeneratorAgent() # 您的 Agent

    # 3. 训练 G (G 在学习)
    for ep in range(TRAINING_EPOCHS_G):
        state = env.reset()
        for t in range(T_MAX):
            # P 使用固定策略决策（不再学习）
            with torch.no_grad():
                action_P = fixed_policy_P.choose_action(state)
            
            # G 决策
            action_G, log_prob_G = current_policy_G.choose_action(state, action_P) # G 观察 P 的计划
            
            # 环境执行并返回遗憾
            next_state, delta_regret, done = env.step(action_P, action_G)
            
            # G 更新（目标：最大化遗憾）
            current_policy_G.update(reward = +delta_regret, log_prob = log_prob_G)
            
            state = next_state
            if done: break

    # 4. 将训练好的 G 加入池中
    PolicyPool_G.add(current_policy_G.get_policy_weights())
```

-----

#### 7\. 关键数据结构 (Python 伪代码)

```python
from dataclasses import dataclass
from typing import List, Dict

# --- 实体 ---
@dataclass
class VehicleState:
    id: int
    location: tuple # (x, y)
    remaining_capacity: float
    current_plan: List[int] # 接下来 H 步要访问的需求 ID 列表

@dataclass
class DemandState:
    id: int
    location: tuple # (x, y)
    quantity: float
    arrival_time: int
    deadline: int
    status: str # 'pending', 'serviced', 'failed'

@dataclass
class Hotspot:
    id: int
    location: tuple # GMM 中心 (x, y)

# --- 状态与动作 ---
@dataclass
class GlobalState:
    current_time: int
    vehicles: List[VehicleState]
    pending_demands: List[DemandState]

@dataclass
class PlannerAction:
    # 键是 vehicle_id，值是该车的 H-step 路线
    vehicle_routes: Dict[int, List[int]] 

@dataclass
class GeneratorAction:
    # 生成的新需求列表（通常只有1个）
    new_demands: List[DemandState] 
```

-----

#### 8\. VSCode Agent 协助任务清单

**您的核心模块：`Generator`**

1.  **[模型] `AdversarialTransformer.py`:**
      * 实现一个继承 `torch.nn.Module` 的类。
      * 实现 `__init__`：定义 `vehicle_embed`, `demand_embed`, `hotspot_embed`, `type_embed`, `transformer_encoder`, `cls_token`, 以及 `location_head`, `quantity_head`, `deadline_head`。
      * 实现 `forward(state, hotspots)`：
          * Tokenization：将 `state` 字典（`GlobalState`）转换为嵌入序列。
          * Encoding：`self.transformer_encoder(...)`。
          * Decoding：从 `encoded_tokens` 中分离出 `cls_output` 和 `hotspot_outputs`。
          * 返回三个动作分布 (`torch.distributions`)。
2.  **[智能体] `GeneratorAgent.py`:**
      * 实现一个 `GeneratorAgent` 类。
      * 实现 `__init__`：初始化 `self.model = AdversarialTransformer(...)` 和 `self.optimizer`。
      * 实现 `choose_action(state, hotspots, constraints)`：
          * 调用 `self.model(...)` 获取分布。
          * `dist.sample()` 采样动作。
          * **[关键]** 应用 `constraints`（预算）来裁剪 (clamp) 采样结果。
          * 返回 `GeneratorAction` 对象和 `log_probs`。
      * 实现 `update(reward, log_prob)`：实现一个标准的策略梯度（如 A2C 或 PPO）更新步骤。`loss = -log_prob * reward` (或 PPO 的 clip loss)。

**环境与博弈循环：**

3.  **[环境] `DynamicCVRPEnv.py`:**
      * 实现 `step(action_P, action_G)` 逻辑（时间推进、状态更新）。
      * 实现一个（**简化的**）`calculate_regret()` 函数。
4.  **[主循环] `main_train.py`:**
      * 实现上述 **PSRO** 的交替训练循环。
      * 管理 `PolicyPool_P` 和 `PolicyPool_G`。
      * 协调 `PlannerAgent` 和 `GeneratorAgent` 的训练步骤。