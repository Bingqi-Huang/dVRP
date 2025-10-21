# TODOs of MARL-CVRP

Now for entire problem, my initial though is creating 2 agents for generating new demands and one for planning and filling demands. These two should be 2 models(networks) that were trained adversirially. Generator is trying to maximizing **regret** while planner is minimizing. One agent is trained while the other one is frozen. 

Our generator $\mathcal{G}$ and planner $\mathcal{P}$ were implemented within a PSRO (Policy Space Response Oracles) to solve the Nash Equilibrium.

The dynamic generation is defined as 

## `dynamic_cvrp_env.py`
- `def _calculate_regret`: implement regret calculation function
- `def __init__ `: Should we initialize some points and demands or we cold start? We can add a seed for each startup.

## `generator_agent.py`
- `def choose_action`: The constrain s about quantity, delay, etc. Should think about where to place these contrains.

## `planner_agent.py`
- `def __init__`: 
    - Instantiate our planner model.
    - Add support of instantiating multiple baseline solving models.
- `def choose_action`: Extract output of the selected solving model and get the routes for all vehicles.

## `adversarial_transformer.py`
- 
