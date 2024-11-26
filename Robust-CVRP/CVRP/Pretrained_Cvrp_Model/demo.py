import torch
import numpy as np
from test_import import import_test_main


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        demand_scaler = 5
        # raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

    # shape: (batch, problem)

    coords = torch.cat( (depot_xy,node_xy), dim=1)  # shape:(batch,problem+1,2)
    coords1 = coords.unsqueeze(2).expand(batch_size, problem_size+1, problem_size+1, 2)
    coords2 = coords.unsqueeze(1).expand(batch_size, problem_size+1, problem_size+1, 2)
    depot_node_matrix = ((coords1 - coords2) ** 2).sum(-1).sqrt()  # 对称且对角线为0了 shape:(batch,problem+1,problem+1)

    return depot_node_matrix, node_demand

depot_node_matrix,node_demand = get_random_problems(batch_size=1000,problem_size=20)
no_aug_score,aug_score = import_test_main(depot_node_matrix,node_demand,0)
print('no_aug_score.shape:{}'.format(no_aug_score.shape))
print('aug_score.shape:{}'.format(aug_score.shape))