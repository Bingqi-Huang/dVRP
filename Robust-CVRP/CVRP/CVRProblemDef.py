import torch

def get_random_noneuclidian_problems(batch_size, problem_size):
    int_min = 0
    int_max = 10
    # demand
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        demand_scaler = 5

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

    # [int_min,int_max]，shape:(batch,problem+1,problem+1)
    depot_node_matrix = torch.randint(low=int_min, high=int_max, size=(batch_size, problem_size+1, problem_size+1))

    dis_symme = torch.round((depot_node_matrix + depot_node_matrix.transpose(1, 2)) / 2)
    dis_symme[:, torch.arange(problem_size+1), torch.arange(problem_size+1)] = 0

    while True:
        old_problems = dis_symme.clone()
        dis_symme, _ = (dis_symme[:, :, None, :] + dis_symme[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_symme == old_problems).all():
            break

    depot_node_matrix = dis_symme.float() / int_max

    return depot_node_matrix, node_demand



def load_problem_from_file(filename):
    cat_dict = torch.load(filename)
    for k, v in cat_dict.items():  # k 参数名 v 对应参数值
        if k == 'depot_xy':
            depot_xy = v
        elif k == 'node_xy':
            node_xy = v
        else:
            node_demand = v
        print(k,v)
    batch_size = node_xy.size(0)
    problem_size = node_xy.size(1)

    coords = torch.cat((depot_xy, node_xy), dim=1)  # shape:(batch,problem+1,2)
    coords1 = coords.unsqueeze(2).expand(batch_size, problem_size + 1, problem_size + 1, 2)
    coords2 = coords.unsqueeze(1).expand(batch_size, problem_size + 1, problem_size + 1, 2)
    depot_node_matrix = ((coords1 - coords2) ** 2).sum(-1).sqrt()  # 对称且对角线为0了 shape:(batch,problem+1,problem+1)

    return depot_node_matrix, node_demand
