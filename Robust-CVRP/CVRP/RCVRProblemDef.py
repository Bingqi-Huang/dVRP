import torch

def get_random_noneuclidian_problems(batch_size, problem_size, problem_gen_params):
    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']
    Gamma = problem_gen_params['gamma']

    # demand
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

    # Generate integer matrices [int_min,int_max]，shape:(batch,problem+1,problem+1)
    depot_node_matrx = torch.randint(low=int_min, high=int_max, size=(batch_size, problem_size+1, problem_size+1, 2))
    # Set Symmetry
    dis_symme = torch.round((depot_node_matrx + depot_node_matrx.transpose(1, 2)) / 2)
    dis_up = dis_symme.max(-1)[0]
    dis_down = dis_symme.min(-1)[0]
    dis_up[:, torch.arange(problem_size+1), torch.arange(problem_size+1)] = 0
    dis_down[:, torch.arange(problem_size+1), torch.arange(problem_size+1)] = 0

    # Make sure the triangle inequality is satisfied
    while True:
        old_dis_up = dis_up.clone()
        dis_up, _ = (dis_up[:, :, None, :] + dis_up[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_up == old_dis_up).all():
            break
    while True:
        old_dis_down = dis_down.clone()
        dis_down, _ = (dis_down[:, :, None, :] + dis_down[:, None, :, :].transpose(2, 3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_down == old_dis_down).all():
            break

    # Make sure that only the Gamma edge in dis_up is bounded, and the rest are equal to dis_down
    num_edges = problem_size * (problem_size - 1) // 2  # total edges
    # Gamma = num_edges  # num_edges // 4 * 3
    for b in range(batch_size):
        edges = torch.tril(torch.ones(problem_size, problem_size), -1).nonzero(as_tuple=True)
        num_possible_edges = edges[0].numel()
        chosen_indices = torch.randperm(num_possible_edges)[:num_edges - Gamma]
        chosen_edges = [(edges[0][i], edges[1][i]) for i in chosen_indices]
        for i, j in chosen_edges:
            dis_up[b, i, j] = dis_down[b, i, j]
            dis_up[b, j, i] = dis_down[b, j, i]

    depot_node_matrx_up = dis_up.float() / scaler
    depot_node_matrx_down = dis_down.float() / scaler

    return depot_node_matrx_up, depot_node_matrx_down,node_demand

def load_problem_from_rawfile(filename):
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


# RCVRP test
def get_single_test_updown(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    n = int(lines.pop(0))

    marker_index = 0
    for i, line in enumerate(lines):
        if line.startswith('node_demand'):
            marker_index = i
            break

    matrix = lines[:marker_index]
    demand = lines[marker_index+1:]

    d_up = [[0] * (n+1) for i in range(n+1)]
    d_down = [[0] * (n+1) for i in range(n+1)]
    # d_up, d_down = dict(),dict()

    for data in matrix:
        dataList = data.split()
        i, j = int(dataList[0]), int(dataList[1])
        d_up[i][j], d_down[i][j] = float(dataList[2]),float(dataList[3])
        d_up[j][i] = d_up[i][j]
        d_down[j][i] = d_down[i][j]

    idx = 0
    q = [0 for _ in range(n)]
    for data in demand:
        q[idx] = float(data)
        idx += 1

    d_up = torch.tensor(d_up).unsqueeze(0)
    d_down = torch.tensor(d_down).unsqueeze(0)
    q = torch.tensor(q)

    return d_up, d_down, q
