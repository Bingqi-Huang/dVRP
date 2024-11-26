import torch

def get_random_noneuclidian_problems(batch_size, problem_size, problem_gen_params):
    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']
    Gamma = problem_gen_params['gamma']

    # Generate integer matrices [int_min,int_max]，
    dis = torch.randint(low=int_min, high=int_max, size=(batch_size, problem_size, problem_size,2))
    #  Set Symmetry
    dis_symme = torch.round((dis + dis.transpose(1,2)) / 2)
    dis_up = dis_symme.max(-1)[0]
    dis_down = dis_symme.min(-1)[0]
    dis_up[:, torch.arange(problem_size), torch.arange(problem_size)] = 0
    dis_down[:, torch.arange(problem_size), torch.arange(problem_size)] = 0

    # Make sure the triangle inequality is satisfied
    while True:
        old_dis_up = dis_up.clone()
        dis_up, _ = (dis_up[:, :, None, :] + dis_up[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_up == old_dis_up).all():
            break
    while True:
        old_dis_down = dis_down.clone()
        dis_down, _ = (dis_down[:, :, None, :] + dis_down[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_down == old_dis_down).all():
            break

    # Make sure that only the Gamma edge in dis_up is bounded, and the rest are equal to dis_down
    num_edges = problem_size * (problem_size - 1) // 2  # total edges
    # Gamma = num_edges  # num_edges // 4 * 3
    for b in range(batch_size):
        edges = torch.tril(torch.ones(problem_size, problem_size), -1).nonzero(as_tuple=True)
        num_possible_edges = edges[0].numel()
        chosen_indices = torch.randperm(num_possible_edges)[:num_edges-Gamma]
        chosen_edges = [(edges[0][i], edges[1][i]) for i in chosen_indices]
        for i, j in chosen_edges:
            dis_up[b, i, j] = dis_down[b, i, j]
            dis_up[b, j, i] = dis_down[b, j, i]
    # Normalization
    scaled_dis_up = dis_up.float() / scaler
    scaled_dis_down = dis_down.float() / scaler
    return scaled_dis_up, scaled_dis_down


def get_single_test_updown(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    n = int(lines.pop(0))
    d_pls = [[0] * n for i in range(n)]
    d_mns = [[0] * n for i in range(n)]
    for data in lines:
        dataList = data.split()
        i, j = int(dataList[0]), int(dataList[1])
        d_pls[i][j], d_mns[i][j] = float(dataList[2]), float(dataList[3])
        d_pls[j][i] = d_pls[i][j]
        d_mns[j][i] = d_mns[i][j]
    dis_up = torch.tensor(d_pls).unsqueeze(0)
    dis_down = torch.tensor(d_mns).unsqueeze(0)
    return dis_up, dis_down