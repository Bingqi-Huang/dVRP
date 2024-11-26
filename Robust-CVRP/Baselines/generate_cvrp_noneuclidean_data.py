import torch

n=problem_size = 50
int_min = 0
int_max = 10

scaler = int_max

def get_rcvrp_noneuclidean_problems():
    # demand
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        demand_scaler = 10

    node_demand = torch.randint(1, 10, size=(problem_size,)) / float(demand_scaler)

    # 2 [int_min,int_max]，shape:(batch,problem+1,problem+1)
    depot_node_matrix = torch.randint(low=int_min, high=int_max, size=(problem_size+1, problem_size+1,2))


    dis_symme = torch.round((depot_node_matrix + depot_node_matrix.transpose(0, 1)) / 2)

    dis_up = dis_symme.max(-1)[0]
    dis_down = dis_symme.min(-1)[0]

    dis_up[ torch.arange(problem_size+1), torch.arange(problem_size+1)] = 0
    dis_down[torch.arange(problem_size + 1), torch.arange(problem_size + 1)] = 0

    #
    while True:
        old_problems = dis_up.clone()
        dis_up, _ = (dis_up[:, None, :] + dis_up[None, :, :].transpose(1, 2)).min(dim=2)
        if (dis_up == old_problems).all():
            break

    while True:
        old_problems = dis_down.clone()
        dis_down, _ = (dis_down[:, None, :] + dis_down[None, :, :].transpose(1, 2)).min(dim=2)
        if (dis_down == old_problems).all():
            break

    # 归一化
    depot_node_matrix_up = dis_up.float() / int_max
    depot_node_matrix_down = dis_down.float() / int_max

    return depot_node_matrix_up.numpy(), depot_node_matrix_down.numpy(), node_demand.numpy()



for idx in range(1,20+1):
    depot_node_matrix_up, depot_node_matrix_down, node_demand = get_rcvrp_noneuclidean_problems()
    fileObject = open('../Data/R-{}-{}/rcvrp-{}-{}-{}.txt'.format(problem_size, int_max, problem_size, int_max,idx),'w')
    fileObject.writelines([str(n) + '\n'])
    for i in range(n+1):
        for j in range(i + 1, n+1):
            fileObject.writelines([str(i) + ' ', str(j) + ' ', str(depot_node_matrix_up[i, j])+' ', str(depot_node_matrix_down[i, j])])
            fileObject.write('\n')
    fileObject.write('node_demand\n')
    for i in range(n):
        fileObject.writelines([str(node_demand[i])])
        fileObject.write('\n')
    fileObject.close()

