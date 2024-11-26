import torch

def get_random_problems(batch_size,problem_size):

    coords = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    coords1 = coords.unsqueeze(2).expand(batch_size, problem_size, problem_size, 2)
    coords2 = coords.unsqueeze(1).expand(batch_size, problem_size, problem_size, 2)
    dis_matrix = ((coords1 - coords2) ** 2).sum(-1).sqrt()  # 对称且对角线为0了

    return dis_matrix

def get_random_noneuclidian_problems(batch_size, problem_size):
    intmax = 1000
    # Generate integer matrices [0,int_max]
    dis = torch.randint(low=0, high=intmax, size=(batch_size, problem_size, problem_size))
    #  Set Symmetry
    dis_symme = torch.round( (dis + dis.transpose(1,2)) / 2 )
    dis_symme[:, torch.arange(problem_size), torch.arange(problem_size)] = 0
    # Make sure the triangle inequality is satisfied
    while True:
        old_problems = dis_symme.clone()
        dis_symme, _ = (dis_symme[:, :, None, :] + dis_symme[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)
        if (dis_symme == old_problems).all():
            break
    # Normalization
    scaled_dis = dis_symme.float() / intmax
    return scaled_dis

def get_single_test(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    n = int(lines.pop(0))
    d = [[0] * n for i in range(n)]
    for data in lines:
        dataList = data.split()
        i, j = int(dataList[0]), int(dataList[1])
        d[i][j] = float(dataList[2])
        d[j][i] = d[i][j]
    dis = torch.tensor(d).unsqueeze(0)
    return dis