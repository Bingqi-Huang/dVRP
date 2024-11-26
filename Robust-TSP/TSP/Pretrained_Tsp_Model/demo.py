import torch
import numpy as np
from test_import import import_test_main

def get_random_problems(batch_size,problem_size):

    coords = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)

    coords1 = coords.unsqueeze(2).expand(batch_size, problem_size, problem_size, 2)
    coords2 = coords.unsqueeze(1).expand(batch_size, problem_size, problem_size, 2)
    dis_matrix = ((coords1 - coords2) ** 2).sum(-1).sqrt()

    return dis_matrix

dis_matrix = get_random_problems(batch_size=1000,problem_size=20)
no_aug_score,aug_score = import_test_main(dis_matrix,0)
print(no_aug_score,aug_score)