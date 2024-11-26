import torch
from math import log

# Change the tours route storage from array to linked list to get the next point of each point
def get_link_seq(solutions):
    # solutions.shape = [batch_size,pomo_size,samples_size,problem_ size]
    batch_size, pomo_size, sample_size,problem_size = solutions.size()
    solutions = solutions.view(batch_size, pomo_size,-1, problem_size)
    rec = torch.zeros((batch_size, pomo_size, sample_size,problem_size), device=solutions.device).long()
    index = torch.zeros((batch_size, pomo_size, sample_size,1), device=solutions.device).long()

    for i in range(problem_size - 1):
        rec.scatter_(3, solutions.gather(3, index + i), solutions.gather(3, index + i + 1))
    rec.scatter_(3, solutions[:, :, :, -1].view(batch_size,pomo_size, -1, 1), solutions.gather(3, index))
    return rec

def reward_func(sol, dist_mat):
    # sol.shape:(batch,pomo,n,samples)
    # dist_mat:(batch,pomo,n,n)
    sol_prime = sol.clone().cuda()
    sol_prime = sol_prime.transpose(2,3)  # (batch,pomo,samples,n)
    sorted_sol,sorted_indices = torch.sort(sol_prime,dim=3)
    link = get_link_seq(sorted_indices)
    # (batch,pomo,samples)
    rews = dist_mat.cuda().gather(-2,link).sum(3)
    return rews

class FastCMA(object):
    def __init__(self, Batch, Pomo, N, samples):
        self.batch_size = Batch
        self.pomo_size = Pomo
        mu = samples // 2
        self.samples = samples
        self.weights = torch.tensor([log(mu + 0.5)]).cuda()  # (log(samples))
        self.weights = self.weights - torch.linspace(
            start=1, end=mu, steps=mu).cuda().log()
        self.weights /= self.weights.sum()

        # Calculate the number of effective evolutionary paths mueff.
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()

        self.weights = self.weights.repeat(self.batch_size,self.pomo_size, 1)  # (batch,pomo,log(samples)
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff)
        self.cmu /= ((N + 2) ** 2 + 2 * self.mueff / 2)

        # variables required for the evolution of the covariance matrix
        self.mean = torch.zeros(self.batch_size,self.pomo_size, N).cuda()  # (batch,pomo,N)
        self.b = torch.stack([torch.stack([torch.eye(N) for i in range(self.pomo_size)]) for j in range(self.batch_size)]).cuda()  # (batch,pomo,N,N)
        self.d = self.b.clone()  # (batch,pomo,N,N)
        bd = self.b * self.d
        self.c = bd * torch.transpose(bd, 2, 3)  # (batch,pomo,N,N)
        self.pc = self.mean.clone()  # (batch,pomo,N)

    def step(self, objective_f, dist_mat, step_size):
        batch_size, pomo_size, N, _ = dist_mat.size()

        # Generate a new set of solutions using the current mean and covariance matrix and calculate the fitness value for each solution
        z = torch.randn(batch_size, pomo_size, N, self.samples).cuda()  # (batch,pomo,N,samples)
        # (batch,pomo,N,samples)
        s = self.mean.view(batch_size,pomo_size, -1, 1) + step_size * self.b.matmul(self.d.matmul(z))

        # (batch,pomo,samples,1)
        batch_results_fitness = objective_f(s, dist_mat).unsqueeze(3)
        # (batch,pomo,samples,n)
        batch_results_parameters = s.transpose(2,3)
        batch_results_z = z.transpose(2,3)

        # Sort all solutions from small to large in terms of fitness, select the first half as the parent generation, and calculate their center of gravity and direction vector
        # (batch,pomo,samples,1)
        sorted_fitness, sorted_indices = torch.sort(batch_results_fitness, dim=2)
        # (batch,pomo,samples,n)
        sorted_parameters = batch_results_parameters.gather(dim=2, index=sorted_indices.expand(batch_size, pomo_size,self.samples, N))
        sorted_z = batch_results_z.gather(dim=2, index=sorted_indices.expand(batch_size,pomo_size, self.samples, N))

        top_k = self.samples // 2
        # (batch,pomo,samples/2,N)
        selected_parameters = sorted_parameters[:, :, :top_k, :]
        selected_z = sorted_z[:, :, :top_k, :]

        # Update mean, covariance matrix and other parameters
        self.mean = (selected_parameters * self.weights.unsqueeze(3)).sum(2)  # (batch,pomo,N)
        self.pc *= (1 - self.cc)  # (batch,pomo,N)
        pc_cov = self.pc.unsqueeze(3) * torch.transpose(self.pc.unsqueeze(3), 2, 3)  # (batch,pomo,N,N)
        pc_cov = pc_cov + self.cc * (2 - self.cc) * self.c  # (batch,pomo,N,N)

        bdz = self.b.matmul(self.d).matmul(torch.transpose(selected_z, 2, 3))  # (batch,pomo,N,samples/2)
        cmu_cov = bdz.matmul(self.weights.diag_embed())  # (batch,pomo,N,samples/2)
        cmu_cov = cmu_cov.matmul(torch.transpose(bdz, 2, 3))  # (batch,pomo,N,N)

        self.c *= (1 - self.c1 - self.cmu)  # (batch,pomo,N,N)
        self.c += (self.c1 * pc_cov) + (self.cmu * cmu_cov)  # (batch,pomo,N,N)
        self.d, self.b = torch.linalg.eigh(self.c, UPLO='U')  # self.d:(batch,pomo,N) self.b:(batch,pomo,N,N)
        self.d = self.d.sqrt().diag_embed()  # (batch,pomo,N)-->(batch,pomo,N,N)
        return sorted_fitness,sorted_parameters


def FastCMA_ES(dist_mat, step_size, sample_size, max_epochs):
    batch, pomo, n, _ = dist_mat.size()
    best_reward = torch.zeros(batch,pomo).cuda()
    best_route = torch.zeros(batch,pomo,n).cuda()

    with torch.no_grad():
        cma_es = FastCMA(Batch=batch,Pomo=pomo, N=n, samples=sample_size)
        flag = 0
        for epoch in range(max_epochs):
            try:
                res_fitness,res_parameters = cma_es.step(objective_f=reward_func, dist_mat=dist_mat, step_size=step_size)
            except Exception as e:
                print(e)
                break

            # If best_reward is 0, that is, it has not been assigned a value, then assign it a value directly
            best_reward = torch.where(best_reward == 0, res_fitness.squeeze(-1)[:, :, 0], best_reward)

            # Determine the index to be updated based on whether the reward is smaller
            res_fitness = res_fitness[:, :, 0, :].reshape(batch * pomo)  # (batch,pomo,1,1) --> (batch*pomo)
            best_reward = best_reward.view(-1)  # (batch,pomo) --> (batch*pomo)
            update_idx = best_reward > res_fitness

            if torch.any(update_idx) == False and epoch != 0:
                flag += 1
            else:
                flag = 0
                best_reward = torch.where(update_idx, res_fitness, best_reward)
                # Update the corresponding route according to the index to be updated
                res_parameters = res_parameters[:, :, 0, :].argsort().reshape(batch * pomo, n)  # (batch,pomo,1,n) --> (batch*pomo,n)
                best_route = best_route.view(batch * pomo, n)  # (batch,pomo,n)-->(batch*pomo,n)
                update_route_idx = update_idx.reshape(batch*pomo,1)
                best_route = torch.where(update_route_idx, res_parameters, best_route)

            # Restore the shape of best_reward and best_route
            best_reward = best_reward.view(batch, -1)
            best_route = best_route.reshape(batch, pomo, n)

            if flag >= 20:
                break
    return best_route, best_reward


