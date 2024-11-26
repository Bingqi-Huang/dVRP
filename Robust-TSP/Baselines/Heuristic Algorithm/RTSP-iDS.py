'''
modeltype: {DM, DG, MM, MG, GM, GG} D→DFJ, M→MTZ, G→GG
cut_type: {h, b} h→hamming distance constraint, b→best-scenario constraint
'''

from math import sqrt, ceil
from gurobipy import *
import itertools
import sys
import time

EPS = 0.0001


def get_mintour(x_sol):
    min_tour = None
    while x_sol:
        i, j = x_sol[0]
        x_sol = x_sol[1:]
        start_node = i
        tour = [i]
        while start_node != j:
            for ii, jj in x_sol:
                if ii == j:
                    i, j = ii, jj
                    tour.append(i)
                    x_sol.remove((i, j))
                    break
        if min_tour is None or len(min_tour) > len(tour):
            min_tour = tour
    return min_tour


def subtour(m, where):
    if where != GRB.Callback.MIPSOL:
        return
    # make a list of edges selected in the solution
    x_sol = m.cbGetSolution(m._x)
    N = {i for i, _ in x_sol}
    n = len(N)
    x_sol = [(i, j) for i, j in x_sol if x_sol[i, j] > 0.5]
    min_tour = get_mintour(x_sol)
    if len(min_tour) < n:
        m.cbLazy(quicksum(m._x[i, j] for i, j in itertools.permutations(min_tour, 2)) <= len(min_tour) - 1)


def solveTSP_Danzig(d, time=None):
    N = {i for i, _ in d}
    model = Model("TSP")
    x = {(i, j): model.addVar(vtype=GRB.BINARY, name="x[{},{}]".format(i, j)) for i, j in d if i != j}
    model.update()

    model.setObjective(quicksum(d[i, j] * x[i, j] for i, j in x), GRB.MINIMIZE)
    model.addConstrs((quicksum(x[i, j] for i in N if i != j) == 1 for j in N))
    model.addConstrs((quicksum(x[i, j] for j in N if i != j) == 1 for i in N))

    # Step 5: set parameters
    model.Params.outputFlag = False
    model.Params.threads = 1
    if time is not None:
        model.Params.timeLimit = time
    model.Params.MIPGap = 0.0
    model._x = x
    model.Params.lazyConstraints = 1

    model.optimize(subtour)
    xsol = {(i, j): x[i, j].x for i, j in x}
    return model.objval, xsol


def get_regret(d_mns, d_pls, sol):
    cost = {(i, j): d_mns[i, j] * (1 - sol[i, j] - sol[j, i]) + d_pls[i, j] * (sol[i, j] + sol[j, i]) for i, j in sol}

    bestVal, ysol = solveTSP_Danzig(cost)

    regret = sum(d_pls[i, j] * sol[i, j] for i, j in sol) - bestVal
    return regret, ysol


def solveMMRTSP_fix(sub_dist, d_mns, d_pls, time=None):
    N = {i for i, _ in sub_dist}
    V = set(N)
    E = {(i, j) for i in V for j in V if i != j}

    model = Model("TSP")
    x = {(i, j): model.addVar(vtype=GRB.BINARY, name="x[{},{}]".format(i, j)) for i, j in E}
    model.update()
    model.setObjective(quicksum(sub_dist[i, j] * x[i, j] for i, j in x), GRB.MINIMIZE)

    model.addConstrs((quicksum(x[i, j] for i in N if i != j) == 1 for j in V))
    model.addConstrs((quicksum(x[i, j] for j in N if i != j) == 1 for i in V))

    model.Params.outputFlag = False
    model.Params.threads = 1
    if time is not None:
        model.Params.timeLimit = time
    model.Params.MIPGap = 0.0
    model._x = x
    model.Params.lazyConstraints = 1

    model.optimize(subtour)
    # If the model is infeasible, an exact solution is obtained.
    if model.status == GRB.INFEASIBLE: return -1, -1
    # If no feasible solution found, terminate the approach
    if model.solCount <= 0: return -1, -1
    # xsol = {(i,j): round(x[i,j].x) for i,j in x}
    xsol = {(i, j): (x[i, j].x) for i, j in x}
    regret, _ = get_regret(d_mns, d_pls, xsol)

    bound = math.ceil(regret / 2)

    return regret, xsol


def setModel_MMRTSP_DS_M(d_m, d_p, left_model):  # set MMR_TSP model use DS method by DFJ model - MTZ model
    V = {i for i, _ in d_m}
    E = {(i, j) for i in V for j in V if i != j}
    n = len(V)

    model = Model("TSP")

    x = {(i, j): model.addVar(vtype=GRB.BINARY, name="x[{},{}]".format(i, j)) for i, j in E}
    alpha = {i: model.addVar(vtype=GRB.CONTINUOUS, name="alpha[{}]".format(i)) for i in V}
    beta = {i: model.addVar(vtype=GRB.CONTINUOUS, name="beta[{}]".format(i)) for i in V}
    gamma = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, name="gamma[{},{}]".format(i, j)) for i, j in E if
             i != 0 and j != 0}
    tau = {i: model.addVar(vtype=GRB.CONTINUOUS, name="tau[{}]".format(i)) for i in V - {0}}

    if left_model == 'G':
        y = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, name="y[{},{}]".format(i, j)) for i, j in E}
    if left_model == 'M':
        u = {i: model.addVar(vtype=GRB.CONTINUOUS, name="u[{}]".format(i)) for i in V - {0}}
        u[0] = 0

    model.update()

    model.setObjective(quicksum(d_p[i, j] * x[i, j] for i, j in x) - quicksum(alpha[i] + beta[i] for i in V)
                       + (n - 1) * quicksum(gamma[i, j] for i, j in gamma)
                       + (n - 1) * quicksum(tau[i] for i in tau), GRB.MINIMIZE)

    model.addConstrs(
        alpha[j] + beta[i] - n * gamma[i, j] <= d_m[i, j] * (1 - x[i, j] - x[j, i]) + d_p[i, j] * (x[i, j] + x[j, i])
        for i, j in gamma)
    model.addConstrs(
        alpha[j] + beta[0] - n * tau[j] <= d_m[0, j] * (1 - x[0, j] - x[j, 0]) + d_p[0, j] * (x[0, j] + x[j, 0]) for j
        in tau)
    model.addConstrs(
        alpha[0] + beta[i] <= d_m[i, 0] * (1 - x[i, 0] - x[0, i]) + d_p[i, 0] * (x[i, 0] + x[0, i]) for i in V - {0})
    model.addConstrs(- quicksum(gamma[ii, j] for ii, j in E if ii == i and j != 0)
                     + quicksum(gamma[j, ii] for j, ii in E if i == ii and j != 0)
                     + tau[i] <= 0 for i in tau)

    model.addConstrs((quicksum(x[i, j] for i in V if i != j) == 1 for j in V))
    model.addConstrs((quicksum(x[i, j] for j in V if i != j) == 1 for i in V))

    if left_model == 'G':
        model.addConstrs((y[i, j] <= (n - 1) * x[i, j] for (i, j) in E))
        model.addConstr(quicksum(y[0, j] for j in V if (0, j) in E) == n - 1)
        model.addConstrs(
            (quicksum(y[i, j] for j in V if (i, j) in E) - quicksum(y[j, i] for j in V if (j, i) in E) == -1 for i in
             V - {0}))

    if left_model == 'M':
        model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1 for (i, j) in E if j != 0))

    model.Params.outputFlag = False
    model.Params.threads = 1
    #     model.Params.timeLimit = 30
    model.Params.MIPGap = 0.0
    model._x = x
    model.Params.lazyConstraints = 1

    return model, x


def setModel_MMRTSP_DS_G(d_m, d_p, left_model):  # set MMR_TSP model use DS method by DFJ model - GG model
    V = {i for i, _ in d_m}
    E = {(i, j) for i in V for j in V if i != j}
    n = len(V)

    model = Model("TSP")

    x = {(i, j): model.addVar(vtype=GRB.BINARY, name="x[{},{}]".format(i, j)) for i, j in E}
    alpha = {i: model.addVar(vtype=GRB.CONTINUOUS, name="alpha[{}]".format(i)) for i in V}
    beta = {i: model.addVar(vtype=GRB.CONTINUOUS, name="beta[{}]".format(i)) for i in V}
    gamma = {i: model.addVar(vtype=GRB.CONTINUOUS, name="gamma[{}]".format(i)) for i in V}
    tau = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, name="tau[{},{}]".format(i, j)) for i, j in E}

    if left_model == 'G':
        y = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, name="y[{},{}]".format(i, j)) for i, j in E}
    if left_model == 'M':
        u = {i: model.addVar(vtype=GRB.CONTINUOUS, name="u[{}]".format(i)) for i in V - {0}}
        u[0] = 0

    model.update()

    model.setObjective(quicksum(d_p[i, j] * x[i, j] for i, j in x) - quicksum(alpha[i] + beta[i] - gamma[i] for i in V)
                       - n * gamma[0], GRB.MINIMIZE)

    model.addConstrs(alpha[j] + beta[i] + (n - 1) * tau[i, j] <= d_m[i, j] * (1 - x[i, j] - x[j, i]) + d_p[i, j] * (
                x[i, j] + x[j, i]) for i, j in x)
    model.addConstrs(gamma[i] - gamma[j] - tau[i, j] <= 0 for i, j in x)
    model.addConstrs(tau[i, j] >= 0 for i, j in x)
    model.addConstrs((quicksum(x[i, j] for i in V if i != j) == 1 for j in V))
    model.addConstrs((quicksum(x[i, j] for j in V if i != j) == 1 for i in V))

    if left_model == 'G':
        model.addConstrs((y[i, j] <= (n - 1) * x[i, j] for (i, j) in E))
        model.addConstr(quicksum(y[0, j] for j in V if (0, j) in E) == n - 1)
        model.addConstrs(
            (quicksum(y[i, j] for j in V if (i, j) in E) - quicksum(y[j, i] for j in V if (j, i) in E) == -1 for i in
             V - {0}))

    if left_model == 'M':
        model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1 for (i, j) in E if j != 0))

    model.Params.outputFlag = False
    model.Params.threads = 1
    #     model.Params.timeLimit = 30
    model.Params.MIPGap = 0.0
    model._x = x
    model.Params.lazyConstraints = 1

    return model, x


def get_instance(fileName):
    with open(fileName, 'r') as f:
        readList = f.readlines()
        n = int(readList.pop(0))
        d_pls = {}
        d_mns = {}
        for data in readList:
            dataList = data.split()
            i = int(dataList[0])
            j = int(dataList[1])
            pls = float(dataList[2])  # 上界
            mns = float(dataList[3])  # 下界

            d_pls[(i, j)] = pls
            d_pls[(j, i)] = pls
            d_mns[(i, j)] = mns
            d_mns[(j, i)] = mns

    return n, d_pls, d_mns


def solveMMRTSP_iDS(d_mns, d_pls, model_type, cut_type, time_limit):
    start_time = time.time()
    time_remain = time_limit

    left_model = model_type[0]
    if left_model != 'D' and left_model != 'M' and left_model != 'G':
        print('error! only can use D*, G*, M*')
        return -1, -1, -1, -1
    right_model = model_type[1]
    if right_model != 'M' and right_model != 'G':
        print('error! only can use *G, *M')
        return -1, -1, -1, -1

    if right_model == 'M':
        model, x = setModel_MMRTSP_DS_M(d_mns, d_pls, left_model)

    elif right_model == 'G':
        model, x = setModel_MMRTSP_DS_G(d_mns, d_pls, left_model)

    else:
        print('error!')
        return -1, -1, -1, -1

    if cut_type == 'h':
        print('hamming cut')
    elif cut_type == 'b':
        print('best case cut')
    else:
        print("error! cut type can be used only 'h' or 'b'")
        return -1, -1, -1, -1

    minRegret = 0
    for iii in d_pls:
        minRegret += d_pls[iii]
    ite_num = 0
    print('minRegret:{}'.format(minRegret))

    while time_remain > 0:
        model.Params.timeLimit = time_remain
        if left_model == 'D':
            model.optimize(subtour)
        else:
            model.optimize()
        # Log the iteration number
        ite_num += 1
        # If the model is infeasible, an exact solution is obtained.
        if model.status == GRB.INFEASIBLE: break
        # If no feasible solution found, terminate the approach
        if model.solCount <= 0: break
        # Compute the regret of the current solution
        curSol = {j: 1 if x[j].x > 0.5 else 0 for j in x}
        curRegret, _ = get_regret(d_mns, d_pls, curSol)

        if cut_type == 'h':
            # Hamming distance
            model.addConstr(quicksum(1 - x[i, j] for i, j in x if curSol[i, j] > 0.5) >= 1 - EPS)
            model.addConstr(quicksum(1 - x[i, j] for i, j in x if curSol[j, i] > 0.5) >= 1 - EPS)
        else:
            # Best scenario
            model.addConstr(quicksum(
                x[i, j] * d_mns[i, j] + curSol[i, j] * (d_pls[i, j] - d_mns[i, j]) * (x[i, j] + x[j, i]) for i, j in x)
                            <= (quicksum(curSol[i, j] * d_pls[i, j] for i, j in x) - 1 + EPS))
            model.addConstr(quicksum(
                x[i, j] * d_mns[i, j] + curSol[j, i] * (d_pls[i, j] - d_mns[i, j]) * (x[i, j] + x[j, i]) for i, j in x)
                            <= (quicksum(curSol[j, i] * d_pls[i, j] for i, j in x) - 1 + EPS))
            # _______________________________________________________________#

        time_remain = time_limit - (time.time() - start_time)
        if minRegret > curRegret + EPS:
            bestSol = curSol
            # minRegret = round(curRegret)
            minRegret = curRegret
            ttb = time_limit - time_remain
            best_method = str(ite_num)
            print(best_method + ':    (' + str(minRegret) + ')')


    return minRegret, bestSol, best_method, ttb


if __name__ == '__main__':
    n = 50
    int_max = 1000

    for idx in range(1, 20 + 1):
        ins_name = '../../Data/NR-{}-{}/NR-{}-{}-{}.txt'.format(n, int_max, n, int_max, idx)
        time_limit = float(3600)

        model_type = 'DM'
        cut_type = 'h'
        n, d_pls, d_mns = get_instance(ins_name)

        t_start = time.time()
        regret, sol, method, ttb = solveMMRTSP_iDS(d_mns, d_pls, model_type, cut_type, time_limit)
        t_end = time.time()
