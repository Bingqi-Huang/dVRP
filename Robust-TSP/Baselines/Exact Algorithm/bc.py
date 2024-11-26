import sys
import time
import os
import gurobipy as gp
from gurobipy import GRB

EPS = 0.0001

def get_wst_scenario(sol, d_mns, d_pls):
    return {e: d_pls[e] if sol[e] > 0.5 else d_mns[e] for e in d_mns}

def get_min_tour(x_sol):
    min_tour = None
    sol = [e for e in x_sol if x_sol[e] > 0.5]
    while sol:
        i, j = sol[0]
        sol = sol[1:]
        start_node, tour = i, [i]
        while start_node != j:
            for ii, jj in sol:
                if ii != j and jj != j:
                    continue
                (i, j) = (ii, jj) if ii == j else (jj, ii)
                tour.append(i)
                sol.remove((ii, jj,))
        if min_tour is None or len(min_tour) > len(tour):
            min_tour = tour
    return min_tour


def gen_sub_tour_cut(mod, where):
    if where != GRB.Callback.MIPSOL:
        return
    x_sol = mod.cbGetSolution(mod._x)
    min_tour = get_min_tour(x_sol=x_sol)
    if len(min_tour) < len(mod._V):
        mod.cbLazy(
            gp.quicksum(mod._x[i, j] for i in min_tour for j in min_tour if (i, j) in mod._x) <= len(min_tour) - 1)
        return
    return


def gen_cut(mod, where):
    """Callback to add a cut for branch-and-cut framework for benders mod"""
    # Execute the function when an incumbent is found
    if where != GRB.Callback.MIPSOL:
        return
    # First check sub tour cut
    x_sol = mod.cbGetSolution(mod._x)
    min_tour = get_min_tour(x_sol=x_sol)
    if len(min_tour) < len(mod._V):
        mod.cbLazy(
            gp.quicksum(mod._x[i, j] for i in min_tour for j in min_tour if (i, j) in mod._x) <= len(min_tour) - 1)
        return

    # Then check Benders cut
    ttb = mod.cbGet(GRB.Callback.RUNTIME)

    # Obtain the incumbent solution
    r_sol = mod.cbGetSolution(mod._r)
    d_pls, d_mns = mod._d_pls, mod._d_mns

    # Prepare worst-case scenario
    d_wst = get_wst_scenario(sol=x_sol, d_mns=d_mns, d_pls=d_pls)

    y_val, y_sol = solve_tsp(V=mod._V, E=d_mns.keys(), d=d_wst)
    y_sol = {e for e in d_mns if y_sol[e] > 0.5}
    if y_val + EPS < r_sol:
        # Add bd cuts
        mod.cbLazy(gp.quicksum(d_mns[e] + (d_pls[e] - d_mns[e]) * mod._x[e] for e in y_sol) >= mod._r)
        return
    mod._ttb = ttb
    return


def get_regret(V, d_mns, d_pls, sol):
    d_wst = get_wst_scenario(sol=sol, d_mns=d_mns, d_pls=d_pls)
    y_val, y_sol = solve_tsp(V=V, E=d_mns.keys(), d=d_wst)
    regret = sum(d_pls[e] * sol[e] for e in sol) - y_val
    return regret, y_sol


def solve_tsp(V, E, d, time_limit=None):
    model = gp.Model("TSP")
    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    model.update()

    model.setObjective(gp.quicksum(d[e] * x[e] for e in E), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in V if (i, j) in E) + gp.quicksum(x[j, i] for j in V if (j, i) in E) == 2 for i in
         V))

    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0
    if time_limit is not None:
        model.Params.timeLimit = time_limit
    model._x = x
    model._V = V
    model.Params.lazyConstraints = 1
    model.optimize(gen_sub_tour_cut)
    if model.SolCount <= 0:
        return None, None
    x_sol = {e: round(x[e].x) for e in x}
    return (model.ObjVal), x_sol


def set_bd_model(V, E, d_mns, d_pls):
    model = gp.Model("BD")

    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    r = model.addVar(vtype=GRB.CONTINUOUS, ub=sum(d_pls.values()), name="r")
    model.update()

    model.setObjective(gp.quicksum(d_pls[e] * x[e] for e in x) - r, GRB.MINIMIZE)
    # flow cut
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in V if (i, j) in E) + gp.quicksum(x[j, i] for j in V if (j, i) in E) == 2 for i in
         V))

    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0

    model._x, model._r = x, r
    model._V = V
    model._d_mns, model._d_pls = d_mns, d_pls
    model.Params.lazyConstraints = 1
    model.update()

    return model, x, r


def solve_bc(n, d_mns, d_pls, time_limit):
    model, x, _ = set_bd_model(range(n), d_mns.keys(), d_mns, d_pls)
    model.Params.timeLimit = time_limit
    model.optimize(gen_cut)
    if model.SolCount <= 0:
        return None, None, None
    sol = [e for e in x if x[e].x > 0.5]
    obj = (model.objVal)
    bound = (model.objBound + 1 - EPS)
    return (obj, bound, sol, model._ttb)


def get_instance(fileName):
    with open(fileName, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    n = int(lines.pop(0))
    d_pls, d_mns = dict(), dict()
    for data in lines:
        dataList = data.split()
        i, j = int(dataList[0]), int(dataList[1])
        d_pls[i, j], d_mns[i, j] = float(dataList[2]), float(dataList[3])
    return n, d_pls, d_mns


def sol_to_route(sol):
    path = []
    cur = sol[0][0]
    while True:
        flag = False
        for i, edge in enumerate(sol):
            if edge[0] == cur and edge[1] not in path:
                path.append(edge[1])
                cur = edge[1]
                flag = True
                break
            elif edge[1] == cur and edge[0] not in path:
                path.append(edge[0])
                cur = edge[0]
                flag = True
                break
        if not flag:
            break
    return path


if __name__ == '__main__':
    n = 20
    int_max = 100
    for idx in range(1, 20 + 1):
        ins_name = '../../Data/Gamma_0.5-{}-{}/Gamma_0.5-{}-{}-{}.txt'.format(n, int_max, n, int_max, idx)
        time_limit = float(3600)
        n, d_pls, d_mns = get_instance(ins_name)

        t_start = time.time()
        regret, bound, sol, ttb = solve_bc(n=n, d_mns=d_mns, d_pls=d_pls, time_limit=time_limit)
        # sol: start from index 0
        sol_list = sol_to_route(sol)
        sol_list = [x + 1 for x in sol_list]  # start from index 1
        print(sol_list)
        t_end = time.time()
        print(regret)
        print(t_end - t_start)



