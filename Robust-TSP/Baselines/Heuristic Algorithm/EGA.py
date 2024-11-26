import gurobipy as gp
from gurobipy import GRB
import sys, time, random

EPS = 0.0001

MIN_TOUR_SET = set()
XSOL_SET = set()
YSOL_SET = set()
TTB = float('inf')


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


def check_bd_cut(mod, where):
    """Callback to add a cut for branch-and-cut framework for benders mod"""
    # Execute the function when an incumbent is found
    if where != GRB.Callback.MIPSOL:
        return
    # Obtain the incumbent solution
    x_sol = dict()
    for e, v in mod._x.items():
        if isinstance(v, int):
            x_sol[e] = 0
        else:
            x_sol[e] = 1 if mod.cbGetSolution(mod._x[e]) > 0.5 else 0
    # Add sub tour cut
    min_tour = get_min_tour(x_sol)
    if len(min_tour) < len(mod._V):
        mod.cbLazy(
            gp.quicksum(mod._x[i, j] for i in min_tour for j in min_tour if (i, j) in mod._x) <= len(min_tour) - 1)
        # print("sub cut added")
        return

    r_sol = mod.cbGetSolution(mod._r)
    d_pls, d_mns = mod._d_pls, mod._d_mns
    # Obtain the Model object
    ttb = mod.cbGet(GRB.Callback.RUNTIME)

    # Prepare worst-case scenario
    d_wst = {e: d_pls[e] if x_sol[e] > 0.5 else d_mns[e] for e in d_mns}

    # First find cut in pool
    min_sol, min_val = None, float('inf')
    global XSOL_SET, YSOL_SET, TTB
    for sol in XSOL_SET | YSOL_SET:
        val = sum(d_wst[e] for e in sol)
        if val + EPS < min_val:
            min_val, min_sol = val, sol
    if min_val + EPS < r_sol:
        mod.cbLazy(gp.quicksum(d_mns[e] + (d_pls[e] - d_mns[e]) * mod._x[e] for e in min_sol) >= mod._r)
        # print("bd cut fast added")
        return

    y_val, y_sol = solve_tsp(V=mod._V, E=d_mns.keys(), d=d_wst)
    min_sol = tuple(sorted(e for e in y_sol if y_sol[e] > 0.5))
    YSOL_SET.add(min_sol)
    # Check terminate condition
    if y_val + EPS < r_sol:
        # Add bd cuts
        mod.cbLazy(gp.quicksum(d_mns[e] + (d_pls[e] - d_mns[e]) * mod._x[e] for e in min_sol) >= mod._r)
        # print("bd cut added")
        return
    TTB = ttb


def sub_tour(mod, where):
    if where != GRB.Callback.MIPSOL:
        return
    x_sol = mod.cbGetSolution(mod._x)
    min_tour = get_min_tour(x_sol)
    global MIN_TOUR_SET
    if len(min_tour) < len(mod._V):
        MIN_TOUR_SET.add(tuple(sorted(min_tour)))
        mod.cbLazy(
            gp.quicksum(mod._x[i, j] for i in min_tour for j in min_tour if (i, j) in mod._x) <= len(min_tour) - 1)


def get_regret(V, d_mns, d_pls, sol):
    cost = {e: d_mns[e] * (1 - sol[e]) + d_pls[e] * sol[e] for e in sol}
    opt_val, y_sol = solve_tsp(V=V, E=d_mns, d=cost)
    regret = sum(d_pls[e] * sol[e] for e in sol) - opt_val
    return regret, y_sol


def solve_tsp(V, E, d, time_limit=None, E_reduce=None):
    model = gp.Model("TSP")
    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    model.update()

    model.setObjective(gp.quicksum(d[e] * x[e] for e in E), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in V if (i, j) in E) + gp.quicksum(x[j, i] for j in V if (j, i) in E) == 2 for i in
         V))
    global MIN_TOUR_SET
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in min_tour for j in min_tour if (i, j) in x) <= len(min_tour) - 1 for min_tour in
         MIN_TOUR_SET))
    if E_reduce is not None:
        model.addConstr(gp.quicksum(x[e] for e in E - E_reduce) >= 2)

    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0
    if time_limit is not None:
        model.Params.timeLimit = time_limit
    model._x = x
    model._V = V
    model.Params.lazyConstraints = 1
    model.optimize(sub_tour)
    if model.SolCount <= 0:
        return None, None
    x_sol = {e: round(x[e].x) for e in x}
    return model.ObjVal, x_sol


def set_bd_model(V, E, d_mns, d_pls):
    model = gp.Model("BD")
    E_all = set(d_mns.keys())

    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    x.update({e: 0 for e in E_all - E})
    r = model.addVar(vtype=GRB.CONTINUOUS, ub=sum(d_pls.values()), name="r")
    model.update()

    model.setObjective(gp.quicksum(d_pls[e] * x[e] for e in x) - r, GRB.MINIMIZE)
    # flow cut
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in V if (i, j) in E) + gp.quicksum(x[j, i] for j in V if (j, i) in E) == 2 for i in
         V))

    # min tour cut
    global MIN_TOUR_SET
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in min_tour for j in min_tour if (i, j) in x) <= len(min_tour) - 1 for min_tour in
         MIN_TOUR_SET))
    # # bd cut
    # global XSOL_SET
    # model.addConstrs(gp.quicksum(d_mns[e] + (d_pls[e] - d_mns[e]) * x[e] for e in sol) >= r for sol in XSOL_SET)
    # global YSOL_SET
    # model.addConstrs(gp.quicksum(d_mns[e] + (d_pls[e] - d_mns[e]) * x[e] for e in sol) >= r for sol in YSOL_SET)

    # # sol cut
    # global XSOL_SET
    # model.addConstrs(gp.quicksum(x[e] for e in sol) <= len(V) - 2 for sol in XSOL_SET)

    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0
    model.update()

    model._x = {e: v for e, v in x.items() if not isinstance(v, int)}
    model._V = V
    model.Params.lazyConstraints = 1

    return model, x, r


def solve_rtsp(n, d_mns, d_pls, time_limit):
    M = 0
    for i in range(n):
        max_d = 0
        for j in range(n):
            if i < j and max_d < d_pls[i, j]:
                max_d = d_pls[i, j]
            elif j < i and max_d < d_pls[j, i]:
                max_d = d_pls[j, i]
        M += max_d

    d_dict_pls = dict()
    for edge in d_mns:
        d_dict_pls[edge] = M * d_pls[edge] + d_mns[edge]

    start_time = time.time()
    best_obj = float('inf')
    time_remain = time_limit
    E_reduce = set()

    global XSOL_SET, YSOL_SET
    _, sol = solve_tsp(V=range(n), E=d_mns.keys(), d=d_dict_pls, time_limit=time_remain, E_reduce=E_reduce)
    if sol is None:
        return None, None, None
    sol_srt = tuple(sorted(e for e in sol if sol[e] > 0.5))
    E_reduce |= set(sol_srt)
    XSOL_SET.add(sol_srt)
    regret, y_sol = get_regret(V=range(n), d_mns=d_mns, d_pls=d_pls, sol=sol)
    sol_srt = tuple(sorted(e for e in y_sol if y_sol[e] > 0.5))
    YSOL_SET.add(sol_srt)
    time_remain = time_limit - time.time() + start_time
    if best_obj - EPS > regret:
        best_sol = sol
        best_obj, best_ttb = regret, time_limit - time_remain

    while time_remain > 0 and len(E_reduce) < n * (n - 1):

        # stage 1
        _, sol = solve_tsp(V=range(n), E=d_mns.keys(), d=d_dict_pls, time_limit=time_remain, E_reduce=E_reduce)
        if sol is None:
            break
        sol_srt = tuple(sorted(e for e in sol if sol[e] > 0.5))
        E_reduce |= set(sol_srt)
        XSOL_SET.add(sol_srt)

        regret, y_sol = get_regret(V=range(n), d_mns=d_mns, d_pls=d_pls, sol=sol)

        sol_srt = tuple(sorted(e for e in y_sol if y_sol[e] > 0.5))
        YSOL_SET.add(sol_srt)
        time_remain = time_limit - time.time() + start_time
        if best_obj - EPS > regret:
            best_sol = sol
            best_obj, best_ttb = regret, time_limit - time_remain

        # stage 2
        time_remain = time_limit - time.time() + start_time
        if time_remain < 0:
            break
        # Initialize the model as a BD model
        model, x, r = set_bd_model(V=set(range(n)), E=E_reduce, d_mns=d_mns, d_pls=d_pls)
        model._x, model._r = x, r
        model._d_pls, model._d_mns = d_pls, d_mns
        model.Params.timeLimit = time_remain
        model.optimize(check_bd_cut)
        if model.solCount >= 1 and model.objVal + EPS < best_obj:
            best_obj = model.objVal
            best_sol = {e: 1 if not isinstance(x[e], int) and x[e].x > 0.5 else 0 for e in x}
            best_ttb = time_limit - time_remain + TTB

    return (best_obj, best_sol, best_ttb)


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

if __name__ == '__main__':
    random.seed(0)
    n = 500
    int_max = 100

    for idx in range(1, 20 + 1):
        ins_name = '../../Data/NR-{}-{}/NR-{}-{}-{}.txt'.format(n, int_max, n, int_max, idx)
        time_limit = float(3600)
        n, d_pls, d_mns = get_instance(ins_name)

        t_start = time.time()
        regret, sol, ttb = solve_rtsp(n=n, d_mns=d_mns, d_pls=d_pls, time_limit=time_limit)
        t_end = time.time()
        print(regret)
        print(t_end - t_start)
