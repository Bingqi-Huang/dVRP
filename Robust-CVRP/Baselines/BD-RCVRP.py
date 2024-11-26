import gurobipy as gp
from gurobipy import GRB
import sys, time, random,os

EPS = 0.0001

def get_wst_scenario(n, sol, d_down, d_up):
    wst = dict()
    for i in range(n+1):
        for j in range(i+1,n+1):
            if sol[(i,j)]>0.5 or sol[(j,i)]>0.5:
                wst[(i, j)] = wst[(j, i)] = d_up[(i,j)]
            else:
                wst[(i, j)] = wst[(j, i)] = d_down[(i,j)]

    return wst


def get_regret(N, Q, q, d_down, d_up, sol):
    d_wst = get_wst_scenario(n=len(N), sol=sol, d_down=d_down, d_up=d_up)
    y_val, y_sol = solve_cvrp_bigM(N=N, E=d_down.keys(), d=d_wst, Q=Q, q=q)
    cost_x = sum(d_up[e] * sol[e] for e in sol)
    regret = sum(d_up[e] * sol[e] for e in sol) - y_val

    return regret, cost_x, y_val, y_sol


#---V：include depot node 0  N：all customer node(i..n)  E：all edges
#   d: cost  q:demand  Q:capacity limitation for the vehicle---#
def solve_cvrp_bigM(N, E, d, Q, q, time_limit=360):
    V = [0] + N
    model = gp.Model("CVRP")
    ### Variable
    # x 0-1
    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    # x = {(i, j): model.addVar(vtype=GRB.BINARY) for i in V for j in V if i != j}
    u = model.addVars(V,vtype=GRB.CONTINUOUS)
    u[0] = 0
    model.update()

    ### Objective
    model.setObjective(gp.quicksum(d[e] * x[e] for e in E), GRB.MINIMIZE)
    # model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i, j in E), GRB.MINIMIZE)

    ### Constraints
    model.addConstrs(gp.quicksum(x[i,j] for j in V if i != j) == 1 for i in N)
    model.addConstrs(gp.quicksum(x[i,j] for i in V if i != j) == 1 for j in N)
    model.addConstrs( ( (u[i] + q[j]) <= (u[j] + Q*(1-x[i,j])) ) for i in V for j in N if i!=j)
    model.addConstrs( ( (u[i] + q[j])
                      >= ( u[j] - (Q - q[i] - q[j])*(1-x[i,j]) ) ) for i in V for j in N if i!=j)


    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0
    # model.Params.lazyConstraints = 1
    if time_limit is not None:
        model.Params.timeLimit = time_limit
    model.optimize()
    if model.SolCount <= 0:
        return None, None
    x_sol = {e: round(x[e].x) for e in x}
    return (model.ObjVal), x_sol


def set_bd_model(N, E, Q, q, d_down, d_up):
    V = [0] + N
    model = gp.Model("BD")
    # x:
    x = {e: model.addVar(vtype=GRB.BINARY, name="x[{}]".format(e)) for e in E}
    # r:  r<=sum(yl+sum y(u-l)x
    r = model.addVar(vtype=GRB.CONTINUOUS, ub=sum(d_up.values())  , name="r")
    u = model.addVars(V, vtype=GRB.CONTINUOUS, name="u")
    u[0] = 0
    model.update()

    model.setObjective(gp.quicksum(d_up[e] * x[e] for e in x) - r, GRB.MINIMIZE)
    # flow cut
    model.addConstrs(gp.quicksum(x[i, j] for j in V if j != i) == 1 for i in N)
    model.addConstrs(gp.quicksum(x[i, j] for i in V if i != j) == 1 for j in N)

    model.addConstrs(((u[i] + q[j]) <= (u[j] + Q * (1 - x[i, j]))) for i in V for j in N if i != j)
    model.addConstrs(((u[i] + q[j])
                      >= (u[j] - (Q - q[i] - q[j]) * (1 - x[i, j]))) for i in V for j in N if i != j)

    model.Params.outputFlag = False
    model.Params.threads = 1
    model.Params.MIPGap = 0.0

    model._x, model._r = x, r
    model._N = N
    model._Q = Q
    model._q = q
    model._d_down, model._d_up = d_down, d_up
    # lazyconstraints callback
    model.Params.lazyConstraints = 1
    model.update()

    return model, x, r



def solve_bd(n, Q, q, d_down, d_up, time_limit):
    N = [i for i in range(1,n+1)]
    model, x, r = set_bd_model(N, d_down.keys(), Q, q, d_down, d_up)
    start_time = time.time()
    bst_val = float('inf')
    while time_limit - time.time() + start_time > 0:
        model.Params.timeLimit = time_limit - time.time() + start_time
        # MP
        # model.optimize(gen_sub_tour_cut)
        model.optimize()
        if model.SolCount <= 0:
            break
        bst_bound = model.objBound
        ttb = time.time() - start_time
        x_sol = {e: 1 if x[e].x > 0.5 else 0 for e in x}
        # Prepare worst-case scenario
        d_wst = get_wst_scenario(n=n, sol=x_sol, d_down=d_down, d_up=d_up)

        # SP
        y_val, y_sol = solve_cvrp_bigM(N=N, E=d_down.keys(), d=d_wst, Q=Q,q=q)
        y_sol = {e for e in d_down if y_sol[e] > 0.5}
        regret = sum(x_sol[e] * d_wst[e] for e in x_sol) - (y_val)
        if bst_val > regret:
            bst_val = regret
            bst_sol = x_sol
            bst_ttb = ttb
        if y_val + EPS > r.x:
            break
        # Add bd cuts
        # bender cutting plane
        model.addConstr(gp.quicksum(d_down[e] + (d_up[e] - d_down[e]) * x[e] for e in y_sol) >= r)

    sol = [e for e in x if bst_sol[e] > 0.5]
    bound = (bst_bound + 1 - EPS)

    x_e = dict()
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                x_e[(i, j)] = 0

    for i in range(n + 1):
        for j in range(n + 1):
            if (i, j) in sol:
                x_e[(i, j)] = 1
    return (bst_val, bound, sol, bst_ttb,x_e)


# RCVRP
def get_robust_rcvrp_instance(filename):
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
    # d = [[0] * (n+1) for i in range(n+1)]
    d_up, d_down = dict(),dict()

    for data in matrix:
        dataList = data.split()
        i, j = int(dataList[0]), int(dataList[1])
        d_up[i,j],d_down[i,j] = float(dataList[2]),float(dataList[3])
        d_up[j,i] = d_up[i,j]
        d_down[j,i] = d_down[i,j]

    q = [0 for _ in range(n+1)]
    q[0] = 0
    idx = 1
    for data in demand:
        q[idx] = float(data)
        idx+=1

    return n, d_up,d_down, q





if __name__ == '__main__':
    n = 50
    int_max = 100
    Q = 1.0
    N = [i for i in range(1, n + 1)]


    for idx in range(1,20+1):
        ins_name = '../Data/R-{}-{}/rcvrp-{}-{}-{}.txt'.format(n,int_max, n, int_max, idx)

        time_limit = float(3600)
        n, d_up, d_down, q = get_robust_rcvrp_instance(ins_name)

        t_start = time.time()
        regret, bound, sol, ttb, x_e = solve_bd(n=n, Q=Q, q=q, d_down=d_down, d_up=d_up, time_limit=time_limit)
        t_end = time.time()

        print(t_end - t_start)
        print('regret:{}'.format(regret))
        print('sol:{}'.format(sol))