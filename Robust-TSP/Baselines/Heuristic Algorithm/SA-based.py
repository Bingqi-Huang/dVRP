import os
import elkai
import math, random, time

def get_instance(fileName):
    with open(fileName, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    n = int(lines.pop(0))
    d_pls, d_mns = dict(), dict()
    for data in lines:
        dataList = data.split()
        i, j = float(dataList[0]), float(dataList[1])
        d_pls[i, j], d_mns[i, j] = float(dataList[2]), float(dataList[3])
    return n, d_pls, d_mns

def nearest_city(current_city, remain_cities, d_mns, d_pls):
    temp_min = float('inf')
    next_city = None
    for i in range(len(remain_cities)):
        if current_city < remain_cities[i]:
            distance = d_pls[current_city, remain_cities[i]]
        else:
            distance = d_pls[remain_cities[i], current_city]
        if distance < temp_min:
            temp_min = distance
            next_city = remain_cities[i]
    return next_city, temp_min

def greedy_initial_route(origin, remain_cities, d_mns, d_pls):
    cand_cities = remain_cities[:]
    current_city = origin
    initial_route = []
    mile_cost = 0
    sol_e = []
    while len(cand_cities) > 0:
        next_city, distance = nearest_city(current_city, cand_cities, d_mns, d_pls)
        mile_cost += distance
        initial_route.append(next_city)
        sol_e.append((current_city, next_city))
        current_city = next_city
        cand_cities.remove(next_city)
    if initial_route[-1] < origin:
        mile_cost += d_pls[initial_route[-1], origin]
    else:
        mile_cost += d_pls[origin, initial_route[-1]]

    initial_route.append(origin)

    return initial_route, mile_cost

def random_swap_2_city(route):
    new_route = route[:]
    rs = random.sample(range(0, n), 2)
    index = sorted(rs)
    L = index[1] - index[0] + 1
    for j in range(L):
        new_route[index[0] + j] = route[index[1] - j]
    return new_route

def ZTour_Cost(route, d_matrix):
    ZCOST = 0
    for i in range(n - 1):
        current_node = route[i]
        next_node = route[i + 1]
        ZCOST += d_matrix[current_node][next_node]
    ZCOST += d_matrix[route[n - 1]][route[0]]
    return ZCOST

def sol_to_wcrmatrix(route):
    sol_e = []
    for i in range(0, n - 1):
        sol_e.append((route[i], route[i + 1]))
    sol_e.append((route[n - 1], route[0]))

    costmatrix_wcr = [[0] * n for i in range(n)]
    wcr = dict()
    for i in range(n):
        for j in range(n):
            if (i, j) in sol_e or (j, i) in sol_e:
                if i < j:
                    costmatrix_wcr[i][j] = d_pls[i, j]
                    costmatrix_wcr[j][i] = d_pls[i, j]
                elif i > j:
                    costmatrix_wcr[i][j] = d_pls[j, i]
                    costmatrix_wcr[j][i] = d_pls[j, i]
            else:
                if i < j:
                    costmatrix_wcr[i][j] = d_mns[i, j]
                    costmatrix_wcr[j][i] = d_mns[i, j]
                elif i > j:
                    costmatrix_wcr[i][j] = d_mns[j, i]
                    costmatrix_wcr[j][i] = d_mns[j, i]

    return costmatrix_wcr


def RCOST(route):
    wcr_matrix = sol_to_wcrmatrix(route)
    x_cost = ZTour_Cost(route, wcr_matrix)
    route_y = elkai.solve_float_matrix(wcr_matrix)
    min_y_cost = ZTour_Cost(route_y, wcr_matrix)
    min_max_regret = x_cost - min_y_cost
    return min_max_regret


def SA_based_heuristic(T_0, Ite_max, Num_max, Time_limit, alpha):
    Temp = T_0
    Ite = 0
    Num = 0

    # Greedy method to construct initial solution
    origin = 0
    remain_cities = []
    for i in range(1, n):
        remain_cities.append(i)
    route_x, x_cost = greedy_initial_route(origin=origin, remain_cities=remain_cities, d_mns=d_mns, d_pls=d_pls)

    min_max_regret = RCOST(route_x)

    best_route, best_value= route_x[:], min_max_regret
    # Record the current solution corresponding to the temperature drop
    record = [best_value]

    time_start = time.time()
    time_cost = 0

    while Num < Num_max and time_cost < Time_limit:
        Ite = Ite + 1
        # print('---------------Iteration:{}----------------'.format(Ite))
        # new candidate solution
        cand_route_x = random_swap_2_city(route_x)
        costmatrix_wcr = sol_to_wcrmatrix(cand_route_x)
        cand_x_cost = ZTour_Cost(cand_route_x, costmatrix_wcr)

        # new optimal path and optimal value under the worst scenario
        route_y = elkai.solve_float_matrix(costmatrix_wcr)
        min_y_cost = ZTour_Cost(route_y, costmatrix_wcr)
        cand_min_max_regret = cand_x_cost - min_y_cost
        deta = cand_min_max_regret - min_max_regret
        if deta <= 0:  # update
            route_x = cand_route_x
            min_max_regret = cand_min_max_regret
            record.append(min_max_regret)

        else:
            rand = random.uniform(0, 1)
            cc = Temp / (Temp ** 2 + deta **2)
            if rand < cc:
                route_x = cand_route_x
                min_max_regret = cand_min_max_regret
                record.append(min_max_regret)

        if min_max_regret < best_value:
            best_route = route_x[:]
            best_value = min_max_regret
            Num = 0

        if Ite == Ite_max:
            Temp = Temp * alpha
            Ite = 0
            Num = Num + 1

        time_end = time.time()
        time_cost = time_end - time_start

    return record, best_value, best_route

if __name__ == '__main__':
    n = 20
    int_max = 100

    # SA_Based_heuristic hyper-parameter
    T0 = 1000  # The larger T0 is, the more likely it is to accept a worse solution and the slower the convergence.
    Itemax = 150 * n
    Nummax = 100  # The larger the value, the better the solution quality. A better solution comes at the cost of more computation time.
    alpha = 0.93
    Timelimit = 3600  # 15 * n

    for idx in range(1, 20 + 1):
        ins_name = '../../Data/Gamma_0.5-{}-{}/Gamma_0.5-{}-{}-{}.txt'.format(n, int_max, n, int_max, idx)
        n, d_pls, d_mns = get_instance(ins_name)

        t_start = time.time()
        record, Best_Rcost, Best_sol = SA_based_heuristic(T0, Itemax, Nummax, Timelimit, alpha)
        t_end = time.time()