# This code was used for the computational experiments of the paper "..." by Julius Hoffmann.
# For explanation of the algorithm we refer to the paper

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
import itertools
import csv


# ---------------------------------------------------------------------------
# 0) Parameters -- cannot be changed during the execution and are preconfigured
# ---------------------------------------------------------------------------
@dataclass(frozen = True)
class SolverParams:
    time_limit: int = 7200            # time_limit for Gurobi solver
    threads: int = 4                  # number of threads per run --cpus-per-task in solve_array.sh needs to be adjusted
    log_to_console_sub: int = 0     # output of Gurobi for submodels in Terminal (off in experiments)
    log_to_console_main: int = 0      # output of Gurobi for main-model in Terminal (off in experiments)
    mem: float = 64                  # GB of memory allowed
    absGap: float = 0.99

# ---------------------------------------------------------------------------
# H1) Follower problem -- Solving the lower-level problem as the optimal response to a given
# leader strategy
# ---------------------------------------------------------------------------
def followerReaction(w: np.ndarray, nr_fac: int, n:int, X_array: np.ndarray, coef: np.ndarray, 
                     BF: np.ndarray, params: SolverParams) -> Tuple[np.ndarray, float, float, float]:       #checked
    I = range(n)

    llProblem = gp.Model()

    omega = llProblem.addVars(n, vtype=gp.GRB.BINARY, name="omega")
    y_ll = llProblem.addVars(n, vtype=gp.GRB.BINARY, name="y_ll")

    #Objective function
    llProblem.setObjective(gp.quicksum(w[i] * omega[i] for i in I) - gp.quicksum(y_ll[j] * coef[j] for j in I), gp.GRB.MAXIMIZE) 
    
    #Setting constraints
    llProblem.addConstr(gp.quicksum(y_ll[j] for j in I) == nr_fac)
    llProblem.addConstrs(omega[i] <= gp.quicksum(y_ll[j] for j in np.flatnonzero(X_array[i, BF[i]]).tolist()) for i in I)

    #Settings for solving the model
    llProblem.setParam('OutputFlag', params.log_to_console_sub)
    llProblem.setParam(gp.GRB.Param.Threads, params.threads)
    llProblem.setParam(gp.GRB.Param.MemLimit, params.mem)

    llProblem.optimize()

    solValue = sum(w[i] * round(omega[i].X) for i in I)
    y_ll_values = np.array([round(y_ll[j].X) for j in I])
    llProblemWork = llProblem.Work
    llProblemRuntime = llProblem.Runtime

    return y_ll_values, solValue, llProblemWork, llProblemRuntime

# ---------------------------------------------------------------------------
# H2) Help function best facility -- Getting the closest opened facility for each customer
# The output is an array with the indice of the closest facility for each customer
# ---------------------------------------------------------------------------
def best_facility_determination(d: np.ndarray, opt_result: np.ndarray) -> np.ndarray:   #checked

    opened_facilities = np.where(opt_result >= 0.99)[0]

    opened_distances = d[:, opened_facilities]
    best_idx = np.argmin(opened_distances, axis=1)

    #for i in I:
    #    distances_of_opened_facilities = np.take(d[i], opened_facilities)
    #    opened_fac_with_min_distance = np.argmin(distances_of_opened_facilities)[0]
    #    best_facility = opened_facilities[opened_fac_with_min_distance]

    return opened_facilities[best_idx]

# ---------------------------------------------------------------------------
# H3) Constraint generation for the upper-level -- From a given follower reaction, we 
# construct the parameters for the constraint which we include in the upper-level.
# In case that there are locations where the follower can open facilities which are preferred by customers over each 
# possible location for the leader, some adjustments must be made by the constraint construction, i.e., if the follower
# opened one of these facilities in a response, the corresponding customers must be added to the constraint regardless
# of any leader decission
# ---------------------------------------------------------------------------
def constraint_parameter_generation(n_points: int, dis_array: np.ndarray, y_ll_values: np.ndarray,
                                    lambda_max_variables: np.ndarray, uniq_vals: list) -> Tuple[np.ndarray, np.ndarray]:            # checked
    I = range(n_points)

    BF = best_facility_determination(dis_array, y_ll_values)

    target = dis_array[np.arange(n_points), BF]                                                         # distances of customers to best fac
    location_points = np.array([np.searchsorted(uniq_vals[i], target[i], side="right") - 1 for i in I])

    #for i in I:
    #    distances = np.unique(dis_array[i])
    #    point = np.where(distances <= dis_array[i, BF[i]])[0][-1]

    relevant_customers = np.flatnonzero(location_points <= lambda_max_variables - 1)

    return location_points, relevant_customers


# ---------------------------------------------------------------------------
# 1) Data load -- loads the problem instance and reads the parameter
# ---------------------------------------------------------------------------
def load_instance(instance: str | Path) -> Tuple[int, int, int, np.ndarray, np.ndarray]:

    with open (instance, newline = "") as file:
        reader = csv.reader(file)
        header = next(reader)
        inst_class, nr_points, inst_nr = int(header[0]), int(header[1]), int(header[2])
        weights = np.array([float(x) for x in next(reader)])

    distances = np.loadtxt(instance, delimiter = ",", dtype = float, skiprows = 3)

    if distances.shape != (nr_points, nr_points):
        raise ValueError(f"Missing distance values in {instance}")     #raise error if distance matrix is not complete
    return inst_class, inst_nr, nr_points, weights, distances


# ---------------------------------------------------------------------------
# 2) Pre calculation of problem parameters -- Calculating the (sorted) preference rankings 
# of the customers, the number of preferred facilities over a given facility for each customer, the 
# epsilon value of the callback function and the starting solution and starting constraints
# ---------------------------------------------------------------------------
def parameter_pre_computations(nr_points: int, nr_leader_facilities: int, nr_follower_facilities: int, distances: np.ndarray, 
                               weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list, list, list, np.ndarray, list, list, np.ndarray, np.ndarray, np.ndarray, list]:
    
    # Locations prefered by a customer over a given leader location
    X = preferedFollowerFacilities(distances)
    
    # Locations prefered by a customer over a given follower location
    Y = preferedLeaderFacilities(distances)
    
    # For each customer, the preference sequence of all locations (best to worst) (2D npArray)
    lamb = orderCalculation(nr_points, distances)
    
    # Get the number of prefered leader locations for a given customer and follower location in ascending order
    # Get the number of prefered follower locations for a given customer and leader location in ascending order
    LF, LL = lengthCalculation(nr_points, X, Y)
    
    lamb_relevant_variables = last_relevant_variable_idx(nr_points, nr_leader_facilities, lamb)
    LL_relevant_variables = last_relevant_variable_idx(nr_points, nr_leader_facilities, LL)
    LF_relevant_variables = last_relevant_variable_idx(nr_points, nr_follower_facilities, LF)

    # COBRA calculation
    pairs, unique_pairs = cobra(nr_points, lamb, lamb_relevant_variables)

    # Calculation of the epsilon value for the biobjective subproblem
    inv_epsilon = invEpsilonCalculation(nr_points, nr_follower_facilities, weights, Y)
    
    # Calculating the coefficients for the multiobjective optimization in the follower problem
    epsilon = (1 / inv_epsilon) * 0.99
    counts = np.count_nonzero(Y, axis=-1)
    coef = epsilon * (weights @ counts)

    # Determining the unique values of the distances
    uniq_vals = uniqueCalculation(nr_points, lamb)

    return X, Y, lamb, LF, LL, coef, pairs, unique_pairs, lamb_relevant_variables, LL_relevant_variables, LF_relevant_variables, uniq_vals

# ---------------------------------------------------------------------------
# 2.1) Determining the follower locations which are prefered by a given customer over a given
# leader location -> result is a np-Array which gives for each customer and follower location
# an array with true/fals values for each leader location, given the condition (right hand side are the 
# leader distances, left hand are the follower distances)
# ---------------------------------------------------------------------------
def preferedFollowerFacilities(d: np.ndarray) -> np.ndarray:
    return d[:, None, :] < d[:, :, None]

# ---------------------------------------------------------------------------
# 2.2) Determining the leader locations which are prefered by a given customer over a given
# follower location
# ---------------------------------------------------------------------------
def preferedLeaderFacilities(d: np.ndarray) -> np.ndarray:
    return d[:, None, :] <= d[:, :, None]

# ---------------------------------------------------------------------------
# 2.3) Determining the preference sequence of the locations for each customer
# ---------------------------------------------------------------------------
def orderCalculation(n: int, d: np.ndarray) -> list:
    I = range(n)
    #sorted_indices = [sorted([j for j in I], key = lambda l: d[i, l], reverse = False) for i in I]
    sorted_indices = np.argsort(d, axis=1, stable=True)
    grouped_sorted_indices = [[(value, tuple(group)) for value, group in itertools.groupby(sorted_indices[i], key=lambda l: d[i, l])] for i in I]
    return grouped_sorted_indices

# ---------------------------------------------------------------------------
# 2.4) Determining the number of prefered leader (follower) locations of each customer for a given 
# follower (leader) location in ascending grouped order.
# ---------------------------------------------------------------------------
def lengthCalculation(n: int, X_array: np.ndarray, Y_array: np.ndarray) -> Tuple[list, list]:
    I = range(n)

    L_Follower_Array = np.array([[np.count_nonzero(Y_array[i, j]) for j in I] for i in I])

    #sorted_indices = [sorted([j for j in I], key=lambda l: L_Follower_Array[i, l]) for i in I]
    sorted_indices = np.argsort(L_Follower_Array, axis=1, stable=True)
    L_Follower_sorted_Array = [[(value, tuple(group)) for value, group in itertools.groupby(sorted_indices[i], key=lambda l: L_Follower_Array[i, l])] for i in I]

    L_Leader_Array = np.array([[np.count_nonzero(X_array[i, j]) for j in I] for i in I])

    #sorted_indices = [sorted([j for j in I], key=lambda l: L_Leader_Array[i, l]) for i in I]
    sorted_indices = np.argsort(L_Leader_Array, axis=1, stable=True)
    L_Leader_sorted_Array = [[(value, tuple(group)) for value, group in itertools.groupby(sorted_indices[i], key=lambda l: L_Leader_Array[i, l])] for i in I]

    return L_Follower_sorted_Array, L_Leader_sorted_Array

# ---------------------------------------------------------------------------
# 2.5) Determining the number of relevant decision variables for the ordered formulated problems
# ---------------------------------------------------------------------------
def last_relevant_variable_idx(n: int, nr_fac: int, Array: list) -> np.ndarray:

    I = range(n)
    idx_list = []
    for i in I:
        idx = nr_fac
        for j in range(1, nr_fac + 1):
            idx -= len(Array[i][-j][1])
            if idx <= 0:
                idx_list.append(len(Array[i]) - j)
                break

    return np.array(idx_list)

# ---------------------------------------------------------------------------
# 2.6) COBRA idea according to Church -- some of the assigning variables of customers to facility preference levels
# can be replaced if customers share part of there preference function
# ---------------------------------------------------------------------------
def cobra(n_value: int, lamb_fct: list, lamb_max_val: np.ndarray) -> Tuple[list, list]:
    I = range(n_value)
    pair_array = [[(i, l) for l in range(lamb_max_val[i])] for i in I]
    unique_pair_array = []
    canonical = {}

    for i in I:
        cum = frozenset()
        for l in range(lamb_max_val[i]):
            cum = cum | frozenset(lamb_fct[i][l][1])
            existing = canonical.get(cum)
            if existing is not None:
                pair_array[i][l] = existing
            else:
                canonical[cum] = (i, l)
                unique_pair_array.append((i, l))

    return pair_array, unique_pair_array

# ---------------------------------------------------------------------------
# 2.7) Determining the epsilon value which is for scaling the second objective in the biobjective
# subproblem
# ---------------------------------------------------------------------------
def invEpsilonCalculation(n: int, r: int, w: np.ndarray, Y_array: np.ndarray) -> int:
    I = range(n)

    epsilon_obj_values = [sum([w[i] * np.count_nonzero(Y_array[i, j]) for i in I]) for j in I]
    epsilon_obj_values.sort(reverse = True)
    inverse_epsilon = sum([epsilon_obj_values[j] for j in range(r)])

    return inverse_epsilon

# ---------------------------------------------------------------------------
# 2.8) Determining the unique values of the distances in sorted order
# ---------------------------------------------------------------------------
def uniqueCalculation(n: int, lamb: list) -> list:
    return [np.array([l[0] for l in lamb[i]]) for i in range(n)]


# ---------------------------------------------------------------------------
# 3) Pre calculation of heuristic solutions and initial constraints -- 
# Calculating the p-median of distances, the p-median of dominating facilities and the 
# alternating heuristic.
# ---------------------------------------------------------------------------
def solution_pre_computations(nr_points: int, nr_leader_facilities: int, nr_follower_facilities: int, distances: np.ndarray, 
                              weights: np.ndarray, X: np.ndarray, Y: np.ndarray, lamb: list, LL: list, 
                              LF: list, lamb_max_var: np.ndarray, LL_max_var: np.ndarray, LF_max_var: np.ndarray, coef: np.ndarray, 
                              uniq_vals: list, para: SolverParams) -> Tuple[list, float, float, tuple]:

    initial_constraint_parameters = []
    # p-median on distances
    UB, constraint_parameters, relevant_customers, y_ll_values, heuristic_wt = p_median(nr_points, nr_leader_facilities, nr_follower_facilities, weights, distances, 
                                         lamb, lamb_max_var, X, lamb_max_var, coef, uniq_vals, para)
    initial_constraint_parameters.append((constraint_parameters, relevant_customers))
    
    
    # p-median on dominated facilities
    solution_value, new_constraint_parameters, new_relevant_customers, new_y_ll_values, wt = p_median(nr_points, nr_leader_facilities, nr_follower_facilities, 
                                                                          weights, distances, LL, LL_max_var, X, lamb_max_var, coef, uniq_vals, para)
    heuristic_values = (UB, solution_value)

    heuristic_wt += wt
    if solution_value < UB:
        UB = solution_value
        y_ll_values = new_y_ll_values
        constraint_parameters = new_constraint_parameters
        relevant_customers = new_relevant_customers
    initial_constraint_parameters.append((new_constraint_parameters, new_relevant_customers))

    # Alternating heuristic
    UB, initial_constraint_parameters, wt = alternating_heuristic(nr_points, nr_leader_facilities, nr_follower_facilities, weights, 
                                                              distances, X, Y, coef, y_ll_values, constraint_parameters, relevant_customers,
                                                              initial_constraint_parameters, lamb_max_var, uniq_vals, UB, para)

    heuristic_wt += wt
    # Adding the first additional follower constraint (parameters)
    further_constraint, further_relevant_customers, wt = further_constraint_generation(nr_points, nr_follower_facilities, weights, distances, LF, LF_max_var, lamb_max_var, uniq_vals, para)
    initial_constraint_parameters.append((further_constraint, further_relevant_customers))
    heuristic_wt += wt

    return initial_constraint_parameters, UB, heuristic_wt, heuristic_values


# ---------------------------------------------------------------------------
# 3.1) p-median -- Solving the p-median problem via Gurobi and returning the first bilevel solution value and constraint 
# of the bilevel problem
# ---------------------------------------------------------------------------
def p_median(n_pmed: int, p_pmed: int, r_pmed: int, w_pmed: np.ndarray, d_pmed: np.ndarray, group_sorted: list, group_max: np.ndarray, 
             X_pmed: np.ndarray, lamb_max_var: np.ndarray, coeffs: np.ndarray, uniq_vals: list, params: SolverParams) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:

    I = range(n_pmed)

    pmed = gp.Model()

    y_ul = pmed.addVars(n_pmed, vtype=gp.GRB.BINARY, name="y")
    pairs = [(i, l) for i in I for l in range(group_max[i])]
    x_ul = pmed.addVars(pairs, vtype=gp.GRB.CONTINUOUS, name="x", lb=0.0, ub=1.0)

    # Objective function
    pmed.setObjective(gp.quicksum(w_pmed[i] * (group_sorted[i][0][0] + gp.quicksum((group_sorted[i][l+1][0] - group_sorted[i][l][0]) * x_ul[i,l] for l in range(group_max[i]))) for i in I), gp.GRB.MINIMIZE)

    # Setting Constraints
    pmed.addConstr(gp.quicksum(y_ul[j] for j in I) == p_pmed)
    pmed.addConstrs(x_ul[i,0] >= 1 - gp.quicksum(y_ul[j] for j in group_sorted[i][0][1]) for i in I)
    pmed.addConstrs(x_ul[i,l+1] >= x_ul[i,l] - gp.quicksum(y_ul[j] for j in group_sorted[i][l + 1][1]) for i in I for l in range(group_max[i] - 1))


    #Settings for solving the model
    pmed.setParam('OutputFlag', params.log_to_console_sub)
    pmed.setParam(gp.GRB.Param.Threads, params.threads)
    pmed.setParam(gp.GRB.Param.MemLimit, params.mem)

    pmed.optimize()

    y_ul_values = np.array([round(y_ul[j].X) for j in I])

    # determining best opened leader facilities for the customers
    BF_pmed = best_facility_determination(d_pmed, y_ul_values)

    # determining the follower response to the p-median solution
    y_ll, sol_value, ll_wt, ll_time = followerReaction(w_pmed, r_pmed, n_pmed, X_pmed, coeffs, BF_pmed, params)

    const_para, relevant_cust = constraint_parameter_generation(n_pmed, d_pmed, y_ll, lamb_max_var, uniq_vals)

    return sol_value, const_para, relevant_cust, y_ll, (ll_wt + pmed.Work)

# ---------------------------------------------------------------------------
# 3.2) Alternating heuristic -- the leader and follower are solving the lower-level problem one after another
# with the response of the other party as input
# ---------------------------------------------------------------------------
def alternating_heuristic(n_alt: int, p_alt: int, r_alt: int, w_alt: np.ndarray, d_alt: np.ndarray, X_alt: np.ndarray, 
                          Y_alt: np.ndarray, coefficients: np.ndarray, Sol: np.ndarray, const: np.ndarray, relC: np.ndarray, constraintSets: list,
                          lamb_max_var: np.ndarray, uniq_vals: list, upper_bound: float, params: SolverParams) -> Tuple[float, list, float]:


    if p_alt == r_alt:
    # !!! In this case, in each iteration, the input is a feasible leader solution and hence, with the response we get a feasible
    # bilevel solution, so we do not have to differ between leader and follower turn.
    # For the case p < r, we know that the leader will 
        stConstraints = set()
        key = tuple(const)

        c = 0
        workTime = 0
        while key not in stConstraints and c < 1000:

            stConstraints.add(key)
            c += 1

            # Determining the solution of the player if he acts as follower
            BF_Prev_Player = best_facility_determination(d_alt, Sol)
            Sol, SolValueOfPlayer, ll_wt, ll_time = followerReaction(w_alt, p_alt, n_alt, X_alt, coefficients, BF_Prev_Player, params)

            # Generating new constraint and check for update of upper bound
            const, relC = constraint_parameter_generation(n_alt, d_alt, Sol, lamb_max_var, uniq_vals)
            key = tuple(const)
            constraintSets.append((const, relC))
            if SolValueOfPlayer < upper_bound:
                upper_bound = SolValueOfPlayer
                c = 0

            workTime += ll_wt

    """else:
    # In this case, we need to recalculate epsilon for the leader. Furthermore, the output of followerReaction is not a valid follower 
    # reaction if the leader plays the role of the follower as he sets a different number of facilities
        stConstraints = set()
        key = tuple(const)

        epsilon_leader = 1 / invEpsilonCalculation(n_alt, p_alt, w_alt, Y_alt) * 0.99
        counts = np.count_nonzero(Y_alt, axis=-1)
        coef_leader = epsilon_leader * (w_alt @ counts)
        FolSol = Sol
        c = 0
        workTime = 0
        while not key in stConstraints and c < 1000:

            stConstraints.add(key)
            c += 1

            # Determining the leader solution
            BF_Follower = best_facility_determination(d_alt, FolSol)
            LeadSol, SolValueOfLeaderAsFollower, ll_wt, ll_time = followerReaction(w_alt, p_alt, n_alt, X_alt, coef_leader, BF_Follower, params)
            workTime += ll_wt

            # Determining the follower reaction to the given leader input
            BF_Leader = best_facility_determination(d_alt, LeadSol)
            FolSol, SolValue, ll_wt, ll_time = followerReaction(w_alt, r_alt, n_alt, X_alt, coefficients, BF_Leader, params)
            workTime += ll_wt

            # Generating new constraint and check for update of upper bound
            const, relC = constraint_parameter_generation(n_alt, d_alt, FolSol, lamb_max_var, uniq_vals)
            key = tuple(const)
            constraintSets.append((const, relC))
            if SolValue < upper_bound:
                upper_bound = SolValue
                c = 0"""

    return upper_bound, constraintSets, workTime

# ---------------------------------------------------------------------------
# 3.3) Constraint generation -- Including a promising constraint to begin with
# ---------------------------------------------------------------------------
def further_constraint_generation(n_con: int, r_con: int, w_con: np.ndarray, d_con: np.ndarray, LF_sorted_con: list, 
                                  LF_max_var_con: np.ndarray, lamb_max_var_con: np.ndarray, uniq_vals: list, params: SolverParams) -> Tuple[np.ndarray, np.ndarray, float]:

    I = range(n_con)

    constGeneration = gp.Model()

    y_con = constGeneration.addVars(n_con, vtype=gp.GRB.BINARY, name="y")
    pairs = [(i, l) for i in I for l in range(LF_max_var_con[i])]
    x_con = constGeneration.addVars(pairs, vtype=gp.GRB.CONTINUOUS, name="x", lb=0.0, ub=1.0)

    constGeneration.setObjective(gp.quicksum(w_con[i] * (LF_sorted_con[i][0][0] + gp.quicksum((LF_sorted_con[i][l+1][0] - LF_sorted_con[i][l][0]) * x_con[i,l] for l in range(LF_max_var_con[i]))) for i in I), gp.GRB.MINIMIZE)

    constGeneration.addConstr(gp.quicksum(y_con[j] for j in I) == r_con)
    constGeneration.addConstrs(x_con[i,0] >= 1 - gp.quicksum(y_con[j] for j in LF_sorted_con[i][0][1]) for i in I)
    constGeneration.addConstrs(x_con[i,l+1] >= x_con[i,l] - gp.quicksum(y_con[j] for j in LF_sorted_con[i][l + 1][1]) for i in I for l in range(LF_max_var_con[i] - 1))

    #Settings for solving the model
    constGeneration.setParam('OutputFlag', params.log_to_console_sub)
    constGeneration.setParam(gp.GRB.Param.Threads, params.threads)
    constGeneration.setParam(gp.GRB.Param.MemLimit, params.mem)
    
    constGeneration.optimize()

    y_values = np.array([round(y_con[j].X) for j in I])

    const_para, relCustomer = constraint_parameter_generation(n_con, d_con, y_values, lamb_max_var_con, uniq_vals)

    return const_para, relCustomer, constGeneration.Work



# ---------------------------------------------------------------------------
# 4) Main model -- Solving the single-level reformulation but with (initially) not all constraints; Solving the model with the callback function
# ---------------------------------------------------------------------------
def main_model(n: int, p: int, r:int, w: np.ndarray, d:np.ndarray, lamb: list, X: np.ndarray, pairs, unique_pairs, lamb_max_var: np.ndarray, ubz: float, 
               init_constraints: list, coeffs: np.ndarray, uniq_vals: list, time_max: float, paramaters: SolverParams) -> gp.Model:

    I = range(n)

    timer_load = time.time()
    model = gp.Model()

    #Solver parameters setting
    model.Params.OutputFlag = paramaters.log_to_console_main
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_max
    model.Params.Threads = paramaters.threads
    model.Params.MemLimit = paramaters.mem
    #model.Params.MIPGapAbs = paramaters.absGap

    model.Params.Presolve = 0
    model.Params.Heuristics = 0

    # Generating decission variables (upper level)
    y_ul = model.addVars(n, vtype=gp.GRB.BINARY, name="y_ul")
    x_ul = model.addVars(unique_pairs, vtype=gp.GRB.CONTINUOUS, name="x_ul",lb=0.0, ub=1.0)
    z = model.addVar(vtype=gp.GRB.CONTINUOUS, name="z")

    #Objective function
    model.setObjective(z, gp.GRB.MINIMIZE)

    #Setting constraints
    model.addConstr(gp.quicksum(y_ul[j] for j in I) == p)
    model.addConstrs(x_ul[pairs[i][0][0], pairs[i][0][1]] >= 1 - gp.quicksum(y_ul[j] for j in lamb[i][0][1]) for i in I)
    model.addConstrs(x_ul[pairs[i][l+1][0], pairs[i][l+1][1]] >= x_ul[pairs[i][l][0], pairs[i][l][1]] - gp.quicksum(y_ul[j] for j in lamb[i][l + 1][1]) for i in I for l in range(lamb_max_var[i] - 1))
    model.addConstr(z <= ubz)
    for con in init_constraints:
        model.addConstr(gp.quicksum(w[i] * x_ul[pairs[i][con[0][i]][0], pairs[i][con[0][i]][1]] for i in con[1]) <= z)

    model._y = y_ul
    model._x = x_ul
    model._z = z
    model._ub = ubz
    model._ll_work = 0
    model._ll_time = 0
    model._sub_time = 0
    model._count = 0
    
    callback = make_callback(n, r, d, w, X, lamb_max_var, pairs, coeffs, uniq_vals, paramaters)

    model._load = time.time() - timer_load

    model.optimize(callback)
    print(model._ub, model.Work)

    return model



# ---------------------------------------------------------------------------
# 5) Callback function -- solving the lower-level problem with the given leader solution and creating follower reaction 
# with corresponding constraint
# ---------------------------------------------------------------------------
def make_callback(nr_points: int, r_value: int, distances: np.ndarray, weights: np.ndarray, X_cal: np.ndarray, 
                  lambda_max_var: np.ndarray, pair_ar: list, coeffients: np.ndarray, uniq_vals: list, params_cal: SolverParams):
    
    def callback(master: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            if master.cbGet(GRB.Callback.MIPSOL_OBJ) < master._ub:
                timer = time.time()
                I = range(nr_points)
                y_upl = master.cbGetSolution(master._y)
                y_ul_values = np.array([round(y_upl[j]) for j in I])
                best_leader_fac = best_facility_determination(distances, y_ul_values)

                y_ll, sol_value, ll_wt, ll_time = followerReaction(weights, r_value, nr_points, X_cal, coeffients, best_leader_fac, params_cal)
                cons_parameter = constraint_parameter_generation(nr_points, distances, y_ll, lambda_max_var, uniq_vals)

                if master._ub > sol_value:
                    master._ub = sol_value
                    master.cbLazy(master._ub >= master._z)

                master.cbLazy(gp.quicksum(weights[i] * master._x[pair_ar[i][cons_parameter[0][i]][0], pair_ar[i][cons_parameter[0][i]][1]] for i in cons_parameter[1]) <= master._z)

                master._sub_time += (time.time() - timer)
                master._ll_time += ll_time
                master._ll_work += ll_wt
                master._count += 1
    return callback


# ---------------------------------------------------------------------------
# 6) Extract results -- determining the relevant values of the run and returning them as a dictionary
# ---------------------------------------------------------------------------
def extract_results(model: gp.Model, inst_class: int, inst_nr: int, n: int, p: int, W: float, pre_calc_para_time: float, pre_calc_heur_time: float, 
                    tot_time: float, wt_heur: float, val_heur: tuple) -> dict[str, Any]:

    if model.Status == gp.GRB.Status.OPTIMAL:
        optimal = 1
    elif model.Status == gp.GRB.Status.TIME_LIMIT:
        optimal = 0
        
    elif model.Status == gp.GRB.Status.LOADED:
        optimal = "not executed"
    else:
        optimal = model.Status
    
    return {
        "problem_catrgory": inst_class,
        "instance_number": inst_nr,
        "n": n,
        "p": p,
        "objective": model._ub,
        "value_pmed_classic": val_heur[0],
        "value_pmed_new": val_heur[1],
        "total_weight": W,
        "status": optimal,
        "bound": model.ObjBound,
        "gap": model.MIPGap if model.SolCount > 0 else "",
        "iterations": model._count,
        "runtime": round(model.Runtime, 4),
        "runtime_ll": model._ll_time,
        "runtime_cb": model._sub_time,
        "runtime_precalc_para": pre_calc_para_time,
        "runtime_precalc_heur": pre_calc_heur_time,
        "model_loading_time": model._load,
        "total_time": tot_time,
        "workTime_heur": wt_heur,
        "workTime_main": model.Work,
        "workTime_ll": model._ll_work,
        "max_memory": model.getAttr('MaxMemUsed'),
        "nr_solutions": model.SolCount,
    }


# ---------------------------------------------------------------------------
# Main function from which the rest is executed
# solve(instance_path, nr of facilities for both players, maximal running time) -> with all measurments
# ---------------------------------------------------------------------------
def solve(instance_path: str, nr_fac, max_running_time: float) -> dict[str, Any]:
    main_timer = time.time()
    
    params_main = SolverParams()

    # Get the necessary data
    id1, id2, nodes, weight_vector, distance_matrix = load_instance(instance_path)
    if isinstance(nr_fac, int):
        lead_fac, foll_fac = nr_fac, nr_fac
    elif isinstance(nr_fac, tuple) and len(nr_fac) == 2:
        lead_fac, foll_fac = nr_fac[0], nr_fac[1]
    else:
        raise SystemExit(f"Falsche Eingabe Anzahl zu errichtender Facilities: {nr_fac}")

    # Determine all necessary parameter values
    timer_prec_para = time.time()
    X_main, Y_main, lamb_main, LF_main, LL_main, coefficients_biobj, pairs_main, unique_pairs_main, lamb_max_var_main, LL_max_var_main, LF_max_var_main, unique_dist_vals = parameter_pre_computations(nodes, lead_fac, foll_fac, distance_matrix, weight_vector)
    parameter_prec_time = time.time() - timer_prec_para

    # Get heuristic solutions and initial constraints
    timer_prec_heur = time.time()
    initial_constr_parameters, z_ub, heur_wt, heur_values = solution_pre_computations(nodes, lead_fac, foll_fac, distance_matrix, weight_vector, X_main, Y_main, lamb_main, LL_main, LF_main, 
                                                                lamb_max_var_main, LL_max_var_main, LF_max_var_main, coefficients_biobj, unique_dist_vals, params_main)
    heur_precalculation_time = time.time() - timer_prec_heur

    model_main = main_model(nodes, lead_fac, foll_fac, weight_vector, distance_matrix, lamb_main, X_main, pairs_main, 
                            unique_pairs_main, lamb_max_var_main, z_ub, initial_constr_parameters, coefficients_biobj, unique_dist_vals, max_running_time, params_main)

    return extract_results(model_main, id1, id2, nodes, lead_fac, np.sum(weight_vector), parameter_prec_time, heur_precalculation_time, time.time() - main_timer, heur_wt, heur_values)

# ---------------------------------------------------------------------------
# Local tests via
#   python methods/meine_methode.py instances/beispiel.csv p
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    params_start = SolverParams()
    if len(sys.argv) > 3:
        max_time = float(sys.argv[3])
    else:
        max_time = params_start.time_limit
    return_dict = solve(sys.argv[1], int(sys.argv[2]), max_time)
