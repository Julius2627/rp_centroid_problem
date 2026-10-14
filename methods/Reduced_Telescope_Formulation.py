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
    #absGap: float = 0.99

# ---------------------------------------------------------------------------
# H1) Follower problem -- Solving the lower-level problem as the optimal response to a given
# leader strategy
# ---------------------------------------------------------------------------
def followerReaction(w: np.ndarray, nr_fac: int, n:int, X_array: np.ndarray, BF: np.ndarray, 
                     params: SolverParams) -> Tuple[np.ndarray, float, float, float]:
    I = range(n)

    llProblem = gp.Model()

    omega = llProblem.addVars(n, vtype=gp.GRB.BINARY, name="omega")
    y_ll = llProblem.addVars(n, vtype=gp.GRB.BINARY, name="y_ll")

    #Objective function
    llProblem.setObjective(gp.quicksum(w[i] * omega[i] for i in I), gp.GRB.MAXIMIZE)
    
    #Setting constraints
    llProblem.addConstr(gp.quicksum(y_ll[j] for j in I) == nr_fac)
    llProblem.addConstrs(omega[i] <= gp.quicksum(y_ll[j] for j in np.flatnonzero(X_array[i, BF[i]]).tolist()) for i in I)

    #Settings for solving the model
    llProblem.setParam('OutputFlag', params.log_to_console_sub)
    llProblem.setParam(gp.GRB.Param.Threads, params.threads)
    llProblem.setParam(gp.GRB.Param.MemLimit, params.mem)
    
    llProblem.optimize()

    solValue = llProblem.getAttr(gp.GRB.Attr.ObjVal)
    y_ll_values = np.array([round(y_ll[j].X) for j in I])
    llProblemWork = llProblem.Work
    llProblemRuntime = llProblem.Runtime

    return y_ll_values, solValue, llProblemWork, llProblemRuntime

# ---------------------------------------------------------------------------
# H2) Help function best facility -- Getting the closest opened facility for each customer
# The output is an array with the indice of the closest facility for each customer
# ---------------------------------------------------------------------------
def best_facility_determination(d: np.ndarray, opt_result: np.ndarray) -> np.ndarray:

    opened_facilities = np.where(opt_result >= 0.99)[0]

    opened_distances = d[:, opened_facilities]
    best_idx = np.argmin(opened_distances, axis=1)

    #for i in I:
    #    distances_of_opened_facilities = np.take(d[i], opened_facilities)
    #    opened_fac_with_min_distance = np.argmin(distances_of_opened_facilities)
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
def constraint_parameter_generation(n_points: int, dis_array: np.ndarray, y_ll_values: np.ndarray, uniq_vals: list) -> np.ndarray:
    I = range(n_points)

    BF = best_facility_determination(dis_array, y_ll_values)

    target = dis_array[np.arange(n_points), BF]
    location_points = np.array([np.searchsorted(uniq_vals[i], target[i], side="right") - 1 for i in I])

    #for i in I:
    #    distances = np.unique(dis_array[i])
    #    point = np.where(distances <= dis_array[i, BF[i]])[0][-1]

    return location_points


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
def parameter_pre_computations(nr_points: int, distances: np.ndarray) -> Tuple[np.ndarray, list, list]:

    # Locations prefered by a customer over a given leader location
    X = preferedFollowerFacilities(distances)
    
    # For each customer, the preference sequence of all locations (best to worst) (2D npArray)
    lamb = orderCalculation(nr_points, distances)

    # Determining the unique values of the distances
    uniq_vals = uniqueCalculation(nr_points, lamb)

    return X, lamb, uniq_vals

# ---------------------------------------------------------------------------
# 2.1) Determining the follower locations which are prefered by a given customer over a given
# leader location -> result is a np-Array which gives for each customer and follower location
# an array with true/fals values for each leader location, given the condition (right hand side are the 
# leader distances, left hand are the follower distances)
# ---------------------------------------------------------------------------
def preferedFollowerFacilities(d: np.ndarray) -> np.ndarray:
    return d[:, None, :] < d[:, :, None]

# ---------------------------------------------------------------------------
# 2.2) Determining the preference sequence of the locations for each customer
# ---------------------------------------------------------------------------
def orderCalculation(n: int, d: np.ndarray) -> list:
    I = range(n)
    sorted_indices = np.argsort(d, axis=1, stable=True)
    grouped_sorted_indices = [[(value, tuple(group)) for value, group in itertools.groupby(sorted_indices[i], key=lambda l: d[i, l])] for i in I]
    return grouped_sorted_indices

# ---------------------------------------------------------------------------
# 2.3) Determining the unique values of the distances in sorted order
# ---------------------------------------------------------------------------
def uniqueCalculation(n: int, lamb: list) -> list:
    return [np.array([l[0] for l in lamb[i]]) for i in range(n)]



# ---------------------------------------------------------------------------
# 3) Main model -- Solving the single-level reformulation but with (initially) not all constraints; Solving the model with the callback function
# ---------------------------------------------------------------------------
def main_model(n: int, p: int, r:int, w: np.ndarray, d:np.ndarray, lamb: list, X: np.ndarray, u_v: list, time_max: float,
               paramaters: SolverParams) -> gp.Model:

    I = range(n)
    ubz = np.sum(w)

    timer_load = time.time()
    model = gp.Model()

    #Solver parameters setting
    model.Params.OutputFlag = paramaters.log_to_console_main
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_max
    model.Params.Threads = paramaters.threads
    model.Params.MemLimit = paramaters.mem
    #model.Params.MIPGapAbs = paramaters.absGap


    # Generating decission variables (upper level)
    y_ul = model.addVars(n, vtype=gp.GRB.BINARY, name="y_ul")
    pairs = [(i, l) for i in I for l in range(len(lamb[i]))]
    x_ul = model.addVars(pairs, vtype=gp.GRB.CONTINUOUS, name="x_ul",lb=0.0, ub=1.0)
    z = model.addVar(vtype=gp.GRB.CONTINUOUS, name="z", ub=ubz)

    #Objective function
    model.setObjective(z, gp.GRB.MINIMIZE)

    #Setting constraints
    model.addConstr(gp.quicksum(y_ul[j] for j in I) == p)
    model.addConstrs(x_ul[i, 0] >= 1 - gp.quicksum(y_ul[j] for j in lamb[i][0][1]) for i in I)
    model.addConstrs(x_ul[i, l+1] >= x_ul[i, l] - gp.quicksum(y_ul[j] for j in lamb[i][l + 1][1]) for i in I for l in range(len(lamb[i]) - 1))
        

    model._y = y_ul
    model._x = x_ul
    model._z = z
    model._ub = ubz
    model._ll_work = 0
    model._ll_time = 0
    model._sub_time = 0
    model._count = 0
    
    callback = make_callback(n, r, d, w, X, u_v, paramaters)

    model._load = time.time() - timer_load
    model.optimize(callback)
    #print(model._ub, model.Work)

    return model


# ---------------------------------------------------------------------------
# 4) Callback function -- solving the lower-level problem with the given leader solution and creating follower reaction 
# with corresponding constraint
# ---------------------------------------------------------------------------
def make_callback(nr_points: int, r_value: int, distances: np.ndarray, weights: np.ndarray, X_cal: np.ndarray, u_val: list, params_cal: SolverParams):
    
    def callback(master: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            timer = time.time()
            I = range(nr_points)
            y_ul = master.cbGetSolution(master._y)
            y_ul_values = np.array([round(y_ul[j]) for j in I])
            best_leader_fac = best_facility_determination(distances, y_ul_values)

            y_ll, sol_value, ll_wt, ll_time = followerReaction(weights, r_value, nr_points, X_cal, best_leader_fac, params_cal)
            cons_parameter = constraint_parameter_generation(nr_points, distances, y_ll, u_val)

            if master._ub > sol_value:
                master._ub = sol_value
                master.cbLazy(master._ub >= master._z)

            master.cbLazy(gp.quicksum(weights[i] * master._x[i, cons_parameter[i]] for i in I) <= master._z)

            master._sub_time += (time.time() - timer)
            master._ll_time += ll_time
            master._ll_work += ll_wt
            master._count += 1
    return callback


# ---------------------------------------------------------------------------
# 5) Extract results -- determining the relevant values of the run and returning them as a dictionary
# ---------------------------------------------------------------------------
def extract_results(model: gp.Model, inst_class: int, inst_nr: int, n: int, p: int, W: float, pre_calc_time: float, tot_time: float) -> dict[str, Any]:

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
        "value_pmed_classic": "-",
        "value_pmed_new": "-",
        "total_weight": W,
        "status": optimal,
        "bound": model.ObjBound,
        "gap": model.MIPGap,
        "iterations": model._count,
        "runtime": round(model.Runtime, 4),
        "runtime_ll": model._ll_time,
        "runtime_cb": model._sub_time,
        "runtime_precalc_para": pre_calc_time,
        "runtime_precalc_heur": 0,
        "model_loading_time": model._load,
        "total_time": tot_time,
        "workTime_heur": 0,
        "workTime_main": model.Work,
        "workTime_ll": model._ll_work,
        "max_memory": model.getAttr('MaxMemUsed'),
        "nr_solutions": model.SolCount,
    }



# ---------------------------------------------------------------------------
# Main function from which the rest is executed
# solve(instance_path) -> dict
# ---------------------------------------------------------------------------
def solve(instance_path: str, nr_fac: int, max_running_time: float) -> dict[str, Any]:
    main_timer = time.time()
    params_main = SolverParams()

    # Get the necessary data
    id1, id2, nodes, weight_vector, distance_matrix = load_instance(instance_path)
    lead_fac, foll_fac = nr_fac, nr_fac

    # Determine all necessary parameter values
    timer_prec = time.time()
    X_main, lamb_main, uniq_v = parameter_pre_computations(nodes, distance_matrix)
    precalculation_time = time.time() - timer_prec


    model_main = main_model(nodes, lead_fac, foll_fac, weight_vector, distance_matrix, lamb_main, X_main, uniq_v, max_running_time, params_main)

    return extract_results(model_main, id1, id2, nodes, nr_fac, np.sum(weight_vector), precalculation_time, time.time() - main_timer)


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