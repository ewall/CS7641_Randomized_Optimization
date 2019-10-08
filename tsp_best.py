#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np
import time


SEED = 1

# Traveling Salesman Problem: spiral within a 6x6 grid, best (lowest) fitness score is ~34.6
coords_list = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
               (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
               (5, 6), (4, 6), (3, 6), (2, 6),  (1, 6), (0, 6),
               (0, 5), (0, 4), (0, 3), (0, 2),
               (1, 2), (2, 2), (3, 2), (4, 2),
               (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]
fitness = mlrose.TravellingSales(coords=coords_list)
problem = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness, maximize=False)

# Print best possible score
# problem.set_state([x for x in range(32)])
# print(problem.get_fitness())

# RHC
rhc_max_attempts = 150
rhc_max_iters = 10000
rhc_restarts = 125
start_time = time.perf_counter()
rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(problem,
                                                                       max_attempts=rhc_max_attempts,
                                                                       max_iters=rhc_max_iters,
                                                                       restarts=rhc_restarts,
                                                                       curve=True,
                                                                       random_state=SEED)
rhc_time = time.perf_counter() - start_time
rhc_func_calls = problem.get_function_calls()
problem.reset_function_calls()  # don't forget to reset before the next run
print("RHC best fitness: {0:.3f} in {1:.3f} seconds, {2} iterations, and {3} function calls.".format(rhc_best_fitness,
                                                                                                     rhc_time,
                                                                                                     len(rhc_curve),
                                                                                                     rhc_func_calls))
print("RHC best state:", rhc_best_state)

# SA
sa_max_attempts = 1000
sa_max_iters = 20000
sa_schedule = mlrose.ArithDecay(init_temp=1)
start_time = time.perf_counter()
sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(problem,
                                                                      schedule=sa_schedule,
                                                                      max_attempts=sa_max_attempts,
                                                                      max_iters=sa_max_iters,
                                                                      curve=True,
                                                                      random_state=SEED)
sa_time = time.perf_counter() - start_time
sa_func_calls = problem.get_function_calls()
problem.reset_function_calls()  # don't forget to reset before the next run
print("SA best fitness: {0:.3f} in {1:.4f} seconds, {2} iterations, and {3} function calls.".format(sa_best_fitness,
                                                                                                    sa_time,
                                                                                                    len(sa_curve),
                                                                                                    sa_func_calls))
print("SA best state:", sa_best_state)

# GA
ga_max_attempts = 1
ga_max_iters = 2
ga_pop_size = 42
ga_mutation_prob = 0.3
start_time = time.perf_counter()
ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(problem,
                                                              pop_size=ga_pop_size,
                                                              mutation_prob=ga_mutation_prob,
                                                              max_attempts=ga_max_attempts,
                                                              max_iters=ga_max_iters,
                                                              curve=True,
                                                              random_state=SEED)
ga_time = time.perf_counter() - start_time
ga_func_calls = problem.get_function_calls()
problem.reset_function_calls()  # don't forget to reset before the next run
print("GA fitness {0:.3f} in {1:.4f} seconds, {2} iterations, and {3} function calls.".format(ga_best_fitness,
                                                                                              ga_time,
                                                                                              len(ga_curve),
                                                                                              ga_func_calls))
print("GA best state:", ga_best_state)

# MIMIC
mimic_max_attempts = 50
mimic_max_iters = np.inf
mimic_pop_size = 400
mimic_keep_pct = 0.5
start_time = time.perf_counter()
mimic_best_state, mimic_best_fitness, mimic_curve = mlrose.mimic(problem,
                                                                 pop_size=mimic_pop_size,
                                                                 keep_pct=mimic_keep_pct,
                                                                 max_attempts=mimic_max_attempts,
                                                                 max_iters=mimic_max_iters,
                                                                 curve=True,
                                                                 random_state=SEED,
                                                                 fast_mimic=False) # bug in fast_mimic, so use false!
mimic_time = time.perf_counter() - start_time
mimic_func_calls = problem.get_function_calls()
problem.reset_function_calls()  # don't forget to reset before the next run
print("MIMIC fitness {0:.3f} in {1:.4f} seconds, {2} iterations, and {3} function calls.".format(mimic_best_fitness,
                                                                                                 mimic_time,
                                                                                                 len(mimic_curve),
                                                                                                 mimic_func_calls))
print("MIMIC best state:", mimic_best_state)
