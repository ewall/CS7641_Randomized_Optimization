#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np
import time

fitness = mlrose.FourPeaks(t_pct=0.1)

# for 4 Peaks of length 50, there are two local maxima of 50, and two optima of 94
problem = mlrose.DiscreteOpt(length=50, fitness_fn=fitness, maximize=True, max_val=2)
init_state = np.array(([0, 1] * 25))

max_iters = 10000
seed = 1

# RHC
rhc_restarts = 1000
rhc_max_attempts = 30
start_time = time.perf_counter()
rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(problem,
                                                                       max_attempts=rhc_max_attempts,
                                                                       max_iters=max_iters,
                                                                       restarts=rhc_restarts,
                                                                       init_state=init_state,
                                                                       curve=True,
                                                                       random_state=seed)
rhc_time = time.perf_counter() - start_time
print("RHC best fitness: {0:.0f} in {1:.4f} seconds and {2} iterations".format(rhc_best_fitness, rhc_time, len(rhc_curve)))
# print('RHC best state:\n', rhc_best_state)

# SA
sa_schedule = mlrose.ArithDecay()
sa_max_attempts = 30
start_time = time.perf_counter()
sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(problem,
                                                                      schedule=sa_schedule,
                                                                      max_attempts=sa_max_attempts,
                                                                      max_iters=max_iters,
                                                                      init_state=init_state,
                                                                      curve=True,
                                                                      random_state=seed)
sa_time = time.perf_counter() - start_time
print("SA best fitness: {0:.0f} in {1:.4f} seconds and {2} iterations".format(sa_best_fitness, sa_time, len(sa_curve)))
# print('SA best state:\n', sa_best_state)

# GA
ga_pop_size = 1000
ga_mutation_prob = 0.1
ga_max_attempts = 30
start_time = time.perf_counter()
ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(problem,
                                                              pop_size=ga_pop_size,
                                                              mutation_prob=ga_mutation_prob,
                                                              max_attempts=ga_max_attempts,
                                                              max_iters=max_iters,
                                                              curve=True,
                                                              random_state=seed)
ga_time = time.perf_counter() - start_time
print("GA fitness {0:.0f} in {1:.4f} seconds and {2} iterations".format(ga_best_fitness, ga_time, len(ga_curve)))
# print('GA best state:\n', ga_best_state)

# MIMIC
mimic_pop_size = 1000
mimic_keep_pct = 0.4
mimic_max_attempts = 40
start_time = time.perf_counter()
mimic_best_state, mimic_best_fitness, mimic_curve = mlrose.mimic(problem,
                                                                 pop_size=mimic_pop_size,
                                                                 keep_pct=mimic_keep_pct,
                                                                 max_attempts=mimic_max_attempts,
                                                                 max_iters=max_iters,
                                                                 curve=True,
                                                                 random_state=seed)
mimic_time = time.perf_counter() - start_time
print("MIMIC fitness {0:.0f} in {1:.4f} seconds and {2} iterations".format(mimic_best_fitness, mimic_time, len(mimic_curve)))
# print('MIMIC best state:\n', mimic_best_state)
