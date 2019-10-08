#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np
import time


SEED = 1

# Four Peaks Problem: with length of 40, there are two local maxima of 40, and two optima of 75
fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True, max_val=2)

# RHC
rhc_max_attempts = 75
rhc_max_iters = 10000
rhc_restarts = 25
start_time = time.perf_counter()
rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(problem,
                                                                       max_attempts=rhc_max_attempts,
                                                                       max_iters=rhc_max_iters,
                                                                       restarts=rhc_restarts,
                                                                       curve=True,
                                                                       random_state=SEED)
rhc_time = time.perf_counter() - start_time
print("RHC best fitness: {0:.0f} in {1:.4f} seconds and {2} iterations".format(rhc_best_fitness,
                                                                               rhc_time,
                                                                               len(rhc_curve)))

# SA
sa_max_attempts = 200
sa_max_iters = 10000
sa_schedule = mlrose.ArithDecay(init_temp=1)
start_time = time.perf_counter()
sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(problem,
                                                                      schedule=sa_schedule,
                                                                      max_attempts=sa_max_attempts,
                                                                      max_iters=sa_max_iters,
                                                                      curve=True,
                                                                      random_state=SEED)
sa_time = time.perf_counter() - start_time
print("SA best fitness: {0:.0f} in {1:.4f} seconds and {2} iterations".format(sa_best_fitness, sa_time, len(sa_curve)))

# GA
ga_max_attempts = 20000
ga_max_iters = np.inf
ga_pop_size = 200
ga_mutation_prob = 0.1
start_time = time.perf_counter()
ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(problem,
                                                              pop_size=ga_pop_size,
                                                              mutation_prob=ga_mutation_prob,
                                                              max_attempts=ga_max_attempts,
                                                              max_iters=ga_max_iters,
                                                              curve=True,
                                                              random_state=SEED)
ga_time = time.perf_counter() - start_time
print("GA fitness {0:.0f} in {1:.4f} seconds and {2} iterations".format(ga_best_fitness, ga_time, len(ga_curve)))

# MIMIC
mimic_max_attempts = 20000
mimic_max_iters = np.inf
mimic_pop_size = 750
mimic_keep_pct = 0.5
start_time = time.perf_counter()
mimic_best_state, mimic_best_fitness, mimic_curve = mlrose.mimic(problem,
                                                                 pop_size=mimic_pop_size,
                                                                 keep_pct=mimic_keep_pct,
                                                                 max_attempts=mimic_max_attempts,
                                                                 max_iters=mimic_max_iters,
                                                                 curve=True,
                                                                 random_state=SEED,
                                                                 fast_mimic=True)
mimic_time = time.perf_counter() - start_time
print("MIMIC fitness {0:.0f} in {1:.4f} seconds and {2} iterations".format(mimic_best_fitness, mimic_time, len(mimic_curve)))
