#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np
import timeit

fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)

init_state = np.array(([0, 1] * 50))
max_attempts = 10
max_iters = 1000
seed = 1

# RHC
rhc_restarts = 100
rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(problem,
                                                                       max_attempts=max_attempts,
                                                                       max_iters=max_iters,
                                                                       restarts=rhc_restarts,
                                                                       init_state=init_state,
                                                                       curve=True,
                                                                       random_state=seed)
#print('# RHC best state:\n', rhc_best_state)
print('# RHC best fitness: ', rhc_best_fitness)

# SA
sa_schedule = mlrose.ArithDecay()
sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(problem,
                                                                      schedule=sa_schedule,
                                                                      max_attempts=max_attempts,
                                                                      max_iters=max_iters,
                                                                      init_state=init_state,
                                                                      curve=True,
                                                                      random_state=seed)
#print('# SA best state:\n', sa_best_state)
print('# SA best fitness: ', sa_best_fitness)

# GA
ga_pop_size = 200
ga_mutation_prob = 0.1
ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(problem,
                                                              pop_size=ga_pop_size,
                                                              mutation_prob=ga_mutation_prob,
                                                              max_attempts=max_attempts,
                                                              max_iters=max_iters,
                                                              curve=True,
                                                              random_state=seed)
#print('# GA best state:\n', ga_best_state)
print('# GA best fitness: ', ga_best_fitness)

# MIMIC
mimic_pop_size = 200
mimic_keep_pct = 0.2
mimic_best_state, mimic_best_fitness, mimic_curve = mlrose.mimic(problem,
                                                                 pop_size=mimic_pop_size,
                                                                 keep_pct=mimic_keep_pct,
                                                                 max_attempts=max_attempts,
                                                                 max_iters=max_iters,
                                                                 curve=True,
                                                                 random_state=seed)
#print('# MIMC best state:\n', mimic_best_state)
print('# MIMIC best fitness: ', mimic_best_fitness)
