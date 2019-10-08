#!/usr/bin/env python -W ignore::DeprecationWarning

# Project 2: Randomized Optimization -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import mlrose
import numpy as np
import os
import pandas as pd
import time

EXPERIMENT_NAME = "TSP_GA"
OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# Traveling Salesman Problem: spiral within a 6x6 grid, best (lowest) fitness score is ~34.6
coords_list = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
               (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
               (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (0, 6),
               (0, 5), (0, 4), (0, 3), (0, 2),
               (1, 2), (2, 2), (3, 2), (4, 2),
               (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]
fitness = mlrose.TravellingSales(coords=coords_list)
problem = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness, maximize=False)
perfect_score = 34.60555127546399

# prep dataset
labels = ['problem', 'max_attempts', 'max_iters', 'pop_size', 'mutation_prob',
          'run_time', 'best_fitness', 'stopped_at', 'func_calls']
results_list = []

# leave these variable static for GA
iterations = np.inf

# run GA over varying options
halt_loop = False
for attempts in (10, 100, 1000):
	for pop_size in (200, 300, 400):
		for mutation_prob in (0.1, 0.2, 0.3):
			start_time = time.perf_counter()
			(_, best_fitness, curve) = mlrose.genetic_alg(problem,
			                                              pop_size=pop_size,
			                                              mutation_prob=mutation_prob,
			                                              max_attempts=attempts,
			                                              max_iters=iterations,
			                                              curve=True,
			                                              random_state=SEED)
			run_time = time.perf_counter() - start_time
			stopped_at = curve.size
			func_calls = problem.get_function_calls()
			problem.reset_function_calls()  # don't forget to reset before the next run
			results_list.append((EXPERIMENT_NAME, attempts, iterations, pop_size, mutation_prob,
			                     run_time, best_fitness, stopped_at, func_calls))
			if best_fitness == perfect_score:
				halt_loop = True
				break
		if halt_loop:
			break
	if halt_loop:
		break

# compile & save results
df_results = pd.DataFrame.from_records(results_list, columns=labels)
df_results.to_excel(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.xlsx'))
df_results.to_pickle(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.pickle'))

# minimal output
print("# Best Run:")
print(df_results.loc[df_results['best_fitness'].idxmin()])
