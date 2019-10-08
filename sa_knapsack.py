#!/usr/bin/env python -W ignore::DeprecationWarning

# Project 2: Randomized Optimization -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import mlrose
import numpy as np
import os
import pandas as pd
import time

EXPERIMENT_NAME = "Knapsack_SA"
OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# 0-1 Knapsack Problem: can only choose each item once, trying to maximize value before filling up the sack
weights = [10, 5, 2, 8, 15, 11, 4, 7, 1, 20]
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_weight_pct = 0.6
fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=len(values), fitness_fn=fitness, maximize=True, max_val=2)

# prep dataset
labels = ['problem', 'max_attempts', 'max_iters', 'temp', 'run_time', 'best_fitness', 'stopped_at', 'func_calls']
results_list = []

# leave these variable static for SA
iterations = np.inf
decay = mlrose.ArithDecay

# run SA over varying options
for attempts in (10, 20, 30):
	for temp in (1, 10, 100, 1000):
		start_time = time.perf_counter()
		(_, best_fitness, curve) = mlrose.simulated_annealing(problem,
															  schedule=decay(init_temp=temp),
															  max_attempts=attempts,
															  max_iters=iterations,
															  curve=True,
															  random_state=SEED)
		run_time = time.perf_counter() - start_time
		stopped_at = curve.size
		func_calls = problem.get_function_calls()
		problem.reset_function_calls()  # don't forget to reset before the next run
		results_list.append((EXPERIMENT_NAME, attempts, iterations, temp, run_time, best_fitness, stopped_at, func_calls))

# compile & save results
df_results = pd.DataFrame.from_records(results_list, columns=labels)
#df_results.to_csv(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.csv'))
df_results.to_excel(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.xlsx'))
df_results.to_pickle(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.pickle'))

# minimal output
print("# Best Run:")
print(df_results.loc[df_results['best_fitness'].idxmax()])
