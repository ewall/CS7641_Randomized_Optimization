#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np
import os
import pandas as pd
import time

EXPERIMENT_NAME = "FourPeaks_Complexity"
OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# prep output
labels = ['complexity', 'algorithm', 'run_time', 'best_fitness', 'stopped_at', 'func_calls']
results_list = []

for complexity in (10, 20, 30, 40):
	# Four Peaks Problem with varying lengths
	fitness = mlrose.FourPeaks(t_pct=0.1)
	problem = mlrose.DiscreteOpt(length=complexity, fitness_fn=fitness, maximize=True, max_val=2)

	# RHC
	rhc_max_attempts = 75
	rhc_max_iters = 10000
	rhc_restarts = 25
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.random_hill_climb(problem,
	                                                  max_attempts=rhc_max_attempts,
	                                                  max_iters=rhc_max_iters,
	                                                  restarts=rhc_restarts,
	                                                  curve=True,
	                                                  random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (complexity, "RHC", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	# SA
	sa_max_attempts = 200
	sa_max_iters = 10000
	sa_schedule = mlrose.ArithDecay(init_temp=1)
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.simulated_annealing(problem,
	                                                    schedule=sa_schedule,
	                                                    max_attempts=sa_max_attempts,
	                                                    max_iters=sa_max_iters,
	                                                    curve=True,
	                                                    random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (complexity, "SA", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	# GA
	ga_max_attempts = 20000
	ga_max_iters = np.inf
	ga_pop_size = 200
	ga_mutation_prob = 0.1
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.genetic_alg(problem,
	                                            pop_size=ga_pop_size,
	                                            mutation_prob=ga_mutation_prob,
	                                            max_attempts=ga_max_attempts,
	                                            max_iters=ga_max_iters,
	                                            curve=True,
	                                            random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (complexity, "GA", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	# MIMIC
	mimic_max_attempts = 20000
	mimic_max_iters = np.inf
	mimic_pop_size = 750
	mimic_keep_pct = 0.5
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.mimic(problem,
	                                      pop_size=mimic_pop_size,
	                                      keep_pct=mimic_keep_pct,
	                                      max_attempts=mimic_max_attempts,
	                                      max_iters=mimic_max_iters,
	                                      curve=True,
	                                      random_state=SEED,
	                                      fast_mimic=True)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (complexity, "MIMIC", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

# compile & save results
df_results = pd.DataFrame.from_records(results_list, columns=labels)
df_results.to_excel(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.xlsx'))
df_results.to_pickle(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.pickle'))
