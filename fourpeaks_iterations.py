#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import os
import pandas as pd
import time

EXPERIMENT_NAME = "FourPeaks_Iterations"
OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# prep output
labels = ['iterations', 'algorithm', 'run_time', 'best_fitness', 'stopped_at', 'func_calls']
results_list = []

# Four Peaks Problem: with length of 40, there are two local maxima of 40, and two optima of 75
fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True, max_val=2)

for iterations in (10, 100, 1000, 10000, 20000):
	# RHC
	rhc_max_attempts = 75
	rhc_restarts = 25
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.random_hill_climb(problem,
	                                                  max_attempts=rhc_max_attempts,
	                                                  max_iters=iterations,
	                                                  restarts=rhc_restarts,
	                                                  curve=True,
	                                                  random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (iterations, "RHC", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	# SA
	sa_max_attempts = 200
	sa_schedule = mlrose.ArithDecay(init_temp=1)
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.simulated_annealing(problem,
	                                                    schedule=sa_schedule,
	                                                    max_attempts=sa_max_attempts,
	                                                    max_iters=iterations,
	                                                    curve=True,
	                                                    random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (iterations, "SA", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	# GA
	ga_max_attempts = 20000
	ga_pop_size = 200
	ga_mutation_prob = 0.1
	start_time = time.perf_counter()
	_, best_fitness, curve = mlrose.genetic_alg(problem,
	                                            pop_size=ga_pop_size,
	                                            mutation_prob=ga_mutation_prob,
	                                            max_attempts=ga_max_attempts,
	                                            max_iters=iterations,
	                                            curve=True,
	                                            random_state=SEED)
	run_time = time.perf_counter() - start_time
	func_calls = problem.get_function_calls()
	problem.reset_function_calls()  # don't forget to reset before the next run

	run_result = (iterations, "GA", run_time, best_fitness, len(curve), func_calls)
	print(run_result)
	results_list.append(run_result)

	if iterations < 1000:  # ???
		# MIMIC
		mimic_max_attempts = 20000
		mimic_pop_size = 750
		mimic_keep_pct = 0.5
		start_time = time.perf_counter()
		_, best_fitness, curve = mlrose.mimic(problem,
		                                      pop_size=mimic_pop_size,
		                                      keep_pct=mimic_keep_pct,
		                                      max_attempts=mimic_max_attempts,
		                                      max_iters=iterations,
		                                      curve=True,
		                                      random_state=SEED,
		                                      fast_mimic=True)
		run_time = time.perf_counter() - start_time
		func_calls = problem.get_function_calls()
		problem.reset_function_calls()  # don't forget to reset before the next run

		run_result = (iterations, "MIMIC", run_time, best_fitness, len(curve), func_calls)
		print(run_result)
		results_list.append(run_result)

# compile & save results
df_results = pd.DataFrame.from_records(results_list, columns=labels)
df_results.to_excel(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.xlsx'))
df_results.to_pickle(os.path.join(OUTPUT_DIRECTORY, EXPERIMENT_NAME + '.pickle'))
