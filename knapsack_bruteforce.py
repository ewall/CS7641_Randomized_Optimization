#!/usr/bin/env python

# Project 2: Randomized Optimization -- GT CS7641 Machine Learning, Fall 2019
# Eric W. Wallace, ewallace8-at-gatech-dot-edu, GTID 903105196

import mlrose
import numpy as np
import time
from itertools import product

SEED = 1

# 0-1 Knapsack Problem: can only choose each item once, trying to maximize value before filling up the sack
weights = [10, 5, 2, 8, 15, 11, 4, 7, 1, 20]
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
max_weight_pct = 0.6
fitness = mlrose.Knapsack(weights, values, max_weight_pct)
# perfect_score = 45

counter = 0
best_fitness = 0
start_time = time.perf_counter()
for s in product([0,1], repeat=10):
	counter += 1
	f = fitness.evaluate(np.array(s))
	if f > best_fitness:
		best_fitness = f
	print(f, s)
	# if f == perfect_score:
	# 	break
run_time = time.perf_counter() - start_time

print("\nBest fitness", best_fitness, "found in", run_time, "seconds with", counter, "iterations/function calls.")
