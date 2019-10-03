#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np

OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# Four Peaks problem -- with length 40, there are two local peaks of 40, and two optima of 75
fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True, max_val=2)

# GA runner
experiment_name = 'FourPeaks_GA'
ga = mlrose.runners.GARunner(problem=problem,
              experiment_name=experiment_name,
              output_directory=OUTPUT_DIRECTORY,
              seed=SEED,
              iteration_list=5 * np.arange(20) + 5,
              max_attempts=1000,
              population_sizes=[100, 200, 300],
              mutation_rates=[0.1, 0.3, 0.5])

df_run_stats, df_run_curves = ga.run()
