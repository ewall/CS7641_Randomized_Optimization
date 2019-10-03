#!/usr/bin/env python -W ignore::DeprecationWarning

import mlrose
import numpy as np

OUTPUT_DIRECTORY = 'experiments'
SEED = 1

# Four Peaks problem -- with length 40, there are two local peaks of 40, and two optima of 75
fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=40, fitness_fn=fitness, maximize=True, max_val=2)

# GA runner
experiment_name = 'FourPeaks_SA'
sa = mlrose.runners.SARunner(problem=problem,
              experiment_name=experiment_name,
              output_directory=OUTPUT_DIRECTORY,
              seed=SEED,
              iteration_list=20 * np.arange(20) + 20,
              max_attempts=5000,
              temperature_list=10 ** np.arange(5))

df_run_stats, df_run_curves = sa.run()
