import logging
import time

from sklearn.model_selection import ParameterGrid

from utils import *

logging.basicConfig(level=logging.INFO, filename='grid.log', filemode='w', format=FORMAT)

# Load the data
prepare_data()
# Prepare to store the information about the best global solution
solution_best_global = None
solution_fitness_best_global = -np.inf
# Generate grid
param_grid = {'n_timesteps':           [5, 15, 25, 35],
              'n_units':               [100, 200, 300, 400],
              'n_subs_layers':         [1, 2, 3, 4],
              'is_bidirectional':      [False, True],
              'overlapping_type':      ['central', 'right', 'left'],
              'overlapping_duration':  [0, 1, 2, 3],
              'n_neurons_dense_layer': [0.5, 1, 2]
              }
grid_gen = ParameterGrid(param_grid)
grid = grid_gen.__iter__()
total_run = grid_gen.__len__()
i = 1
# Run on every combination
start = time.time()
try:
    while True:
        # Generate the next iteraion
        parameters_to_test = grid.__next__()
        # Map the parameters
        n_timesteps = parameters_to_test['n_timesteps']
        n_units = parameters_to_test['n_units']
        n_subs_layers = parameters_to_test['n_subs_layers']
        is_bidirectional = parameters_to_test['is_bidirectional']
        overlapping_type = parameters_to_test['overlapping_type']
        overlapping_duration = parameters_to_test['overlapping_duration']
        n_neurons_dense_layer = parameters_to_test['n_neurons_dense_layer']
        # Test the combination
        solution = encode(n_timesteps, n_units, n_subs_layers, is_bidirectional, overlapping_type,
                          overlapping_duration, n_neurons_dense_layer)
        fitness = fitness_func(solution)
        # Log information
        d = format_log(None, i, total_run, fitness, solution, None)
        logging.info('**** Tested new set of parameters ****', extra=d)
        if fitness > solution_fitness_best_global:
            solution_best_global = solution
            solution_fitness_best_global = fitness
        i = i + 1
except StopIteration:
    end = time.time()
    duration = end - start
    d = format_log(None, None, None, solution_fitness_best_global, solution_best_global, duration)
    logging.info('**** Best solution of the grid search ****', extra=d)
