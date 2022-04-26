import logging
import time

from sklearn.model_selection import ParameterGrid

from utils import *

logging.basicConfig(level=logging.INFO, filename='logs/grid.log', filemode='w', format=FORMAT)

# Load the data
prepare_data()
# Prepare to store the information about the best global solution
solution_best_global = None
solution_fitness_best_global = -np.inf
# Generate grid
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
        timesteps = parameters_to_test['timesteps']
        n_units = parameters_to_test['n_units']
        n_subs_layers = parameters_to_test['n_subs_layers']
        is_bidirectional = parameters_to_test['is_bidirectional']
        overlapping_type = parameters_to_test['overlapping_type']
        overlapping_epochs = parameters_to_test['overlapping_epochs']
        mults = parameters_to_test['mults']
        # Test the combination
        solution = encode(timesteps, n_units, n_subs_layers, is_bidirectional, overlapping_type,
                          overlapping_epochs, mults)
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
