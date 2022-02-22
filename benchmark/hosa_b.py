import logging
import time

from hosa.optimization.hosa import HOSARNN
from utils import *

logging.basicConfig(level=logging.INFO, filename='benchmark/hosa.log', filemode='w', format=FORMAT)

# Load the data
prepare_data()
# Run HOSA
d = format_log(None, None, None, None, None, None)
logging.info('**** Starting HOSA ****', extra=d)
# Start HOSA
clf = HOSARNN(x_train, y_train, RNNClassification, 2, param_grid, 0.01, apply_rsv=False)
start = time.time()
clf.fit(max_n_subs_layers=5, show_progress=True, verbose=0, shuffle=False,
        imbalance_correction=False)
end = time.time()
duration = end - start
# Compute fitness
solution_fitness_best_global = clf.score(x_train, x_test)
# Extract the parameters
best_parameters = clf.get_params()
n_timesteps = best_parameters['n_timesteps']
n_units = best_parameters['n_units']
n_subs_layers = best_parameters['n_subs_layers']
is_bidirectional = best_parameters['is_bidirectional']
overlapping_type = best_parameters['overlapping_type']
overlapping_duration = best_parameters['overlapping_duration']
n_neurons_dense_layer = best_parameters['n_neurons_dense_layer']
# Encode the solution
solution_best_global = encode(n_timesteps, n_units, n_subs_layers, is_bidirectional,
                              overlapping_type, overlapping_duration, n_neurons_dense_layer)
# Log the information
d = format_log(None, None, None, solution_fitness_best_global, solution_best_global, duration)
logging.info('**** Best solution of the grid search ****', extra=d)
