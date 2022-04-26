import time

from hosa.optimization.hosa import HOSARNN
from utils import *

# Load the data
x_train, y_train, x_test, y_test = prepare_data()

x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# Start HOSA
del (param_grid['n_subs_layers'])
clf = HOSARNN(x_train, y_train, RNNClassification, 2, param_grid, 0.01, apply_rsv=False, log_path='logs/hosa.log')
start = time.time()
clf.fit(max_n_subs_layers=5, show_progress=True, verbose=0, shuffle=False, imbalance_correction=False)
# Compute fitness
solution_fitness_best_global = clf.score(x_test, y_test)
# Extract the parameters
best_parameters = clf.get_params()
timesteps = best_parameters['timesteps']
n_units = best_parameters['n_units']
n_subs_layers = best_parameters['n_subs_layers']
is_bidirectional = best_parameters['is_bidirectional']
overlapping_type = best_parameters['overlapping_type']
overlapping_epochs = best_parameters['overlapping_epochs']
n_neurons_dense_layer = best_parameters['n_neurons_dense_layer']
