import logging

import numpy as np
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences

# Logging option
from hosa.helpers import create_overlapping
from hosa.models.rnn import RNNClassification

FORMAT = '%(levelname)s — %(asctime)s — %(message)s: Iteration: %(iteration)s — Run: %(run)s/%(' \
         'total_run)s — Best parameters: %(best_parameters)s — Best parameters fitness: ' \
         '%(best_parameters_fitness)s — Duration: %(duration)s'
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format=FORMAT)

x_train = None
y_train = None
x_test = None
y_test = None


def encode(n_timesteps, n_units, n_subs_layers, is_bidirectional, overlapping_type,
           overlapping_duration, n_neurons_dense_layer):
    # Encode number of timesteps
    if n_timesteps == 5:
        n_timesteps_encoded = [0, 0]
    elif n_timesteps == 15:
        n_timesteps_encoded = [0, 1]
    elif n_timesteps == 25:
        n_timesteps_encoded = [1, 0]
    elif n_timesteps == 35:
        n_timesteps_encoded = [1, 1]
    # Encode number of hidden units
    if n_units == 100:
        n_units_encoded = [0, 0]
    elif n_units == 200:
        n_units_encoded = [0, 1]
    elif n_units == 300:
        n_units_encoded = [1, 0]
    elif n_units == 400:
        n_units_encoded = [1, 1]
    # Encode number of recurrent layers
    if n_subs_layers == 1:
        n_subs_layers_encoded = [0, 0, 0]
    if n_subs_layers == 2:
        n_subs_layers_encoded = [0, 0, 1]
    if n_subs_layers == 3:
        n_subs_layers_encoded = [0, 1, 0]
    if n_subs_layers == 4:
        n_subs_layers_encoded = [0, 1, 1]
    elif n_subs_layers == 5:
        n_subs_layers_encoded = [1, 0, 0]
    # Encode "is bidirectional?"
    if is_bidirectional:
        is_bidirectional_encode = [1]
    else:
        is_bidirectional_encode = [0]
    # Encode overlapping scenario
    if overlapping_type == 'central':
        overlapping_type_encode = [0, 0]
    elif overlapping_type == 'right':
        overlapping_type_encode = [0, 1]
    elif overlapping_type == 'left':
        overlapping_type_encode = [1, 0]
    # Encode overlapping duration
    if overlapping_duration == 0:
        overlapping_duration_encode = [0, 0]
    elif overlapping_duration == 1:
        overlapping_duration_encode = [0, 1]
    elif overlapping_duration == 2:
        overlapping_duration_encode = [1, 0]
    elif overlapping_duration == 3:
        overlapping_duration_encode = [1, 1]
    # Encode the number of neurons of the dense layer
    if n_neurons_dense_layer == 0.5:
        n_neurons_dense_layer_encode = [0, 0]
    elif n_neurons_dense_layer == 1:
        n_neurons_dense_layer_encode = [0, 1]
    if n_neurons_dense_layer == 2:
        n_neurons_dense_layer_encode = [1, 0]
    return np.concatenate((n_timesteps_encoded, n_units_encoded, n_subs_layers_encoded,
                           is_bidirectional_encode, overlapping_type_encode,
                           overlapping_duration_encode, n_neurons_dense_layer_encode))


def decode(chromosome):
    n_timesteps = chromosome[:2]
    n_units = chromosome[2:4]
    n_subs_layers = chromosome[4:7]
    is_bidirectional = chromosome[7]
    overlapping_type = chromosome[8:10]
    overlapping_duration = chromosome[10:12]
    n_neurons_dense_layer_mult = chromosome[12:14]
    # Decode number of timesteps
    if np.array_equal(n_timesteps, [0, 0]):
        n_timesteps = 5
    elif np.array_equal(n_timesteps, [0, 1]):
        n_timesteps = 15
    elif np.array_equal(n_timesteps, [1, 0]):
        n_timesteps = 25
    elif np.array_equal(n_timesteps, [1, 1]):
        n_timesteps = 35
    # Decode number of hidden units
    if np.array_equal(n_units, [0, 0]):
        n_units = 100
    elif np.array_equal(n_units, [0, 1]):
        n_units = 200
    elif np.array_equal(n_units, [1, 0]):
        n_units = 300
    elif np.array_equal(n_units, [1, 1]):
        n_units = 400
    # Decode number of recurrent layers
    if np.array_equal(n_subs_layers, [0, 0, 0]):
        n_subs_layers = 1
    elif np.array_equal(n_subs_layers, [0, 0, 1]):
        n_subs_layers = 2
    elif np.array_equal(n_subs_layers, [0, 1, 0]):
        n_subs_layers = 3
    elif np.array_equal(n_subs_layers, [0, 1, 1]):
        n_subs_layers = 4
    else:
        n_subs_layers = 5
    # Decode "is bidirectional?"
    if is_bidirectional == 0:
        is_bidirectional = False
    else:
        is_bidirectional = True
    # Decode overlapping scenario
    if np.array_equal(overlapping_type, [0, 0]):
        overlapping_type = 'central'
    elif np.array_equal(overlapping_type, [0, 1]):
        overlapping_type = 'right'
    else:
        overlapping_type = 'left'
    # Decode overlapping duration
    if np.array_equal(overlapping_duration, [0, 0]):
        overlapping_duration = 0
    elif np.array_equal(overlapping_duration, [0, 1]):
        overlapping_duration = 1
    elif np.array_equal(overlapping_duration, [1, 0]):
        overlapping_duration = 2
    elif np.array_equal(overlapping_duration, [1, 1]):
        overlapping_duration = 3
    # Decode the number of neurons of the dense layer
    if np.array_equal(n_neurons_dense_layer_mult, [0, 0]):
        n_neurons_dense_layer_mult = 0.5
    elif np.array_equal(n_neurons_dense_layer_mult, [0, 1]):
        n_neurons_dense_layer_mult = 1
    else:
        n_neurons_dense_layer_mult = 2
    return n_timesteps, n_units, n_subs_layers, is_bidirectional, overlapping_type, \
           overlapping_duration, n_neurons_dense_layer_mult


def prepare_data():
    global x_train, y_train, x_test, y_test
    num_distinct_words = 5000
    max_sequence_length = 300
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_train = pad_sequences(x_train, maxlen=max_sequence_length, value=0.0)
    x_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)


def fitness_func(solution, solution_idx=None):
    global x_train, y_train, x_test, y_test
    n_timesteps, n_units, n_subs_layers, is_bidirectional, overlapping_type, \
    overlapping_duration, n_neurons_dense_layer_mult = decode(solution)
    n_neurons_dense_layer = np.int(np.floor(n_neurons_dense_layer_mult * n_units))
    # Overlap the data
    x_train_overlapped, y_train_overlapped = create_overlapping(x_train, y_train, RNNClassification,
                                                                overlapping_duration,
                                                                overlapping_type,
                                                                n_timesteps=n_timesteps)
    # Create the model and fit
    clf = RNNClassification(n_outputs=2, n_neurons_dense_layer=n_neurons_dense_layer,
                            is_bidirectional=is_bidirectional, n_units=n_units,
                            n_subs_layers=n_subs_layers, model_type='lstm')
    clf.prepare(x_train_overlapped, y_train_overlapped)
    clf.compile()
    clf.fit(x_train_overlapped, y_train_overlapped, verbose=0)
    fitness, *_ = clf.score(x_train_overlapped, y_train_overlapped)
    return fitness


def format_log(iteration, run, total_run, best_parameters_fitness, best_parameters, durantion):
    return {'iteration':               iteration, 'run': run, 'total_run': total_run,
            'best_parameters_fitness': best_parameters_fitness, 'best_parameters': best_parameters,
            'duration':                durantion}
