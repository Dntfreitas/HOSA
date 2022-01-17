import numpy as np


def get_default_cnn_classification():
    # GofL sizes
    n_layers = np.arange(1, 5)
    first_layers = 2 ** np.arange(4, 8)
    multiplication_factor = [0.5, 1, 2]
    gol_sizes_test = []
    for m in multiplication_factor:
        for f in first_layers:
            for n in n_layers:
                element = [f]
                if n == 1:
                    gol_sizes_test.append(element)
                else:
                    for _ in range(n - 1):
                        element.append(np.int(np.floor(f * m)))
                    gol_sizes_test.append(element)
    # Number of neurons of the first dense layer
    n_neurons_first_dense_layer_test = np.arange(50, 200, 50)
    # Overlapping duration
    overlapping_duration_test = np.insert(np.arange(1, 37, 2), 0, 0)
    # Overlapping scenarios
    overlapping_scenarios_test = ['left', 'central', 'right']
    # Compose the final parameter grid
    param_grid = [{
            'n_neurons_first_dense_layer': [10, 15, 20],
            'gol_sizes':                   [[3, 3], [4, 4]],
            'overlapping_type':            ['left'],
            'overlapping_epochs':          [3],
            'stride':                      [1],
    }]
    return param_grid
