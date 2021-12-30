import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project.CNN import CNNClassification
from project.aux import create_parameter_grid, n_points, create_overlapping
from project.defaults import get_default_cnn_classification


def hosa(model, parameters, X_train, X_test, y_train, y_test, tr):
    n_total_parameters, step = n_points(parameters)
    parameter_grid = create_parameter_grid(parameters)
    steps_check_improvment = (n_total_parameters // step)
    stop = False
    step = 0
    # According to the model, initialize the metrics and compare function
    if 'Regression' in str(model):
        initial_metric_value = np.inf
        compare_function = np.less
    elif 'Classification' in str(model):
        initial_metric_value = -np.inf
        compare_function = np.greater
    else:
        raise TypeError('The type of the model is invalid.')
    # Initalize variables
    best_net_prev = best_net_current = best_net = best_metric = None
    best_metric_prev = initial_metric_value
    best_metric_current = initial_metric_value
    overlapping_scenarios = overlapping_duration = None
    X_train_win = y_train_win = X_test_win = y_test_win = None
    # Get first specification
    specification = next(parameter_grid, None)
    # Test all parameters, until stop criterion is met or there is no more elements to test
    while not stop and specification is not None:
        # If necessary, create new overlap
        if overlapping_scenarios != specification['overlapping_scenarios'] or overlapping_duration != specification['overlapping_duration']:
            overlapping_scenarios = specification['overlapping_scenarios']
            overlapping_duration = specification['overlapping_duration']
            X_train_win, y_train_win = create_overlapping(X_train, y_train, overlapping_scenarios, overlapping_duration)
            X_test_win, y_test_win = create_overlapping(X_test, y_test, overlapping_scenarios, overlapping_duration)
        # Delete the overlapping parameters
        del specification['overlapping_duration']
        del specification['overlapping_scenarios']
        # Create and fit the model
        net = model(number_classes=2, **specification)
        net.prepare(X_train_win, y_train_win)
        net.compile()
        net.fit(X_train_win, y_train_win)
        # Collect metrics about the model
        metric, *_ = net.score(X_test_win, y_test_win)
        # Compare with the current metrics
        if compare_function(metric, best_metric_current):
            best_metric_current = metric
            best_net_current = net
        # Delete the network from the memory
        del net
        # Increment step
        step = step + 1
        # If we need to check the performance metrics
        if step % steps_check_improvment == 0:
            if best_metric_current - best_metric_prev <= tr:
                if compare_function(best_metric_current, best_metric_prev):
                    best_net = best_net_current
                    best_metric = best_metric_current
                else:
                    best_net = best_net_prev
                    best_metric = best_metric_prev
                stop = True
            else:
                best_net = best_net_current
                best_metric = best_metric_current
                # The previous becames the current
                best_net_prev = best_net_current
                best_metric_prev = best_metric_current
                # Delete current values
                best_metric_current = initial_metric_value
                best_net_current = None
        # Get next specification
        specification = next(parameter_grid, None)
    return best_net, best_metric


model = CNNClassification
parameters = get_default_cnn_classification()
tr = 0.01

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_net, best_metric = hosa(model, parameters, X_train, X_test, y_train, y_test, tr)
print(best_metric)
