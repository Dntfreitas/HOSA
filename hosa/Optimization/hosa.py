import numpy as np
from sklearn.model_selection import ShuffleSplit

from hosa.aux import create_parameter_grid, n_points, create_overlapping


class HOSA:
    def __init__(self, model, number_outputs, parameters, X, y, tr, apply_rsv=True, validation_size=.25, n_splits=10):
        self.model, self.number_outputs, self.parameters, self.X, self.y, self.tr, self.apply_rsv, self.validation_size, self.n_splits = model, number_outputs, parameters, X, y, tr, apply_rsv, validation_size, n_splits
        self.best_net = self.best_metric = self.best_specification = None
        n_total_parameters, step = n_points(parameters)
        self.parameter_grid = create_parameter_grid(parameters)
        self.steps_check_improvment = (n_total_parameters // step)
        # According to the model, initialize the metrics and compare function
        if 'Regression' in str(model):
            self.initial_metric_value = np.inf
            self.compare_function = np.less
            self.stop_check = self.__stop_check_decrease
        elif 'Classification' in str(model):
            self.initial_metric_value = -np.inf
            self.compare_function = np.greater
            self.stop_check = self.__stop_check_increase
        else:
            raise TypeError('The type of the model is invalid.')

    def __prepare_param_overlapping(self, specification):
        overlapping_epochs = specification['overlapping_epochs']
        if overlapping_epochs > 0 and 'overlapping_type' in specification:
            overlapping_type = specification['overlapping_type']
        else:
            overlapping_type = None
        if 'stride' in specification:
            stride = specification['stride']
        else:
            stride = 1
        if 'timesteps' in specification:
            timesteps = specification['timesteps']
        else:
            timesteps = None
        return overlapping_type, overlapping_epochs, stride, timesteps

    def __stop_check_increase(self, best_metric_current, best_metric_prev):
        return best_metric_current - best_metric_prev <= self.tr

    def __stop_check_decrease(self, best_metric_current, best_metric_prev):
        return best_metric_prev - best_metric_current <= self.tr

    def assess_model(self, net, X_win, y_win, **kwargs):
        # If we need to use random sub-sampling validation
        if self.apply_rsv:
            # Generate the splits
            rs = ShuffleSplit(n_splits=self.n_splits, test_size=self.validation_size)
            metrics = np.empty(self.n_splits)
            i = 0
            for train_index, validation_index in rs.split(X_win):
                X_win_train, y_win_train = X_win[train_index], y_win[train_index]
                X_win_validation, y_win_validation = X_win[validation_index], y_win[validation_index]
                net.fit(X_win_train, y_win_train, **kwargs)
                # Collect metrics about the model
                metric, *_ = net.score(X_win_validation, y_win_validation)
                metrics[i] = metric
                i = i + 1
            return np.mean(metrics)
        else:
            net.fit(X_win, y_win, **kwargs)
            # Collect metrics about the model
            metric, *_ = net.score(X_win, y_win)
            return metric

    def fit(self, **kwargs):
        # Initalize variables
        best_net_prev = best_net_current = None
        best_specification_current = best_specification_prev = None
        best_metric_prev = self.initial_metric_value
        best_metric_current = self.initial_metric_value
        overlapping_type = overlapping_epochs = stride = timesteps = None
        X_win = y_win = None
        stop = False
        step = 0
        # Get first specification
        specification = next(self.parameter_grid, None)
        # Test all parameters, until stop criterion is met or there is no more elements to test
        while not stop and specification is not None:
            # If necessary, create new overlap
            overlapping_type_new, overlapping_epochs_new, stride_new, timesteps_new = self.__prepare_param_overlapping(specification)
            changed = overlapping_type != overlapping_type_new or overlapping_epochs != overlapping_epochs_new or stride != stride_new or timesteps != timesteps_new
            if changed:
                overlapping_type, overlapping_epochs, stride, timesteps = overlapping_type_new, overlapping_epochs_new, stride_new, timesteps_new
                X_win, y_win = create_overlapping(self.X, self.y, self.model, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
            # Generate the model
            net = self.model(number_outputs=self.number_outputs, **specification)
            net.prepare(X_win, y_win)
            net.compile()
            # Asses the model
            metric = self.assess_model(net, X_win, y_win, **kwargs)
            # Compare with the current metrics
            if self.compare_function(metric, best_metric_current):
                best_metric_current = metric
                best_net_current = net
                best_specification_current = specification
            # Increment step
            step = step + 1
            # If we need to check the performance metrics
            if step % self.steps_check_improvment == 0:
                if self.stop_check(best_metric_current, best_metric_prev):
                    if self.compare_function(best_metric_current, best_metric_prev):
                        self.best_net = best_net_current
                        self.best_metric = best_metric_current
                        self.best_specification = best_specification_current
                    else:
                        self.best_net = best_net_prev
                        self.best_metric = best_metric_prev
                        self.best_specification = best_specification_prev
                    stop = True
                else:
                    self.best_net = best_net_current
                    self.best_metric = best_metric_current
                    self.best_specification = best_specification_current
                    # The previous becames the current
                    best_net_prev = best_net_current
                    best_metric_prev = best_metric_current
                    best_specification_prev = best_specification_current
                    # Delete current values
                    best_metric_current = self.initial_metric_value
                    best_net_current = None
                    best_specification_current = None
            # Get next specification
            specification = next(self.parameter_grid, None)
        return self.best_net

    def get_params(self):
        return self.best_specification

    def predict(self, X, **kwargs):
        overlapping_type, overlapping_epochs, stride, timesteps = self.__prepare_param_overlapping(self.get_params())
        X, y = create_overlapping(X, None, self.best_net, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
        return self.best_net.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        overlapping_type, overlapping_epochs, stride, timesteps = self.__prepare_param_overlapping(self.get_params())
        X, y = create_overlapping(X, y, self.best_net, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
        return self.best_net.score(X, y, **kwargs)
