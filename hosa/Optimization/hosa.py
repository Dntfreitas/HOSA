import abc

import numpy as np
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from hosa.aux import create_parameter_grid, n_points, create_overlapping, prepare_param_overlapping


class HOSA:
    def __init__(self, X, y, model, n_outputs, parameters, tr, apply_rsv=True, validation_size=.25, n_splits=10):
        """ Heuristic Oriented Search Algorithm (HOSA)

        This class implments the HOSA. Following a heuristic search, the algorithm finetunes the most relevant models' parameters. Thus, HOSA avoids testing every possible combination, and therefore, an exhaustive search.

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.HOSACNN` or :class:`.HOSARNN`.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            model (object): Class of the object to be optimized. Available options are: :class:`.RNNClassification`, :class:`.RNNRegression`, :class:`.CNNClassification` and :class:`.CNNRegression`.
            n_outputs (int): Number of class labels in classification, or the number of numerical values to predict in regression.
            parameters (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            tr (float): Minimum threshold of improvement of the performance metric.
            apply_rsv (bool): ``True`` if random sub-sampling validation should be used during the optimization procedure.
            validation_size (float): Proportion of the dataset to include in the validation split on the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
            n_splits (int): Number of splits used in the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
        """
        self.model, self.n_outputs, self.parameters, self.X, self.y, self.tr, self.apply_rsv, self.validation_size, self.n_splits = model, n_outputs, [parameters], X, y, tr, apply_rsv, validation_size, n_splits
        # Check the type of the model
        self.is_cnn = 'CNNRegression' in str(self.model) or 'CNNClassification' in str(self.model)
        self.is_rnn = 'RNNRegression' in str(self.model) or 'RNNClassification' in str(self.model)
        # Check if the parameters are valid
        self.__check_params()
        # Intialize variables
        self.best_model = self.best_metric = self.best_specification = None
        n_total_parameters, step = n_points(self.parameters)
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

    def __stop_check_increase(self, best_metric_current, best_metric_prev):
        """

        Checks if the stopping criterion is met for maximization problems (e.g., classification problems).

        Args:
            best_metric_current (float): Current best metric found.
            best_metric_prev (float): Previous best metric found.

        Returns:
            bool: Returns ``True`` if the HOSA procedure should be stopped. ``False`` otherwise.

        """
        return best_metric_current - best_metric_prev <= self.tr

    def __stop_check_decrease(self, best_metric_current, best_metric_prev):
        """

        Checks if the stopping criterion is met for minimization problems (e.g., regression problems).

        Args:
            best_metric_current (float): Current best metric found.
            best_metric_prev (float): Previous best metric found.

        Returns:
            bool: Returns ``True`` if the HOSA procedure should be stopped. ``False`` otherwise.

        """
        return best_metric_prev - best_metric_current <= self.tr

    def __fit_assess_model(self, model, X_win, y_win, **kwargs):
        """

        Fits the model and computes the chosen performance metric of the target values based on the inputs.

        Args:
            model (tensorflow.keras.Sequential): Model to be fitted and assessed.
            X_win (numpy.ndarray): Windowed input data.
            y_win (numpy.ndarray): Windowed target values (class labels in classification, real numbers in regression).
            **kwargs: Extra arguments explicitly used for regression or classification models, including the additional arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            float: Returns the value of the chosen performance metric on the fitted model.

        """
        # If we need to use random sub-sampling validation
        if self.apply_rsv:
            # Generate the splits
            rs = ShuffleSplit(n_splits=self.n_splits, test_size=self.validation_size)
            metrics = np.empty(self.n_splits)
            i = 0
            for train_index, validation_index in rs.split(X_win):
                X_win_train, y_win_train = X_win[train_index], y_win[train_index]
                X_win_validation, y_win_validation = X_win[validation_index], y_win[validation_index]
                model.fit(X_win_train, y_win_train, **kwargs)
                # Collect metrics about the model
                metric, *_ = model.score(X_win_validation, y_win_validation)
                metrics[i] = metric
                i = i + 1
            return np.mean(metrics)
        else:
            model.fit(X_win, y_win, **kwargs)
            # Collect metrics about the model
            metric, *_ = model.score(X_win, y_win)
            return metric

    def __check_params(self):
        """
        According to the type of model, checks if the user has specified all the mandatory parameters.

        Raises:
            ValueError: If the user did not set some mandatory parameter.

        """
        for parameter in self.required_parameters:
            if parameter not in self.parameters[0]:
                raise ValueError('The parameter `' + parameter + '` must be specified in the `parameters` dictionary.')

    def grid_search(self, n_kernels=None, n_neurons_dense_layer=None, n_units=None, n_subs_layers=None, **kwargs):
        """
        Runs a grid search on the remaining moldel's parameters.

        Args:
            n_kernels (list or None): *i*-th element represents the number of output filters of the convolution layer in the *i*-th GofLayer. **Ignored in the case of optimizing an RNN**.
            n_neurons_dense_layer (int or None): Number of neurons units of the penultimate dense layer (i.e., before the output layer). **Ignored in the case of optimizing an CNN**.
            n_units (int or None): Dimensionality of the output space, i.e., the dimensionality of the hidden state. **Ignored in the case of optimizing an CNN**.
            n_subs_layers (int or None): **Ignored in the case of optimizing an CNN**.
            **kwargs: Extra arguments explicitly used for regression or classification models, including the additional arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns the best TensorFlow model found.

        """
        # Initalize variables
        best_model = best_specification = None
        best_metric = self.initial_metric_value
        overlapping_type = overlapping_epochs = stride = timesteps = None
        X_win = y_win = None
        # Generate parameter grid
        parameter_grid = create_parameter_grid(self.parameters)
        # Get first specification
        specification = next(parameter_grid, None)
        # Test all parameters, until stop criterion is met or there is no more elements to test
        while specification is not None:
            # If necessary, create new overlap
            overlapping_type_new, overlapping_epochs_new, stride_new, timesteps_new = prepare_param_overlapping(specification)
            changed = overlapping_type != overlapping_type_new or overlapping_epochs != overlapping_epochs_new or stride != stride_new or timesteps != timesteps_new
            if changed:
                overlapping_type, overlapping_epochs, stride, timesteps = overlapping_type_new, overlapping_epochs_new, stride_new, timesteps_new
                X_win, y_win = create_overlapping(self.X, self.y, self.model, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
            # Generate the model
            if self.is_cnn:
                model = self.model(n_kernels=n_kernels, n_outputs=self.n_outputs, **specification)
            elif self.is_rnn:
                model = self.model(n_outputs=self.n_outputs, n_neurons_dense_layer=n_neurons_dense_layer, n_units=n_units, n_subs_layers=n_subs_layers, **specification)
            model.prepare(X_win, y_win)
            model.compile()
            # Fit and asses the model
            metric = self.__fit_assess_model(model, X_win, y_win, **kwargs)
            # Compare with the current metrics
            if self.compare_function(metric, best_metric):
                best_metric = metric
                best_model = model
                best_specification = specification
            # Get next specification
            specification = next(parameter_grid, None)
        return best_specification, best_model, best_metric

    def get_params(self):
        """
        Get parameters for the best model found.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return self.best_specification

    def get_model(self):
        """
        Get the best model found.

        Returns:
            tensorflow.keras.Sequential: Returns the best TensorFlow model found.

        """
        return self.best_model.model

    def predict(self, X, **kwargs):
        """
        Predicts the target values using the input data in the best model found.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.

        Returns:
            numpy.ndarray: Returns an array containing the estimates that were obtained on the best-fitted model found.
        """
        overlapping_type, overlapping_epochs, stride, timesteps = prepare_param_overlapping(self.get_params())
        X, y = create_overlapping(X, None, self.best_model, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
        return self.best_model.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        """
        Computes the performance metrics on the given input data and target values in the best model found.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            **kwargs: Only used for classification, in order to set the value of the parameter ``inbalance_correction``.

        Returns:
            tuple: Returns a tuple containing the performance metric according to the type of model.
        """
        overlapping_type, overlapping_epochs, stride, timesteps = prepare_param_overlapping(self.get_params())
        X, y = create_overlapping(X, y, self.best_model, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
        return self.best_model.score(X, y, **kwargs)

    @abc.abstractmethod
    def fit(self, **kwargs):
        """
        Optimize the model following the HOSA approach with all sets of parameters.

        Args:
            **kwargs: Extra arguments explicitly used for regression or classification models.
        """
        raise NotImplementedError


class HOSACNN(HOSA):

    def __init__(self, X, y, model, n_outputs, parameters, tr, apply_rsv=True, validation_size=.25, n_splits=10):
        """ Heuristic Oriented Search Algorithm (HOSA) for CNNs.

        This class implments the HOSA for optimizing CNNs. Following a heuristic search, the algorithm finetunes the most relevant models' parameters. Thus, HOSA avoids testing every possible combination, and therefore, an exhaustive search.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            model (object): Class of the object to be optimized. Available options are: :class:`.RNNClassification`, :class:`.RNNRegression`, :class:`.CNNClassification` and :class:`.CNNRegression`.
            n_outputs (int): Number of class labels in classification, or the number of numerical values to predict in regression.
            parameters (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            tr (float): Minimum threshold of improvement of the performance metric.
            apply_rsv (bool): ``True`` if random sub-sampling validation should be used during the optimization procedure.
            validation_size (float): Proportion of the dataset to include in the validation split on the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
            n_splits (int): Number of splits used in the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
        """
        self.required_parameters = ['n_kernels_first_gol', 'mults']
        super().__init__(X, y, model, n_outputs, parameters, tr, apply_rsv, validation_size, n_splits)
        self.n_kernels_first_gol = self.parameters[0]['n_kernels_first_gol']
        self.mults = self.parameters[0]['mults']
        del (self.parameters[0]['n_kernels_first_gol'])
        del (self.parameters[0]['mults'])

    def fit(self, max_gol_sizes, show_progress=True, **kwargs):
        """
        Optimize the model following the HOSA approach with all sets of parameters.

        Args:
            max_gol_sizes (int): Maximum number of GofLayers to add to the model.
            show_progress (bool): `True` to show a progress bar; `False` otherwise.
            **kwargs: Extra arguments explicitly used for regression or classification models, including the additional arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tuple: Returns a tuple containing the object of the best model found and respective performance metric and optimized parameters.
        """
        # Initalize variables
        best_metric = self.initial_metric_value
        best_model = best_specification = None
        k_construction_size = []
        stop = False
        # Show progess bar?
        with tqdm(total=max_gol_sizes, disable=not show_progress, colour='green') as pbar_all:
            # Perform optimization
            while len(k_construction_size) < max_gol_sizes and not stop:
                # Inicialize the best current metric for comparing with the best metric found
                best_metric_current = self.initial_metric_value
                best_model_current = best_specification_current = None
                # If there is just one GofL
                if len(k_construction_size) == 0:
                    n_kernels_test = np.array(self.n_kernels_first_gol).reshape((len(self.n_kernels_first_gol), 1))
                else:
                    n_kernels_test = [k_construction_size + [np.floor(k_construction_size[-1] * mult)] for mult in self.mults]
                # Test each kernel size
                for n_kernel in n_kernels_test:
                    # Run grid search
                    specification, model, metric = self.grid_search(n_kernel, **kwargs)
                    # Compare with the current metrics, and update the current best values if necessary
                    if self.compare_function(metric, best_metric_current):
                        best_model_current = model
                        best_metric_current = metric
                        best_specification_current = specification
                # Check the stopping criterion
                if self.stop_check(best_metric_current, best_metric):
                    if self.compare_function(best_metric_current, best_metric):
                        self.best_model = best_model_current
                        self.best_metric = best_metric_current
                        self.best_specification = best_specification_current
                    else:
                        self.best_model = best_model
                        self.best_metric = best_metric
                        self.best_specification = best_specification
                    stop = True
                else:
                    best_model = best_model_current
                    best_metric = best_metric_current
                    best_specification = best_specification_current
                    k_construction_size.append(best_model.n_kernels[-1])
                # Update progress bar
                best_specification_complete = best_model.__dict__()
                pbar_all.set_postfix(n_goflayers=len(best_specification_complete['n_kernels']), no_kernels=best_specification_complete['n_kernels'])
                pbar_all.update(1)
        best_specification.update(self.best_model.__dict__())
        self.best_model, self.best_metric, self.best_specification = best_model, best_metric, best_specification
        return self.best_model, self.best_metric, self.best_specification


class HOSARNN(HOSA):

    def __init__(self, X, y, model, n_outputs, parameters, tr, apply_rsv=True, validation_size=.25, n_splits=10):
        """ Heuristic Oriented Search Algorithm (HOSA) for RNNs.

        This class implments the HOSA for optimizing RNNs. Following a heuristic search, the algorithm finetunes the most relevant models' parameters. Thus, HOSA avoids testing every possible combination, and therefore, an exhaustive search.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            model (object): Class of the object to be optimized. Available options are: :class:`.RNNClassification`, :class:`.RNNRegression`, :class:`.CNNClassification` and :class:`.CNNRegression`.
            n_outputs (int): Number of class labels in classification, or the number of numerical values to predict in regression.
            parameters (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
            tr (float): Minimum threshold of improvement of the performance metric.
            apply_rsv (bool): ``True`` if random sub-sampling validation should be used during the optimization procedure.
            validation_size (float): Proportion of the dataset to include in the validation split on the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
            n_splits (int): Number of splits used in the random sub-sampling validation. **Ignored if ``apply_rsv = False``**.
        """
        self.required_parameters = ['n_hidden_units', 'mults']
        super().__init__(X, y, model, n_outputs, parameters, tr, apply_rsv, validation_size, n_splits)
        self.n_hidden_units = self.parameters[0]['n_hidden_units']
        self.mults = self.parameters[0]['mults']
        del (self.parameters[0]['n_hidden_units'])
        del (self.parameters[0]['mults'])

    def fit(self, max_n_subs_layers, show_progress=True, **kwargs):
        """
        Optimize the model following the HOSA approach with all sets of parameters.

        Args:
            max_n_subs_layers (int): Maximum number of subsequent layers to add to the model.
            **kwargs: Extra arguments explicitly used for regression or classification models, including the additional arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tuple: Returns a tuple containing the object of the best model found and respective performance metric and optimized parameters.
        """
        # Initalize variables
        best_metric = self.initial_metric_value
        best_model = best_specification = None
        n_subs_layers_construction = 1
        stop = False
        # Show progess bar?
        with tqdm(total=max_n_subs_layers, disable=not show_progress, colour='green') as pbar_all:
            # Perform optimization
            while n_subs_layers_construction < max_n_subs_layers and not stop:
                # Inicialize the best current metric for comparing with the best metric found
                best_metric_current = self.initial_metric_value
                best_model_current = best_specification_current = None
                # Test each number of hidden units
                for n_units in self.n_hidden_units:
                    # Test each number of units in the dense layer
                    for mult in self.mults:
                        # Run grid search
                        specification, model, metric = self.grid_search(n_neurons_dense_layer=np.floor(n_units * mult), n_units=n_units, n_subs_layers=n_subs_layers_construction, **kwargs)
                        # Compare with the current metrics, and update the current best values if necessary
                        if self.compare_function(metric, best_metric_current):
                            best_model_current = model
                            best_metric_current = metric
                            best_specification_current = specification
                # Check the stopping criterion
                if self.stop_check(best_metric_current, best_metric):
                    if self.compare_function(best_metric_current, best_metric):
                        self.best_model = best_model_current
                        self.best_metric = best_metric_current
                        self.best_specification = best_specification_current
                    else:
                        self.best_model = best_model
                        self.best_metric = best_metric
                        self.best_specification = best_specification
                    stop = True
                else:
                    best_model = best_model_current
                    best_metric = best_metric_current
                    best_specification = best_specification_current
                    n_subs_layers_construction = n_subs_layers_construction + 1
                # Update progress bar
                best_specification_complete = best_model.__dict__()
                pbar_all.set_postfix(n_subs_layers=best_specification_complete['n_subs_layers'], n_hidden_units=best_specification_complete['n_units'], n_hidden_dense=best_specification_complete['n_neurons_dense_layer'])
                pbar_all.update(1)
        self.best_model, self.best_metric, self.best_specification = best_model, best_metric, best_specification
        return self.best_model, self.best_metric, self.best_specification
