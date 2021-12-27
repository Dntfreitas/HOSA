import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from project.Callbacks import EarlyStoppingAtMinLoss
from project.aux import metrics_multiclass


class BaseLSTM:
    def __init__(self, is_bidirectional, n_units, n_subs_layers,
                 n_neurons_last_dense_layer, optimizer='adam', dropout_percentage=0.1,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patientece=5, verbose=1):

        """Base class for Long Short-Term Memory (LSTM) models for classification and regression.

        Each LSTM model comprises an input layer (an LSTM or a bidirectional LSTM cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer.

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.LSTMClassification` or :class:`.LSTMRegression`.

        Args:
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the LSTM model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            n_neurons_last_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            optimizer (str): Name of optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            activation_function_dense (str): Activation function to use on the last dense layer. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Number of epochs to train the model.
            patientece (int): Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. Available options are ``0``, for silent mode, or ``1``, for a progress bar.

        .. note::
            The parameters used in this library were adapted from the same parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.

        """
        self.is_bidirectional, self.n_units, self.n_subs_layers, self.n_neurons_last_dense_layer, self.optimizer, self.dropout_percentage, self.activation_function_dense, self.kernel_initializer, self.batch_size, self.epochs, self.patientece, self.verbose = is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patientece, verbose
        self.model = tf.keras.models.Sequential()

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GofLayers, and flatten and dense layers.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        """
        # Input layer
        n_features = X.shape[-1]
        self.model.add(tf.keras.layers.InputLayer(input_shape=(n_features, 1)))
        if self.is_bidirectional:
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units, return_sequences=self.n_subs_layers > 0)))
        else:
            self.model.add(tf.keras.layers.LSTM(self.n_units, return_sequences=self.n_subs_layers > 0))
        # Subsequent layers
        for n in range(self.n_subs_layers):
            if self.is_bidirectional:
                self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.n_units, return_sequences=n < self.n_subs_layers - 1)))
            else:
                self.model.add(tf.keras.layers.LSTM(self.n_units, return_sequences=n < self.n_subs_layers - 1))
        # Dropout layer
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))
        # Dense layer
        self.model.add(tf.keras.layers.Dense(self.n_neurons_last_dense_layer, kernel_initializer=self.kernel_initializer, activation=self.activation_function_dense))

    def aux_fit(self, X, y, callback, class_weights, validation_size, **kwargs):
        """
        Auxiliar function for classification and regression models compatibility.

        .. warning::
            This function is not meant to be called by itself. It is just an auxiliary function called by the child classes' ``fit`` function.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            callback (EarlyStoppingAtMinLoss): Early stopping callback for halting the model's training.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            validation_size (float or int): Proportion of the train dataset to include in the validation split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)
        callbacks = [callback(self, self.patientece, (X_validation, y_validation))]
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_validation, y_validation), callbacks=callbacks, class_weight=class_weights, verbose=self.verbose, **kwargs)

    @abc.abstractmethod
    def compile(self, **kwargs):
        """

        Compiles the model for training.

        Args:
            **kwargs: Extra arguments that are used in the TensorFlow's model ``compile`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.
        """
        raise NotImplemented

    @abc.abstractmethod
    def score(self, X, y):
        """

        Computes the performance metric(s) (e.g., accuracy) on the given input data and target values.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        """
        raise NotImplemented

    @abc.abstractmethod
    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.
        """
        raise NotImplemented


class LSTMClassification(BaseLSTM):
    def __init__(self, number_classes, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer,
                 optimizer='adam', dropout_percentage=0.1, metrics=None, activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patientece=5,
                 verbose=1):
        """Long Short-Term Memory (LSTM) model classifier.

        The model comprises an input layer (an LSTM or a bidirectional LSTM cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer.

        Args:
            number_classes (int): Number of classes (or labels) of the classification problem.
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the LSTM model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            n_neurons_last_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            optimizer (str): Name of optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a built-in function), function or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. By default, ``metrics=['accuracy']``.
            activation_function_dense (str): Activation function to use on the last dense layer. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Number of epochs to train the model.
            patientece (int): Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. Available options are ``0``, for silent mode, or ``1``, for a progress bar.
        """
        if metrics is None:
            metrics = ['accuracy']
        self.metrics, self.number_classes, self.is_binary = metrics, number_classes, None
        super().__init__(is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patientece, verbose)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GofLayers, and flatten and dense layers.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).

        Returns:
            tensorflow.keras.Sequential: Returns an untrained TensorFlow model.
        """
        super().prepare(X, y)
        self.model.add(tf.keras.layers.Dense(self.number_classes, activation='softmax'))
        return self.model

    def fit(self, X, y, class_weights=None, validation_size=0.33, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            class_weights (dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            validation_size (float or int): Proportion of the train dataset to include in the validation split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, class_weights, validation_size, **kwargs)
        return self.model

    def score(self, X, y):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            X: Input data.
            y: Target values (i.e., class labels).

        Returns:
            list: List containing the area under the ROC curve (AUC), accuracy, sensitivity and sensitivity.

        .. note::
            This function can be used for both binary and multiclass classification.
        """
        y_probs, y_pred = self.predict(X)
        auc_value, accuracy, sensitivity, specificity = metrics_multiclass(y, y_probs, self.number_classes)
        return auc_value, accuracy, sensitivity, sensitivity

    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.
        """
        # TODO: check why we don't need  `X = np.expand_dims(X, axis=-1)`
        y_probs = self.model.predict(X, **kwargs)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

    def compile(self, **kwargs):
        """

        Compiles the model for training.

        Args:
            **kwargs: Extra arguments that are used in the TensorFlow's model ``compile`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=self.metrics, **kwargs)
        return self.model


class LSTMRegression(BaseLSTM):
    def __init__(self, number_outputs, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer,
                 optimizer='adam', dropout_percentage=0.1, metrics=None, activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patientece=5,
                 verbose=1):
        """Long Short-Term Memory (LSTM) model regressor.

        The model comprises an input layer (an LSTM or a bidirectional LSTM cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer.

        Args:
            number_outputs (int): Dimension of the target output vector.
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the LSTM model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            n_neurons_last_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            optimizer (str): Name of optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a built-in function), function or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. By default, ``metrics = ['mean_squared_error']``.
            activation_function_dense (str): Activation function to use on the last dense layer. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Number of epochs to train the model.
            patientece (int): Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. Available options are ``0``, for silent mode, or ``1``, for a progress bar.
        """
        if metrics is None:
            metrics = ['mean_squared_error']
        self.number_outputs, self.metrics = number_outputs, metrics
        super().__init__(is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patientece, verbose)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GofLayers, and flatten and dense layers.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers)

        Returns:
            tensorflow.keras.Sequential: Returns an untrained TensorFlow model.

        """
        super().prepare(X, y)
        self.model.add(tf.keras.layers.Dense(self.number_outputs, activation='linear'))
        return self.model

    def fit(self, X, y, validation_size=0.33, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            validation_size (float or int): Proportion of the train dataset to include in the validation split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, None, validation_size, **kwargs)

    def score(self, X, y):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            X: Input data.
            y: Target values (i.e., class labels).

        Returns:
            list: List containing the mean squared error (MSE) and coefficient of determination (:math:`R^2`).

        """
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        return mse, r2

    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.
        """
        y_pred = self.model.predict(X, **kwargs)
        return y_pred

    def compile(self, **kwargs):
        """

        Compiles the model for training.

        Args:
            **kwargs: Extra arguments that are used in the TensorFlow's model ``compile`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=self.metrics, **kwargs)
