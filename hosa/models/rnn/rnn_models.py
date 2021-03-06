"""
Utilities for Recurrent Neural Network (RNN) models.
"""
import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hosa.callbacks import EarlyStoppingAtMinLoss
from hosa.helpers.functions import metrics_multiclass


class BaseRNN:
    """Base class for Recurrent Neural Network (RNN) models for classification and regression.

    Each RNN model comprises an input layer (an RNN or a bidirectional RNN cell),
    ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer,
    a dense layer, and an output layer. The output layer is a dense layer with ``n_outputs``
    units, with the linear activation function.

    .. warning::
        This class should not be used directly. Use derived classes instead, i.e.,
        :class:`.RNNClassification` or :class:`.RNNRegression`.

    .. note::
        The parameters used in this library were adapted from the exact parameters of the
        TensorFlow library. Descriptions were thus modified accordingly to our approach.
        However, refer to the TensorFlow documentation for more details about each of those
        parameters.

    Args:
        n_outputs (int): Number of class labels in classification, or the number of numerical
            values to predict in regression.
        n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (
            i.e., before the output layer).
        n_units (int): Dimensionality of the output space, i.e., the dimensionality of the
            hidden state.
        n_subs_layers (int): Number of subsequent recurrent layers beteween the input and
            output layers.
        is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build
            the RNN model.
        model_type(str): Type of RNN model to be used. Available options are ``lstm``,
            for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
        optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers
            <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
        dropout_percentage (float): Fraction of the input units to drop.
        activation_function_dense (str): Activation function to use on the penultimate dense
            layer. If not specified, no activation is applied (i.e., uses the linear activation
            function). See `tensorflow.keras.activations
            <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
        kernel_initializer (str): Initializer for the kernel weights matrix, used for the
            linear transformation of the inputs.
        batch_size (int or None): Number of samples per batch of computation. If ``None``,
            ``batch_size`` will default to 32.
        epochs (int): Maximum number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be
            stopped.
        **kwargs: *Ignored*. Extra arguments that are used for compatibility???s sake.
    """

    def __init__(self, n_outputs, n_neurons_dense_layer, n_units, n_subs_layers,
                 is_bidirectional=False,
                 model_type='lstm', optimizer='adam', dropout_percentage=0.1,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patience=5, **kwargs):
        self.n_outputs = n_outputs
        self.n_neurons_dense_layer = n_neurons_dense_layer
        self.n_units = n_units
        self.n_subs_layers = n_subs_layers
        self.is_bidirectional = is_bidirectional
        self.model_type = model_type
        self.optimizer = optimizer
        self.dropout_percentage = dropout_percentage
        self.activation_function_dense = activation_function_dense
        self.kernel_initializer = kernel_initializer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.model = tf.keras.models.Sequential()

    def prepare(self, x, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers``
        subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in
                regression).
        """
        # Choose type of layer based on the model choosen by the user
        if self.model_type == 'lstm':
            layer_type = tf.keras.layers.LSTM
        elif self.model_type == 'gru':
            layer_type = tf.keras.layers.GRU
        else:
            raise ValueError(
                    'Type of RNN model invalid. Available options are ``lstm``, for a Long '
                    'Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.')
        # Input layer (no. of timesteps and no. of features)
        self.model.add(tf.keras.layers.InputLayer(input_shape=x.shape[1:]))
        if self.is_bidirectional:
            self.model.add(tf.keras.layers.Bidirectional(
                    layer_type(self.n_units, return_sequences=self.n_subs_layers > 0)))
        else:
            self.model.add(layer_type(self.n_units, return_sequences=self.n_subs_layers > 0))
        # Subsequent layers
        for n in range(self.n_subs_layers):
            if self.is_bidirectional:
                self.model.add(tf.keras.layers.Bidirectional(
                        layer_type(self.n_units, return_sequences=n < self.n_subs_layers - 1)))
            else:
                self.model.add(
                        layer_type(self.n_units, return_sequences=n < self.n_subs_layers - 1))
        # Dropout layer
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))
        # Dense layer
        self.model.add(tf.keras.layers.Dense(self.n_neurons_dense_layer,
                                             kernel_initializer=self.kernel_initializer,
                                             activation=self.activation_function_dense))

    def aux_fit(self, x, y, callback, validation_size, rtol=1e-03, atol=1e-04, class_weights=None,
                imbalance_correction=None, shuffle=True, **kwargs):
        """
        Auxiliar function for classification and regression models compatibility.

        .. warning::
            This function is not meant to be called by itself. It is just an auxiliary function
            called by the child classes' ``fit`` function.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in
                regression).
            callback (object): Early stopping callback for halting the model's training.
            validation_size (float or int): Proportion of the training dataset that will be used
                the validation split.
            atol (float): Absolute tolerance used for early stopping based on the performance
                metric.
            rtol (float): Relative tolerance used for early stopping based on the performance
                metric.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight
                (float) value, used for weighting the loss function (during training only). **Only
                used for classification problems. Ignored for regression.**
            imbalance_correction (None or bool): Whether to apply correction to class imbalances.
                **Only used for classification problems. Ignored for regression.**
            shuffle (bool): Whether to shuffle the data before splitting.
            **kwargs: Extra arguments used in the TensorFlow's model ``fit`` function. See `here
                <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        x_train, x_validation, y_train, y_validation = train_test_split(x, y,
                                                                        test_size=validation_size,
                                                                        shuffle=shuffle)
        callbacks = [
                callback(self, self.patience, (x_validation, y_validation), imbalance_correction,
                         rtol, atol)]
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(x_validation, y_validation), callbacks=callbacks,
                       class_weight=class_weights, **kwargs)

    @abc.abstractmethod
    def fit(self, x, y, **kwargs):
        """

        Fits the model to data matrix x and target(s) y.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in
                regression).
            **kwargs: Extra arguments explicitly used for regression or classification models.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compile(self):
        """

        Compiles the model for training.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, x, y, **kwargs):
        """

        Computes the performance metric(s) (e.g., accuracy for classification) on the given input
        data and target values.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in
                regression).
            **kwargs: Extra arguments that are explicitly used for regression or classification
                models.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            x (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict``
                function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model
                #predict>`_.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __dict__(self):
        """
        Prepares a dictonary with the parameters of the model.
        """
        raise NotImplementedError


class RNNClassification(BaseRNN):
    """Recurrent Neural Network (RNN) model classifier.

    The model comprises an input layer (an RNN or a bidirectional RNN cell),
    ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer,
    a dense layer, and an output layer.

    Args:
        n_outputs (int): Number of class labels to predict.
        n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (
            i.e., before the output layer).
        n_units (int): Dimensionality of the output space, i.e., the dimensionality of the
            hidden state.
        n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
        is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build
            the RNN model.
        model_type(str): Type of RNN model to be used. Available options are ``lstm``,
            for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
        optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers
            <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
        dropout_percentage (float): Fraction of the input units to drop.
        metrics (list): List of metrics to be evaluated by the model during training and
            testing. Each item of the list can be a string (name of a TensorFlow's built-in
            function), function, or a `tf.keras.metrics.Metric
            <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If
            ``None``, ``metrics`` will default to ``['accuracy']``.
        activation_function_dense (str): Activation function to use on the penultimate dense
            layer. If not specified, no activation is applied (i.e., uses the linear activation
            function). See `tensorflow.keras.activations
            <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
        kernel_initializer (str): Initializer for the kernel weights matrix, used for the
            linear transformation of the inputs.
        batch_size (int or None): Number of samples per batch of computation. If ``None``,
            ``batch_size`` will default to 32.
        epochs (int): Maximum number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be
            stopped.
        **kwargs: *Ignored*. Extra arguments that are used for compatibility???s sake.

    Examples:
        .. code-block:: python
            :linenos:

            from keras.datasets import imdb
            from keras_preprocessing.sequence import pad_sequences
            from tensorflow import keras

            from hosa.models.rnn import RNNClassification
            from hosa.aux import create_overlapping

            # 1 - Load and split the data
            max_sequence_length = 50
            fashion_mnist = keras.datasets.fashion_mnist
            (x_train, y_train), (X_test, y_test) = imdb.load_data(num_words=50)
            # 2 - Prepare the data for rnn input
            x_train = pad_sequences(x_train, maxlen=max_sequence_length, value=0.0)
            X_test = pad_sequences(X_test, maxlen=max_sequence_length, value=0.0)
            x_train, y_train = create_overlapping(x_train, y_train, RNNClassification,
            'central', 3, stride=1, timesteps=2)
            X_test, y_test = create_overlapping(X_test, y_test, RNNClassification, 'central',
            3, stride=1, timesteps=2)
            # 3 - Create and fit the model
            clf = RNNClassification(2, 10, is_bidirectional=True)
            clf.prepare(x_train, y_train)
            clf.compile()
            clf.fit(x_train, y_train)
            # 4 - Calculate predictions
            clf.predict(X_test)
            # 5 - Compute the score
            score = clf.score(X_test, y_test)
    """

    def __init__(self, n_outputs, n_neurons_dense_layer, n_units, n_subs_layers,
                 is_bidirectional=False,
                 model_type='lstm', optimizer='adam', dropout_percentage=0.1, metrics=None,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patience=5, **kwargs):
        if metrics is None:
            metrics = ['accuracy']
        self.metrics, self.is_binary = metrics, None
        super().__init__(n_outputs, n_neurons_dense_layer, n_units, n_subs_layers, is_bidirectional,
                         model_type, optimizer, dropout_percentage, activation_function_dense,
                         kernel_initializer, batch_size, epochs, patience, **kwargs)

    def prepare(self, x, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers``
        subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
        """
        super().prepare(x, y)
        self.model.add(tf.keras.layers.Dense(self.n_outputs, activation='softmax'))

    def fit(self, x, y, validation_size=0.33, shuffle=True, rtol=1e-03, atol=1e-04,
            class_weights=None, imbalance_correction=False, **kwargs):
        """

        Fits the model to data matrix x and target(s) y.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            class_weights (dict): Dictionary mapping class indices (integers) to a weight (float)
                value, used for weighting the loss function (during training only).
            validation_size (float or int): Proportion of the train dataset to include in the
                validation split.
            shuffle (bool): Whether to shuffle the data before splitting.
            atol (float): Absolute tolerance used for early stopping based on the performance
                metric.
            rtol (float): Relative tolerance used for early stopping based on the performance
                metric.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight
                (float) value, used for weighting the loss function (during training only).
                imbalance_correction (bool): `True` if correction for imbalance should be applied to
                the metrics; `False` otherwise.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function.
                See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(x, y, callback, validation_size, rtol=rtol, atol=atol,
                        class_weights=class_weights, imbalance_correction=imbalance_correction,
                        shuffle=shuffle, **kwargs)
        return self.model

    def score(self, x, y, imbalance_correction=False):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            imbalance_correction (bool): `True` if correction for imbalance should be applied to
                the metrics; `False` otherwise.

        Returns:
            tuple: Returns a tuple containing the area under the ROC curve (AUC), accuracy,
            sensitivity, and sensitivity.

        .. note::
            This function can be used for both binary and multiclass classification.
        """
        y_probs, _ = self.predict(x)
        auc_value, accuracy, sensitivity, specificity = metrics_multiclass(y, y_probs,
                                                                           self.n_outputs,
                                                                           imbalance_correction=imbalance_correction)
        return auc_value, accuracy, sensitivity, specificity

    def predict(self, x, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            x (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict``
                function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model
                #predict>`_.

        Returns:
            tuple: Returns a tuple containing the probability estimates and predicted classes.
        """
        y_probs = self.model.predict(x, **kwargs)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

    def compile(self):
        """

        Compiles the model for training.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer,
                           metrics=self.metrics)
        return self.model

    def __dict__(self):
        """
        Prepares a dictonary with the parameters of the model.

        Returns:
            dict: Dictonary with the parameter names mapped to their values.
        """
        parameters = {'n_outputs':                 self.n_outputs,
                      'n_neurons_dense_layer':     self.n_neurons_dense_layer,
                      'n_units':                   self.n_units,
                      'n_subs_layers':             self.n_subs_layers,
                      'is_bidirectional':          self.is_bidirectional,
                      'model_type':                self.model_type,
                      'optimizer':                 self.optimizer,
                      'dropout_percentage':        self.dropout_percentage,
                      'metrics':                   self.metrics,
                      'activation_function_dense': self.activation_function_dense,
                      'kernel_initializer':        self.kernel_initializer,
                      'batch_size':                self.batch_size,
                      'epochs':                    self.epochs,
                      'patience':                  self.patience}

        return parameters


class RNNRegression(BaseRNN):
    """Recurrent Neural Network (RNN) model regressor.

    The model comprises an input layer (an RNN or a bidirectional RNN cell),
    ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer,
    a dense layer, and an output layer.

    Args:
        n_outputs (int): Number of numerical values to predict in regression.
        n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (
            i.e., before the output layer).
        n_units (int): Dimensionality of the output space, i.e., the dimensionality of the
            hidden state.
        n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
        is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build
            the RNN model.
        model_type(str): Type of RNN model to be used. Available options are ``lstm``,
            for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
        optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers
            <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
        dropout_percentage (float): Fraction of the input units to drop.
        metrics (list): List of metrics to be evaluated by the model during training and
            testing. Each item of the list can be a string (name of a TensorFlow's built-in
            function), function, or a `tf.keras.metrics.Metric
            <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If
            ``None``, ``metrics`` will default to ``['mean_squared_error']``.
        activation_function_dense (str): Activation function to use on the penultimate dense
            layer. If not specified, no activation is applied (i.e., uses the linear activation
            function). See `tensorflow.keras.activations
            <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
        kernel_initializer (str): Initializer for the kernel weights matrix, used for the
            linear transformation of the inputs.
        batch_size (int or None): Number of samples per batch of computation. If ``None``,
            ``batch_size`` will default to 32.
        epochs (int): Maximum number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be
            stopped.
        **kwargs: *Ignored*. Extra arguments that are used for compatibility???s sake.

    Examples:
        .. code-block:: python
            :linenos:

            import pandas as pd

            from hosa.models.rnn import RNNRegression
            from hosa.aux import create_overlapping

            # 1 - Download, load, and split the data
            dataset = pd.read_csv(
            'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers
            .csv', header=0, index_col=0)
            x = dataset.Passengers.to_numpy().reshape((len(dataset), 1))
            y = dataset.Passengers.to_numpy()
            x_train, y_train = x[:100], y[:100]
            X_test, y_test = x[100:], y[100:]
            # 2 - Prepare the data for cnn input
            x_train, y_train = create_overlapping(x_train, y_train, RNNRegression, 'central',
            10, timesteps=1)
            X_test, y_test = create_overlapping(X_test, y_test, RNNRegression, 'central', 10,
            timesteps=1)
            # 3 - Create and fit the model
            clf = RNNRegression(1, 200, epochs=500, patience=500)
            clf.prepare(x_train, y_train)
            clf.compile()
            clf.fit(x_train, y_train)
            # 4 - Calculate predictions
            clf.predict(X_test)
            # 5 - Compute the score
            score = clf.score(X_test, y_test)

    """

    def __init__(self, n_outputs, n_neurons_dense_layer, n_units, n_subs_layers,
                 is_bidirectional=False, model_type='lstm', optimizer='adam',
                 dropout_percentage=0.1, metrics=None, activation_function_dense='relu',
                 kernel_initializer='normal', batch_size=1000, epochs=50, patience=5, **kwargs):
        if metrics is None:
            metrics = ['mean_squared_error']
        self.metrics = metrics
        super().__init__(n_outputs, n_neurons_dense_layer, n_units, n_subs_layers, is_bidirectional,
                         model_type, optimizer, dropout_percentage, activation_function_dense,
                         kernel_initializer, batch_size, epochs, patience, **kwargs)

    def prepare(self, x, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers``
        subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
        """
        super().prepare(x, y)
        self.model.add(tf.keras.layers.Dense(self.n_outputs, activation='linear'))

    def fit(self, x, y, validation_size=0.33, atol=1e-04, rtol=1e-03, shuffle=True, **kwargs):
        """

        Fits the model to data matrix x and target(s) y.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
            validation_size (float or int): Proportion of the train dataset to include in the
                validation split.
            atol (float): Absolute tolerance used for early stopping based on the performance
                metric.
            rtol (float): Relative tolerance used for early stopping based on the performance
                metric.
            shuffle (bool): Whether to shuffle the data before splitting.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function.
                See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(x, y, callback, validation_size, atol=atol, rtol=rtol, class_weights=None,
                        imbalance_correction=None, shuffle=shuffle, **kwargs)

    def score(self, x, y, **kwargs):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
            **kwargs: *Ignored*. Only included here for compatibility???s sake.

        Returns:
            tuple: Returns a tuple containing the mean squared error (MSE) and coefficient of
            determination (:math:`R^2`).

        """
        y_pred = self.predict(x)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        return mse, r2

    def predict(self, x, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            x (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict``
                function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model
                #predict>`_.

        Returns:
            numpy.ndarray: Returns an array containing the estimates.
        """
        y_pred = self.model.predict(x, **kwargs)
        return y_pred

    def compile(self):
        """

        Compiles the model for training.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer,
                           metrics=self.metrics)

    def __dict__(self):
        """
        Prepares a dictonary with the parameters of the model.

        Returns:
            dict: Dictonary with the parameter names mapped to their values.
        """
        parameters = {'n_outputs':                 self.n_outputs,
                      'n_neurons_dense_layer':     self.n_neurons_dense_layer,
                      'n_units':                   self.n_units,
                      'n_subs_layers':             self.n_subs_layers,
                      'is_bidirectional':          self.is_bidirectional,
                      'model_type':                self.model_type,
                      'optimizer':                 self.optimizer,
                      'dropout_percentage':        self.dropout_percentage,
                      'metrics':                   self.metrics,
                      'activation_function_dense': self.activation_function_dense,
                      'kernel_initializer':        self.kernel_initializer,
                      'batch_size':                self.batch_size,
                      'epochs':                    self.epochs,
                      'patience':                  self.patience}
        return parameters
