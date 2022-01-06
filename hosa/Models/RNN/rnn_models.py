import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hosa.Callbacks import EarlyStoppingAtMinLoss
from hosa.aux import metrics_multiclass


class BaseRNN:
    def __init__(self, n_outputs, n_neurons_dense_layer, is_bidirectional=False, n_units=10, n_subs_layers=2,
                 model_type='lstm', optimizer='adam', dropout_percentage=0.1,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patience=5, **kwargs):

        """Base class for Recurrent Neural Network (RNN) models for classification and regression.

        Each RNN model comprises an input layer (an RNN or a bidirectional RNN cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer. The output layer is a dense layer with ``n_outputs`` units, with the linear activation function.

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.RNNClassification` or :class:`.RNNRegression`.

        .. note::
            The parameters used in this library were adapted from the exact parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.

        Args:
            n_outputs (int): Number of class labels in classification, or the number of numerical values to predict in regression.
            n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the RNN model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            model_type(str): Type of RNN model to be used. Available options are ``lstm``, for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            activation_function_dense (str): Activation function to use on the penultimate dense layer. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.
        """
        self.n_outputs, self.n_neurons_dense_layer, self.is_bidirectional, self.n_units, self.n_subs_layers, self.model_type, self.optimizer, self.dropout_percentage, self.activation_function_dense, self.kernel_initializer, self.batch_size, self.epochs, self.patience = n_outputs, n_neurons_dense_layer, is_bidirectional, n_units, n_subs_layers, model_type, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patience
        self.model = tf.keras.models.Sequential()

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers`` subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        """
        # Choose type of layer based on the model choosen by the user
        if self.model_type == 'lstm':
            layer_type = tf.keras.layers.LSTM
        elif self.model_type == 'gru':
            layer_type = tf.keras.layers.GRU
        else:
            raise ValueError('Type of RNN model invalid. Available options are ``lstm``, for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.')
        # Input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape=X.shape[1:]))
        if self.is_bidirectional:
            self.model.add(tf.keras.layers.Bidirectional(layer_type(self.n_units, return_sequences=self.n_subs_layers > 0)))
        else:
            self.model.add(layer_type(self.n_units, return_sequences=self.n_subs_layers > 0))
        # Subsequent layers
        for n in range(self.n_subs_layers):
            if self.is_bidirectional:
                self.model.add(tf.keras.layers.Bidirectional(layer_type(self.n_units, return_sequences=n < self.n_subs_layers - 1)))
            else:
                self.model.add(layer_type(self.n_units, return_sequences=n < self.n_subs_layers - 1))
        # Dropout layer
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))
        # Dense layer
        self.model.add(tf.keras.layers.Dense(self.n_neurons_dense_layer, kernel_initializer=self.kernel_initializer, activation=self.activation_function_dense))

    def aux_fit(self, X, y, callback, validation_size, rtol=1e-03, atol=1e-04, class_weights=None, inbalance_correction=None, **kwargs):
        """
        Auxiliar function for classification and regression models compatibility.

        .. warning::
            This function is not meant to be called by itself. It is just an auxiliary function called by the child classes' ``fit`` function.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            callback (object): Early stopping callback for halting the model's training.
            validation_size (float or int): Proportion of the training dataset that will be used the validation split.
            atol (float): Absolute tolerance used for early stopping based on the performance metric.
            rtol (float): Relative tolerance used for early stopping based on the performance metric.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). **Only used for classification problems. Ignored for regression.**
            inbalance_correction (None or bool): Whether to apply correction to class imbalances. **Only used for classification problems. Ignored for regression.**
            **kwargs: Extra arguments used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)
        callbacks = [callback(self, self.patience, (X_validation, y_validation), inbalance_correction, rtol, atol)]
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_validation, y_validation), callbacks=callbacks, class_weight=class_weights, **kwargs)

    @abc.abstractmethod
    def fit(self, X, y, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            **kwargs: Extra arguments explicitly used for regression or classification models.
        """
        raise NotImplemented

    @abc.abstractmethod
    def compile(self):
        """

        Compiles the model for training.

        """
        raise NotImplemented

    @abc.abstractmethod
    def score(self, X, y, **kwargs):
        """

        Computes the performance metric(s) (e.g., accuracy for classification) on the given input data and target values.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            **kwargs: Extra arguments that are explicitly used for regression or classification models.
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


class RNNClassification(BaseRNN):
    def __init__(self, n_outputs, n_neurons_dense_layer, is_bidirectional=False, n_units=10, n_subs_layers=2,
                 model_type='lstm', optimizer='adam', dropout_percentage=0.1, metrics=None,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patience=5, **kwargs):
        """Recurrent Neural Network (RNN) model classifier.

        The model comprises an input layer (an RNN or a bidirectional RNN cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer.

        Args:
            n_outputs (int): Number of class labels to predict.
            n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the RNN model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            model_type(str): Type of RNN model to be used. Available options are ``lstm``, for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a TensorFlow's built-in function), function, or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If ``None``, ``metrics`` will default to ``['accuracy']``.
            activation_function_dense (str): Activation function to use on the penultimate dense layer. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.

        Examples:
            .. code-block:: python
                :linenos:

                from keras.datasets import imdb
                from keras_preprocessing.sequence import pad_sequences
                from tensorflow import keras

                from hosa.Models.RNN import RNNClassification
                from hosa.aux import create_overlapping

                # 1 - Load and split the data
                max_sequence_length = 50
                fashion_mnist = keras.datasets.fashion_mnist
                (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=50)
                # 2 - Prepare the data for RNN input
                X_train = pad_sequences(X_train, maxlen=max_sequence_length, value=0.0)
                X_test = pad_sequences(X_test, maxlen=max_sequence_length, value=0.0)
                X_train, y_train = create_overlapping(X_train, y_train, RNNClassification, 'central', 3, stride=1, timesteps=2)
                X_test, y_test = create_overlapping(X_test, y_test, RNNClassification, 'central', 3, stride=1, timesteps=2)
                # 3 - Create and fit the model
                clf = RNNClassification(2, 10, is_bidirectional=True)
                clf.prepare(X_train, y_train)
                clf.compile()
                clf.fit(X_train, y_train)
                # 4 - Calculate predictions
                clf.predict(X_test)
                # 5 - Compute the score
                score = clf.score(X_test, y_test)
        """
        if metrics is None:
            metrics = ['accuracy']
        self.metrics, self.is_binary = metrics, None
        super().__init__(n_outputs, n_neurons_dense_layer, is_bidirectional, n_units, n_subs_layers, model_type, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patience, **kwargs)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers`` subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
        """
        super().prepare(X, y)
        self.model.add(tf.keras.layers.Dense(self.n_outputs, activation='softmax'))

    def fit(self, X, y, validation_size=0.33, rtol=1e-03, atol=1e-04, class_weights=None, inbalance_correction=False, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            class_weights (dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            validation_size (float or int): Proportion of the train dataset to include in the validation split.
            atol (float): Absolute tolerance used for early stopping based on the performance metric.
            rtol (float): Relative tolerance used for early stopping based on the performance metric.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            inbalance_correction (bool): Whether to apply correction to class imbalances.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, validation_size, rtol, atol, class_weights, inbalance_correction, **kwargs)
        return self.model

    def score(self, X, y, inbalance_correction=False):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            inbalance_correction (bool): Whether to apply correction to class imbalances.

        Returns:
            tuple: Returns a list containing the area under the ROC curve (AUC), accuracy, sensitivity, and sensitivity.

        .. note::
            This function can be used for both binary and multiclass classification.
        """
        y_probs, y_pred = self.predict(X)
        auc_value, accuracy, sensitivity, specificity = metrics_multiclass(y, y_probs, self.n_outputs, inbalance_correction=inbalance_correction)
        return auc_value, accuracy, sensitivity, sensitivity

    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.

        Returns:
            tuple: Returns a list containing the probability estimates and predicted classes.
        """
        y_probs = self.model.predict(X, **kwargs)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

    def compile(self):
        """

        Compiles the model for training.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=self.metrics)
        return self.model


class RNNRegression(BaseRNN):
    def __init__(self, n_outputs, n_neurons_dense_layer, is_bidirectional=False, n_units=10, n_subs_layers=2,
                 model_type='lstm', optimizer='adam', dropout_percentage=0.1, metrics=None,
                 activation_function_dense='relu', kernel_initializer='normal',
                 batch_size=1000, epochs=50, patience=5, **kwargs):
        """Recurrent Neural Network (RNN) model regressor.

        The model comprises an input layer (an RNN or a bidirectional RNN cell), ``n_subs_layers`` subsequent layers (similar to the input cell), a dropout layer, a dense layer, and an output layer.

        Args:
            n_outputs (int): Number of numerical values to predict in regression.
            n_neurons_dense_layer (int): Number of neurons units of the penultimate dense layer (i.e., before the output layer).
            is_bidirectional (bool): If ``true``, then bidirectional layers will be used to build the RNN model.
            n_units (int): Dimensionality of the output space, i.e., the dimensionality of the hidden state.
            n_subs_layers (int): Number of subsequent layers beteween the input and output layers.
            model_type(str): Type of RNN model to be used. Available options are ``lstm``, for a Long Short-Term Memory model, or ``gru``, for a Gated Recurrent Unit model.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            dropout_percentage (float): Fraction of the input units to drop.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a TensorFlow's built-in function), function, or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If ``None``, ``metrics`` will default to ``['mean_squared_error']``.
            activation_function_dense (str): Activation function to use on the penultimate dense layer. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            kernel_initializer (str): Initializer for the kernel weights matrix, used for the linear transformation of the inputs.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.

        Examples:
            .. code-block:: python
                :linenos:

                import pandas as pd

                from hosa.Models.RNN import RNNRegression
                from hosa.aux import create_overlapping

                # 1 - Download, load and split the data
                dataset = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', header=0, index_col=0)
                X = dataset.Passengers.to_numpy().reshape((len(dataset), 1))
                y = dataset.Passengers.to_numpy()
                X_train, y_train = X[:100], y[:100]
                X_test, y_test = X[100:], y[100:]
                # 2 - Prepare the data for CNN input
                X_train, y_train = create_overlapping(X_train, y_train, RNNRegression, 'central', 10, timesteps=1)
                X_test, y_test = create_overlapping(X_test, y_test, RNNRegression, 'central', 10, timesteps=1)
                # 3 - Create and fit the model
                clf = RNNRegression(1, 200, epochs=500, patience=500)
                clf.prepare(X_train, y_train)
                clf.compile()
                clf.fit(X_train, y_train)
                # 4 - Calculate predictions
                clf.predict(X_test)
                # 5 - Compute the score
                score = clf.score(X_test, y_test)

        """
        if metrics is None:
            metrics = ['mean_squared_error']
        self.metrics = metrics
        super().__init__(n_outputs, n_neurons_dense_layer, is_bidirectional, n_units, n_subs_layers, model_type, optimizer, dropout_percentage, activation_function_dense, kernel_initializer, batch_size, epochs, patience, **kwargs)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, ``n_subs_layers`` subsequent layers, a dropout layer, a dense layer, and an output layer.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
        """
        super().prepare(X, y)
        self.model.add(tf.keras.layers.Dense(self.n_outputs, activation='linear'))

    def fit(self, X, y, validation_size=0.33, rtol=1e-03, atol=1e-04, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
            validation_size (float or int): Proportion of the train dataset to include in the validation split.
            atol (float): Absolute tolerance used for early stopping based on the performance metric.
            rtol (float): Relative tolerance used for early stopping based on the performance metric.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, validation_size, rtol, atol, class_weights=None, inbalance_correction=None, **kwargs)

    def score(self, X, y, **kwargs):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., real numbers).
            **kwargs: *Ignored*. Only included here for compatibility with :class:`.CNNClassification`.

        Returns:
            tuple: List containing the mean squared error (MSE) and coefficient of determination (:math:`R^2`).

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

        Returns:
            list: Returns a list containing the estimates.
        """
        y_pred = self.model.predict(X, **kwargs)
        return y_pred

    def compile(self):
        """

        Compiles the model for training.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=self.metrics)
