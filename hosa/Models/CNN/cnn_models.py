import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from hosa.Callbacks import EarlyStoppingAtMinLoss
from hosa.aux import metrics_multiclass


class BaseCNN:
    def __init__(self, n_outputs, n_neurons_dense_layer, n_kernels,
                 optimizer='adam', cnn_dim=1, kernel_size=2, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid', dropout_percentage=0.1,
                 activation_function_gol='relu', activation_function_dense='relu', batch_size=1000, epochs=50, patience=5,
                 **kwargs):

        """Base class for Convolutional Neural Network (CNN) models for classification and regression.

        Each CNN model comprises an input layer, a set of groups of layers (GolLayers [Men20]_)—where each group is composed of one convolution layer, followed by one pooling layer and a dropout layer—, a dense layer, and an output layer. The output layer is a dense layer with ``n_outputs`` units, and with the softmax activation function (for classification) or the linear activation function (for regression).

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.CNNClassification` or :class:`.CNNRegression`.

        .. note::
            The parameters used in this library were adapted from the exact parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.

        Args:
            n_outputs (int): Number of class labels in classification, or the number of numerical values to predict in regression.
            n_neurons_dense_layer (int): Number of neuron units in the dense layer (i.e., the dense layer after the set of groups of layers).
            n_kernels (list): *i*-th element represents the number of output filters of the convolution layer in the *i*-th GolLayer.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            cnn_dim (int): Number of dimensions applicable to *all* the convolution layers of the GolLayers.
            kernel_size (int or tuple): Integer or tuple/list of integers, specifying the length (for ``cnn_dim`` = 1), the height and width (for ``cnn_dim`` = 2), or  the depth, height and width (for ``cnn_dim`` = 3) of the convolution window. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            pool_size (int or tuple): Size of the max-pooling window applicable to *all* the max-pooling layers of the GolLayers. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            strides_convolution (int or tuple): Integer or tuple/list of integers, specifying the strides of the convolution. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            strides_pooling (int or tuple): Integer or tuple/list of integers, specifying the strides of the pooling. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the pooling layers of the GolLayers.
            padding (str): Available options are ``valid`` or ``same``. ``valid`` means no padding. ``same`` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. This applies to *all* the pooling layers of the GolLayers.
            dropout_percentage (float): Fraction of the input units to drop. This applies to *all* the dropout layers of the GolLayers.
            activation_function_gol (str or None): Activation function to use in the convolution layers of the GolLayers. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_. This applies to *all* the convolution layers of the GolLayers.
            activation_function_dense (str or None): Activation function to use on the dense layer (i.e., the dense layer after the set of groups of layers). If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.

        References:
            .. [Men20] Mendonça, F.; Mostafa, S.; Morgado-Dias, F.; Juliá-Serdá, G.; Ravelo-García, A. A Method for Sleep Quality Analysis Based on CNN Ensemble With Implementation in a Portable Wireless Device. *IEEE Access* **2020**, 8, 158523–158537.
       """

        self.n_outputs, self.n_neurons_dense_layer, self.n_kernels, self.optimizer, self.cnn_dim, self.kernel_size, self.pool_size, self.strides_convolution, self.strides_pooling, self.padding, self.dropout_percentage, self.activation_function_gol, self.activation_function_dense, self.batch_size, self.epochs, self.patience = n_outputs, n_neurons_dense_layer, n_kernels, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, dropout_percentage, activation_function_gol, activation_function_dense, batch_size, epochs, patience
        self.model = tf.keras.models.Sequential()

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GolLayers, flatten and dense layers.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        """
        if self.cnn_dim == 1:
            input_shape = (X.shape[-1], 1)
        elif self.cnn_dim == 2:
            input_shape = (X.shape[-2], X.shape[-1], 1)
        elif self.cnn_dim == 3:
            input_shape = (X.shape[-3], X.shape[-2], X.shape[-1], 1)
        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for n_kernel in self.n_kernels:
            self.add_gol(n_kernel, self.cnn_dim)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_neurons_dense_layer, activation=self.activation_function_dense))

    def add_gol(self, n_kernel, cnn_dim):
        """

        Adds a GolLayer to the estimator.

        Args:
            n_kernel (int): Number of output filters.
            cnn_dim (int): Number of dimensions.

        Raises:
            ValueError: If ``cnn_dim`` is not valid.
        """
        if cnn_dim == 1:
            self.model.add(tf.keras.layers.Conv1D(n_kernel, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        elif cnn_dim == 2:
            self.model.add(tf.keras.layers.Conv2D(n_kernel, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        elif cnn_dim == 3:
            self.model.add(tf.keras.layers.Conv3D(n_kernel, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling3D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        else:
            raise ValueError('`cnn_dim` parameter must be 1, 2 or 3.')
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))

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
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)
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


class CNNClassification(BaseCNN):
    def __init__(self, n_outputs, n_neurons_dense_layer, n_kernels,
                 optimizer='adam', metrics=None, cnn_dim=1, kernel_size=2, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid', dropout_percentage=0.1,
                 activation_function_gol='relu', activation_function_dense='relu', batch_size=1000, epochs=50, patience=5,
                 **kwargs):
        """Convolutional Neural Network (CNN) classifier.

        The model comprises an input layer, a set of groups of layers (GolLayers [Men20]_)—where each group is composed of one convolution layer, followed by one pooling layer and a dropout layer—, a dense layer, and an output layer. The output layer is a dense layer with ``n_outputs`` units, with the softmax activation function.

        .. note::
            The parameters used in this library were adapted from the exact parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.

        Args:
            n_outputs (int): Number of class labels to predict.
            n_neurons_dense_layer (int): Number of neuron units in the dense layer (i.e., the dense layer after the set of groups of layers).
            n_kernels (list): *i*-th element represents the number of output filters of the convolution layer in the *i*-th GolLayer.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a TensorFlow's built-in function), function, or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If ``None``, ``metrics`` will default to ``['accuracy']``.
            cnn_dim (int): Number of dimensions applicable to *all* the convolution layers of the GolLayers.
            kernel_size (int or tuple): Integer or tuple/list of integers, specifying the length (for ``cnn_dim`` = 1), the height and width (for ``cnn_dim`` = 2), or  the depth, height and width (for ``cnn_dim`` = 3) of the convolution window. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            pool_size (int or tuple): Size of the max-pooling window applicable to *all* the max-pooling layers of the GolLayers. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            strides_convolution (int or tuple): Integer or tuple/list of integers, specifying the strides of the convolution. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            strides_pooling (int or tuple): Integer or tuple/list of integers, specifying the strides of the pooling. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the pooling layers of the GolLayers.
            padding (str): Available options are ``valid`` or ``same``. ``valid`` means no padding. ``same`` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. This applies to *all* the pooling layers of the GolLayers.
            dropout_percentage (float): Fraction of the input units to drop. This applies to *all* the dropout layers of the GolLayers.
            activation_function_gol (str or None): Activation function to use in the convolution layers of the GolLayers. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_. This applies to *all* the convolution layers of the GolLayers.
            activation_function_dense (str or None): Activation function to use on the dense layer (i.e., the dense layer after the set of groups of layers). If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.

        References:
            .. [Men20] Mendonça, F.; Mostafa, S.; Morgado-Dias, F.; Juliá-Serdá, G.; Ravelo-García, A. A Method for Sleep Quality Analysis Based on CNN Ensemble With Implementation in a Portable Wireless Device. *IEEE Access* **2020**, 8, 158523–158537.

        Examples:
            .. code-block:: python
                :linenos:

                from tensorflow import keras

                from hosa.Models.CNN import CNNClassification

                # 1 - Load and split the data
                fashion_mnist = keras.datasets.fashion_mnist
                (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
                # 2 - Normalize the images, and reshape the images for CNN input
                X_train = X_train / 255.0
                X_test = X_test / 255.0
                X_train = X_train.reshape((-1, 28 * 28))
                X_test = X_test.reshape((-1, 28 * 28))
                # 3 - Create and fit the model
                clf = CNNClassification(10, 10, [4, 3])
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
        self.metrics, self.n_outputs, self.is_binary = metrics, n_outputs, None
        super().__init__(n_outputs, n_neurons_dense_layer, n_kernels, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, dropout_percentage, activation_function_gol, activation_function_dense, batch_size, epochs, patience, **kwargs)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GolLayers, flatten and dense layers.

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
            tuple: Returns a tuple containing the area under the ROC curve (AUC), accuracy, sensitivity, and sensitivity.

        .. note::
            This function can be used for both binary and multiclass classification.
        """
        y_probs, y_pred = self.predict(X)
        auc_value, accuracy, sensitivity, specificity = metrics_multiclass(y, y_probs, self.n_outputs, inbalance_correction=inbalance_correction)
        return auc_value, accuracy, sensitivity, specificity

    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.

       Returns:
            tuple: Returns a tuple containing the probability estimates and predicted classes.
       """
        X = np.expand_dims(X, axis=-1)
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


class CNNRegression(BaseCNN):

    def __init__(self, n_outputs, n_neurons_dense_layer, n_kernels,
                 optimizer='adam', metrics=None, cnn_dim=1, kernel_size=2, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid', dropout_percentage=0.1,
                 activation_function_gol='relu', activation_function_dense='relu', batch_size=1000, epochs=50, patience=5,
                 **kwargs):
        """Convolutional Neural Network (CNN) regressor.

        The model comprises an input layer, a set of groups of layers (GolLayers [Men20]_)—where each group is composed of one convolution layer, followed by one pooling layer and a dropout layer—, a dense layer, and an output layer. The output layer is a dense layer with ``n_outputs`` units, with the linear activation function.

        .. note::
            The parameters used in this library were adapted from the exact parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.

        Args:
            n_outputs (int): Number of numerical values to predict in regression.
            n_neurons_dense_layer (int): Number of neuron units in the dense layer (i.e., the dense layer after the set of groups of layers).
            n_kernels (list): *i*-th element represents the number of output filters of the convolution layer in the *i*-th GolLayer.
            optimizer (str): Name of the optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            metrics (list): List of metrics to be evaluated by the model during training and testing. Each item of the list can be a string (name of a TensorFlow's built-in function), function, or a `tf.keras.metrics.Metric <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric>`_ instance. If ``None``, ``metrics`` will default to ``['mean_squared_error']``.
            cnn_dim (int): Number of dimensions applicable to *all* the convolution layers of the GolLayers.
            kernel_size (int or tuple): Integer or tuple/list of integers, specifying the length (for ``cnn_dim`` = 1), the height and width (for ``cnn_dim`` = 2), or  the depth, height and width (for ``cnn_dim`` = 3) of the convolution window. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            pool_size (int or tuple): Size of the max-pooling window applicable to *all* the max-pooling layers of the GolLayers. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            strides_convolution (int or tuple): Integer or tuple/list of integers, specifying the strides of the convolution. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the convolution layers of the GolLayers.
            strides_pooling (int or tuple): Integer or tuple/list of integers, specifying the strides of the pooling. For ``cnn_dim`` = 1, use a tuple with 1 integer; For ``cnn_dim`` = 2, a tuple with 2 integers and for ``cnn_dim`` = 3, a tuple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions. This applies to *all* the pooling layers of the GolLayers.
            padding (str): Available options are ``valid`` or ``same``. ``valid`` means no padding. ``same`` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. This applies to *all* the pooling layers of the GolLayers.
            dropout_percentage (float): Fraction of the input units to drop. This applies to *all* the dropout layers of the GolLayers.
            activation_function_gol (str or None): Activation function to use in the convolution layers of the GolLayers. If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_. This applies to *all* the convolution layers of the GolLayers.
            activation_function_dense (str or None): Activation function to use on the dense layer (i.e., the dense layer after the set of groups of layers). If not specified, no activation is applied (i.e., uses the linear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            **kwargs: *Ignored*. Extra arguments that are used for compatibility reasons.

        References:
            .. [Men20] Mendonça, F.; Mostafa, S.; Morgado-Dias, F.; Juliá-Serdá, G.; Ravelo-García, A. A Method for Sleep Quality Analysis Based on CNN Ensemble With Implementation in a Portable Wireless Device. *IEEE Access* **2020**, 8, 158523–158537.

        Examples:
            .. code-block:: python
                :linenos:

                import pandas as pd

                from hosa.Models.CNN import CNNRegression
                from hosa.aux import create_overlapping

                # 1 - Download, load, and split the data
                dataset = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', header=0, index_col=0)
                X = dataset.Passengers.to_numpy().reshape((len(dataset), 1))
                y = dataset.Passengers.to_numpy()
                X_train, y_train = X[:100], y[:100]
                X_test, y_test = X[100:], y[100:]
                # 2 - Prepare the data for CNN input
                X_train, y_train = create_overlapping(X_train, y_train, CNNRegression, 'central', 4)
                X_test, y_test = create_overlapping(X_test, y_test, CNNRegression, 'central', 4)
                # 3 - Create and fit the model
                clf = CNNRegression(1, 50, [5], epochs=500, patience=500)
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
        self.metrics, self.n_outputs = metrics, n_outputs
        super().__init__(n_outputs, n_neurons_dense_layer, n_kernels, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, dropout_percentage, activation_function_gol, activation_function_dense, batch_size, epochs, patience, **kwargs)

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GolLayers, flatten and dense layers.

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
            tuple: Returns a tuple containing the mean squared error (MSE) and coefficient of determination (:math:`R^2`).

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
            numpy.ndarray: Returns an array containing the estimates.
        """
        X = np.expand_dims(X, axis=-1)
        y_pred = self.model.predict(X, **kwargs)
        return y_pred

    def compile(self):
        """

        Compiles the model for training.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=self.metrics)
