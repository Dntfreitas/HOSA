import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import MSE

from project.Callbacks import EarlyStoppingAtMinLoss
from project.aux import metrics_multiclass


class BaseCNN:
    def __init__(self, n_neurons_first_dense_layer, gol_sizes,
                 optimizer='adam', cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, dropout_percentage=0.1, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, verbose=1):

        """Base class for Convolutional Neural Network (CNN) models for classification and regression.

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.CNNClassification` or :class:`.CNNRegression`.

        Args:
            n_neurons_first_dense_layer (str): Number of neuron units in the first dense layer.
            gol_sizes (numpy.ndarray): *i*-th element represents the number of output filters in the *i*-th GofLayer.
                Each GofLayer comprises one convolution layer, followed by one subsampling layer and a 10% dropout layer.
            optimizer (str): Name of optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            cnn_dim (int): Number of dimensions applicable to all the convolution layers of the GofLayers.
            kernel_size (int or tuple): Integer or tuple/list of integers, specifying the length (for ``cnn_dim`` = 1), the height and width (for ``cnn_dim`` = 2), or  the depth, height and width (for ``cnn_dim`` = 3) of convolution window. It can also be a single integer to specify the same value for all spatial dimensions.
            pool_size (int or tuple): Size of the max pooling window applicable to all the max pooling layers of the GofLayers. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            strides_convolution (int or tuple): Integer or tuple/list of integers, specifying the strides of the convolution. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            strides_pooling (int or tuple): Integer or tuple/list of integers, specifying the strides of the pooling. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. It can also be a single integer to specify the same value for all spatial dimensions.
            dropout_percentage (float): Fraction of the input units to drop.
            padding (str): Available options are ``valid``or ``same``. ``valid`` means no padding. ``same`` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
            activation_function_gol (str or None): Activation function to use. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            activation_function_dense: Activation function to use on the last dense layer. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Number of epochs to train the model.
            patientece (int): Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. Available options are ``0``, for silent mode, or ``1```, for a progress bar.

        .. note::
            The parameters used in this library were adapted from the same parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.
        """
        self.optimizer, self.n_neurons_first_dense_layer, self.gol_sizes, self.cnn_dim, self.kernel_size, self.pool_size, self.strides_convolution, self.strides_pooling, self.padding, self.activation_function_gol, self.activation_function_dense, self.batch_size, self.epochs, self.patientece, self.dropout_percentage, self.verbose = optimizer, n_neurons_first_dense_layer, gol_sizes, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose
        self.model = tf.keras.models.Sequential()

    def prepare(self, X, y):
        """

        Prepares the model by adding the layers to the estimator: input layer, GofLayers, and flatten and dense layers.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        """
        n_features = X.shape[-1]
        self.model.add(tf.keras.layers.InputLayer(input_shape=(n_features, 1)))
        for n_output_filters in self.gol_sizes:
            self.__add_gol(n_output_filters, self.cnn_dim)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_neurons_first_dense_layer, activation=self.activation_function_dense))

    def __add_gol(self, n_output_filters, cnn_dim):
        """

        Adds the GofLayers to the estimator.

        Args:
            n_output_filters (int): Number of output filters.
            cnn_dim (int): Number of dimensions.
        """
        if cnn_dim == 1:
            self.model.add(tf.keras.layers.Conv1D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        elif cnn_dim == 2:
            self.model.add(tf.keras.layers.Conv2D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        elif cnn_dim == 3:
            self.model.add(tf.keras.layers.Conv3D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling3D(pool_size=self.pool_size, strides=self.strides_pooling, padding=self.padding))
        else:
            raise ValueError('`cnn_dim` parameter must be 1, 2 or 3.')
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))

    def aux_fit(self, X, y, callback, class_weights, test_size, **kwargs):
        """
        Auxiliar function for classification and regression models compatibility.

        .. warning::
            This function is not meant to be called by itself. It is just an auxiliary function called by the child classes' ``fit`` function.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
            callback (EarlyStoppingAtMinLoss): Early stopping callback to halt the training of the model.
            class_weights (None or dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            test_size (float or int): Proportion of the dataset to include in the test split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.
        """

        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_size)
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


class CNNClassification(BaseCNN):
    def __init__(self, number_classes, n_neurons_first_dense_layer, gol_sizes,
                 optimizer='adam', metrics='accuracy', cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.metrics, self.number_classes, self.is_binary = metrics, number_classes, None
        super().__init__(n_neurons_first_dense_layer, gol_sizes, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, dropout_percentage, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, verbose)

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

    def fit(self, X, y, class_weights=None, test_size=0.33, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            class_weights (dict): Dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
            test_size (float or int): Proportion of the dataset to include in the test split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, class_weights, test_size, **kwargs)
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
        X = np.expand_dims(X, axis=-1)
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


class CNNRegression(BaseCNN):
    def __init__(self, number_outputs,
                 n_neurons_first_dense_layer, gol_sizes,
                 optimizer='adam', cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.number_outputs = number_outputs
        super().__init__(n_neurons_first_dense_layer, gol_sizes, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, dropout_percentage, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, verbose)

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

    def fit(self, X, y, test_size=0.33, **kwargs):
        """

        Fits the model to data matrix X and target(s) y.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target values (i.e., class labels).
            test_size (float or int): Proportion of the dataset to include in the test split.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``fit`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_.

        Returns:
            tensorflow.keras.Sequential: Returns a trained TensorFlow model.
        """
        callback = EarlyStoppingAtMinLoss
        super().aux_fit(X, y, callback, None, test_size, **kwargs)

    def predict(self, X, **kwargs):
        """

        Predicts the target values using the input data in the trained model.

        Args:
            X (numpy.ndarray): Input data.
            **kwargs: Extra arguments that are used in the TensorFlow's model ``predict`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_.
        """
        y_pred = self.model.predict(X, **kwargs)
        return y_pred

    def score(self, X, y):
        """

        Computes the performance metrics on the given input data and target values.

        Args:
            X: Input data.
            y: Target values (i.e., class labels).

        Returns:
            list: List containing the mean squared error (MSE) and coefficient of determination (:math:`R^2`).

        """
        X = np.expand_dims(X, axis=-1)
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        return mse, r2

    def compile(self, **kwargs):
        """

        Compiles the model for training.

        Args:
            **kwargs: Extra arguments that are used in the TensorFlow's model ``compile`` function. See `here <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`_.

        Returns:
            tensorflow.keras.Sequential: Returns an untrained but compiled TensorFlow model.
        """
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=[MSE], **kwargs)
