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
                 batch_size=1000, epochs=50, patientece=5,verbose=1):
        """Base class for Convolutional Neural Network (CNN) models for classification and regression.

        .. warning::
            This class should not be used directly. Use derived classes instead, i.e., :class:`.CNNClassification` or :class:`.CNNRegression`.

        Args:
            n_neurons_first_dense_layer (str): Number of neuron units in the first dense layer.
            gol_sizes (numpy.ndarray): *i*-th element represents the number of output filters in the *i*-th *GofLayer*.
                Each *GofLayer* is composed of one convolution layer, followed by one subsampling layer, and a 10% dropout layer.
            optimizer (str): Name of optimizer. See `tensorflow.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_.
            cnn_dim (int): Number of dimensions applicable to all the convolution layers of the GofLayers.
            kernel_size (int or tuple): Integer or tuple/list of integers, specifying the length (for ``cnn_dim`` = 1), the height and width (for ``cnn_dim`` = 2), or  the depth, height and width (for ``cnn_dim`` = 3) of convolution window. Can also be a single integer to specify the same value for all spatial dimensions.
            pool_size (int or tuple): Size of the max pooling window applicable to all the max pooling layers of the GofLayers. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. Can also be a single integer to specify the same value for all spatial dimensions.
            strides_convolution (int or tuple): Integer or tuple/list of integers, specifying the strides of the convolution. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. Can also be a single integer to specify the same value for all spatial dimensions.
            strides_pooling (int or tuple): Integer or tuple/list of integers, specifying the strides of the pooling. For ``cnn_dim`` = 1, use a tupple with 1 integer; For ``cnn_dim`` = 2, a tupple with 2 integers and for ``cnn_dim`` = 3, a tupple with 3 integers. Can also be a single integer to specify the same value for all spatial dimensions.
            dropout_percentage (float): Fraction of the input units to drop.
            padding (str): Available options are ``valid``or ``same``. ``valid`` means no padding. ``same`` results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
            activation_function_gol (str or None): Activation function to use. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            activation_function_dense: Activation function to use on the last dense layer. If not specified, no activation is applied (i.e., applies the liear activation function). See `tensorflow.keras.activations <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_.
            batch_size (int or None): Number of samples per batch of computation. If ``None``, ``batch_size`` will default to 32.
            epochs (int): Number of epochs to train the model.
            patientece: Number of epochs with no improvement after which training will be stopped.
            verbose (int): Verbosity mode. Available options are ``0``, for silent mode, or ``1```, for progress bar.

        .. note::
            The parameters used in this library were adapted from the same parameters of the TensorFlow library. Descriptions were thus modified accordingly to our approach.  However, refer to the TensorFlow documentation for more details about each of those parameters.


        """


        self.optimizer, self.n_neurons_first_dense_layer, self.gol_sizes, self.cnn_dim, self.kernel_size, self.pool_size, self.strides_convolution, self.strides_pooling, self.padding, self.activation_function_gol, self.activation_function_dense, self.batch_size, self.epochs, self.patientece, self.dropout_percentage, self.verbose = optimizer, n_neurons_first_dense_layer, gol_sizes, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose
        self.model = tf.keras.models.Sequential()

    def prepare(self, x, y):
        n_features = x.shape[-1]
        self.model.add(tf.keras.layers.InputLayer(input_shape=(n_features, 1)))
        for n_output_filters in self.gol_sizes:
            self.__add_gol(n_output_filters, self.cnn_dim)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_neurons_first_dense_layer, activation=self.activation_function_dense))

    def __add_gol(self, n_output_filters, cnn_dim):
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

    def ffit(self, X, y, callback, class_weights, test_size):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_size)
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)
        callbacks = [callback(self, self.patientece, (X_validation, y_validation))]
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_validation, y_validation), callbacks=callbacks, class_weight=class_weights, verbose=self.verbose)

    @abc.abstractmethod
    def compile(self):
        raise NotImplemented

    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplemented

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplemented


class CNNClassification(BaseCNN):
    def __init__(self, number_classes, n_neurons_first_dense_layer, gol_sizes,
                 optimizer='adam', metrics='accuracy', cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.metrics, self.number_classes, self.is_binary = metrics, number_classes, None
        super().__init__(n_neurons_first_dense_layer, gol_sizes, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose)

    def prepare(self, x, y):
        super().prepare(x, y)
        self.model.add(tf.keras.layers.Dense(self.number_classes, activation='softmax'))
        return self.model

    def fit(self, X, y, class_weights=None, test_size=0.33):
        callback = EarlyStoppingAtMinLoss
        super().ffit(X, y, callback, class_weights, test_size)

    def score(self, X, y):
        X = np.expand_dims(X, axis=-1)
        y_probs, y_pred = self.predict(X)
        auc_value, accuracy, sensitivity, specificity = metrics_multiclass(y, y_probs, self.number_classes)
        return auc_value, accuracy, sensitivity, specificity

    def predict(self, X):
        y_probs = self.model.predict(X)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

    def compile(self):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=self.metrics)


class CNNRegression(BaseCNN):
    def __init__(self, number_outputs,
                 n_neurons_first_dense_layer, gol_sizes,
                 optimizer='adam', cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.number_outputs = number_outputs
        super().__init__(n_neurons_first_dense_layer, gol_sizes, optimizer, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose)

    def prepare(self, X, y):
        super().prepare(X, y)
        self.model.add(tf.keras.layers.Dense(self.number_outputs, activation='linear'))
        return self.model

    def fit(self, X, y, test_size=0.33):
        callback = EarlyStoppingAtMinLoss
        super().ffit(X, y, callback, None, test_size)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def score(self, X, y):
        X = np.expand_dims(X, axis=-1)
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        return mse, r2

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=[MSE])
