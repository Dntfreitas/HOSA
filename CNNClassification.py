import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import AUC

from Callbacks.EarlyStoppingAtMinLoss import EarlyStoppingAtMinLoss


class CNN(abc.ABC):
    def __init__(self, n_neurons_first_dense_layer, gol_sizes,
                 cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.n_neurons_first_dense_layer, self.gol_sizes, self.cnn_dim, self.kernel_size, self.pool_size, self.strides_convolution, self.strides_pooling, self.padding, self.activation_function_gol, self.activation_function_dense, self.batch_size, self.epochs, self.patientece, self.dropout_percentage, self.verbose = n_neurons_first_dense_layer, gol_sizes, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose
        self.model = tf.keras.models.Sequential()
        self.n_features = None

    def prepare(self, n_features):
        self.n_features = n_features
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.n_features, 1)))
        for n_output_filters in self.gol_sizes:
            self.__add_gol(n_output_filters, self.cnn_dim)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_neurons_first_dense_layer, activation=self.activation_function_dense))

    def __add_gol(self, n_output_filters, cnn_dim):
        if cnn_dim == 1:
            self.model.add(tf.keras.layers.Conv1D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, strides=self.strides_pooling))
        elif cnn_dim == 2:
            self.model.add(tf.keras.layers.Conv2D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides_pooling))
        elif cnn_dim == 3:
            self.model.add(tf.keras.layers.Conv3D(n_output_filters, kernel_size=self.kernel_size, activation=self.activation_function_gol, strides=self.strides_convolution))
            self.model.add(tf.keras.layers.MaxPooling3D(pool_size=self.pool_size, strides=self.strides_pooling))
        else:
            raise ValueError('`cnn_dim` parameter must be int: 1, 2 or 3.')
        self.model.add(tf.keras.layers.Dropout(self.dropout_percentage))

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplemented

    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplemented

    def predict(self, X):
        y_probs = self.model.predict(X)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels


class CNNClassification(CNN):
    def __init__(self, number_classes, class_weights,
                 n_neurons_first_dense_layer, gol_sizes,
                 cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.number_classes, self.class_weights = number_classes, class_weights
        super().__init__(n_neurons_first_dense_layer, gol_sizes, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose)

    def prepare(self, n_features):
        super().prepare(n_features)
        self.model.add(tf.keras.layers.Dense(self.number_classes, activation='softmax'))

    def fit(self, X, y):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33)
        callbacks = EarlyStoppingAtMinLoss(self, self.patientece, (X_validation, y_validation))
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_validation, y_validation), class_weight=self.class_weights, callbacks=[callbacks])

    def score(self, X, y):
        X = np.expand_dims(X, axis=-1)
        y_probs, y_pred = self.predict(X)
        y = np.argmax(y, axis=1)  # reverse the to categorical
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        fpr, tpr, thresholds = roc_curve(y, y_probs[:, 1])
        auc_value = auc(fpr, tpr)
        return accuracy, sensitivity, specificity, auc_value


class CNNRegression(CNN):
    def __init__(self, number_outputs,
                 n_neurons_first_dense_layer, gol_sizes,
                 cnn_dim=1, kernel_size=3, pool_size=2, strides_convolution=1, strides_pooling=2, padding='valid',
                 activation_function_gol='relu', activation_function_dense='relu',
                 batch_size=1000, epochs=50, patientece=5, dropout_percentage=0.1,
                 verbose=1):
        self.number_outputs = number_outputs
        super().__init__(n_neurons_first_dense_layer, gol_sizes, cnn_dim, kernel_size, pool_size, strides_convolution, strides_pooling, padding, activation_function_gol, activation_function_dense, batch_size, epochs, patientece, dropout_percentage, verbose)

    def prepare(self, n_features):
        super().prepare(n_features)
        self.model.add(tf.keras.layers.Dense(self.number_outputs, activation='linear'))

    def fit(self, X, y):  # TODO
        raise NotImplemented

    def score(self, X, y):  # TODO
        raise NotImplemented
