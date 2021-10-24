import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import AUC

from Callbacks.EarlyStoppingAtMinLoss import EarlyStoppingAtMinLoss


class CNNClassification:
    def __init__(self, class_weights, n_neurons_first_dense_layer, gof_layer_sizes, batch_size=1000, epochs=50, patientece=5, verbose=1):
        self.class_weights, self.n_neurons_first_dense_layer, self.gof_layer_sizes, self.batch_size, self.epochs, self.patientece, self.verbose = class_weights, n_neurons_first_dense_layer, gof_layer_sizes, batch_size, epochs, patientece, verbose
        self.model = tf.keras.models.Sequential()
        self.n_features = None

    def prepare(self, n_features):
        self.n_features = n_features
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.n_features, 1)))
        for n_output_filters in self.gof_layer_sizes:
            self.__add_gof_layer(n_output_filters)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.n_neurons_first_dense_layer, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

    def __add_gof_layer(self, n_output_filters):
        self.model.add(tf.keras.layers.Conv1D(n_output_filters, kernel_size=2, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))
        self.model.add(tf.keras.layers.Dropout(0.1))

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])

    def fit(self, X, y):
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33)
        callbacks = EarlyStoppingAtMinLoss(self, self.patientece, (X_validation, y_validation))
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_validation, y_validation), class_weight=self.class_weights, callbacks=[callbacks])

    def predict(self, X):
        y_probs = self.model.predict(X)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

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
