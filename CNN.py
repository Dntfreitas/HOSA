import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense
from tensorflow_core.python.keras.metrics import AUC
from tensorflow_core.python.layers.core import Flatten


class CNN:
    def __init__(self, class_weights, n_neurons_first_dense_layer, gof_layer_sizes, batch_size=1000, epochs=20, verbose=1):
        self.class_weights, self.verbose, self.n_neurons_first_dense_layer, self.batch_size, self.epochs, self.gof_layer_sizes = class_weights, verbose, n_neurons_first_dense_layer, batch_size, epochs, gof_layer_sizes
        self.model = Sequential()
        self.prepare_model()
        self.n_features = None

    def prepare_model(self):
        for n_output_filters in self.gof_layer_sizes:
            self.add_gof_layer(n_output_filters)
        self.model.add(Flatten())
        self.model.add(Dense(self.n_neurons_first_dense_layer, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

    def add_gof_layer(self, n_output_filters):
        self.model.add(Conv1D(n_output_filters, kernel_size=2, strides=1, activation='relu', input_shape=(self.n_features, 1)))
        self.model.add(MaxPooling1D(pool_size=2, strides=2))
        self.model.add(Dropout(0.1))

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[AUC()])

    def fit(self, X, y):
        self.n_features = X.shape[1]
        X, X_validation, y, y_validation = train_test_split(X, y, test_size=0.33)
        self.model.fit(X, y,
                       batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(X_validation, y_validation),
                       shuffle=True, class_weight=self.class_weights,
                       callbacks=EarlyStoppingAtMinLoss(self.model, patienteceValue, X_validation, y_validation))

    def predict(self, X):
        y_probs = self.model.predict(X)
        y_pred_labels = np.argmax(y_probs, axis=1)
        return y_probs, y_pred_labels

    def score(self, X, y):
        y_probs, y_pred = self.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        fpr, tpr, thresholds = roc_curve(y, y_probs[:, 1])
        auc_value = auc(fpr, tpr)
        return accuracy, sensitivity, specificity, auc_value

    class EarlyStoppingAtMinLoss(Callback):
        """Stop training when the loss is at its min, i.e. the loss stops decreasing.

        Arguments:
            patience: Number of epochs to wait after min has been hit. After this
            number of no improvement, training stops.
        """

        def __init__(self, model, patienteceValue, valid_data):
            super(EarlyStoppingAtMinLoss, self).__init__()
            self.patience = patienteceValue
            self.best_weights = None
            self.validation_data = valid_data
            self.model = model

        def on_train_begin(self, logs=None):
            # The number of epoch it has waited when loss is no longer minimum.
            self.wait = 0
            # The epoch the training stops at.
            self.stopped_epoch = 0
            # Initialize the best as infinity.
            #    self.best = np.Inf
            self.best = 0.2
            self._data = []
            self.current_auc = 0.2
            print('Train started')

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val = self.validation_data[0], self.validation_data[1]
            y_predict = np.asarray(self.model.predict(X_val))

            fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_val, axis=1), y_predict[:, 1])
            auc_keras = auc(fpr_keras, tpr_keras)
            self.current_auc = auc_keras
            current = auc_keras
            #    self.curentAUC = current
            #    print('AUC %05d' % (self.bestAUC))
            print('AUC : ', current)

            #    current = logs.get('loss')
            if np.greater(self.current_auc, self.best + thresholdAphase):  # np.less
                print('Update')
                self.best = self.current_auc
                self.wait = 0
                # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
