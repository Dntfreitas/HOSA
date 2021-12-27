import numpy as np
import tensorflow as tf


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

    def __init__(self, class_model, patience, validation_data, rtol=1e-03, atol=1e-04):
        super().__init__()
        self.class_model = class_model
        self.model, self.patience = self.class_model.model, patience
        self.X_validation, self.y_validation = validation_data
        self.best_weights = self.wait = self.stopped_epoch = self.best_metric_value = self.compare_function = None
        self.rtol, self.atol = rtol, atol
        self.early_stopping = False

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if 'Regression' in str(type(self.class_model)):
            self.best_metric_value = np.inf
            self.compare_function = np.less
        elif 'Classification' in str(type(self.class_model)):
            self.best_metric_value = -np.inf
            self.compare_function = np.greater
        else:
            raise ValueError('The class of the model is invalid. Only regression and classification models are currently available.')

    def on_epoch_end(self, epoch, logs=None):
        current_metric, *_ = self.class_model.score(self.X_validation, self.y_validation)
        if self.compare_function(current_metric, self.best_metric_value) and not np.isclose(current_metric, self.best_metric_value, rtol=self.rtol, atol=self.atol):
            self.best_metric_value = current_metric
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.early_stopping = True