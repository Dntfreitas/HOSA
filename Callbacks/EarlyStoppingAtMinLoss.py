import numpy as np
import tensorflow as tf


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

    def __init__(self, class_model, patience, validation_data):
        super().__init__()
        self.class_model, self.model, self.patience = class_model, self.class_model.model, patience
        self.X_validation, self.y_validation = validation_data
        self.best_weights = self.wait = self.stopped_epoch = self.best_metric_value = None
        self.early_stopping = False

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum
        self.wait = 0
        # The epoch the training stops at
        self.stopped_epoch = 0
        # Initialize the best as infinity
        self.best_metric_value = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_metric, *_ = self.class_model.score(self.X_validation, self.y_validation)
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self.wait = 0
            # Record the best weights if current results is better (less)
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
