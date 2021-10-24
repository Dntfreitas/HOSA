import numpy as np
import tensorflow as tf


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

    def __init__(self, class_model, patience, validation_data):
        super().__init__()
        self.class_model = class_model
        self.model = self.class_model.model
        self.patience = patience
        self.X_validation, self.y_validation = validation_data
        self.best_weights = None
        self.wait = None
        self.stopped_epoch = None
        self.best_auc = None

    def on_train_begin(self, logs=None):
        print('Train started')
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_auc = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy, current_sensitivity, current_specificity, current_auc = self.class_model.score(self.X_validation, self.y_validation)
        print(f'Current AUC {current_auc:.2f}')
        if current_auc > self.best_auc:
            print('Update')
            self.best_auc = current_auc
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
            print(f'Epoch {self.stopped_epoch + 1:.0f}: early stopping')
