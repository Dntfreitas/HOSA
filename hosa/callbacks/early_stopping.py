"""
Utilities for implementing early stopping callbacks.
"""
import numpy as np
import tensorflow as tf


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """ This class implements the early stopping for avoiding overfitting the model.
    The training is stopped when the monitored metric has stopped improving.

    Args:
        class_model: Class of the object to be optimized. Available options are:
        :class:`.RNNClassification`, :class:`.RNNRegression`, :class:`.CNNClassification`
        and
        :class:`.CNNRegression`.
        patience (int): Number of epochs with no improvement after which training will be
        stopped.
        validation_data (numpy.ndarray): Input data extracted from the validation dataset (
        which was itself extracted from the training dataset).
        imbalance_correction (bool): `True` if correction for imbalance should be applied to
        the metrics; `False` otherwise.
        rtol (float): The relative tolerance parameter, as used in `numpy.isclose`. See
        `numpy.isclose <https://numpy.org/doc/stable/reference/generated/numpy.isclose
        .html>`_.
        atol (float): The absolute tolerance parameter, as used in `numpy.isclose`. See
        `numpy.isclose <https://numpy.org/doc/stable/reference/generated/numpy.isclose
        .html>`_.
    """

    def __init__(self, class_model, patience, validation_data, imbalance_correction=False,
                 rtol=1e-03, atol=1e-04):
        super().__init__()
        self.class_model = class_model
        self.model, self.patience, self.imbalance_correction = self.class_model.model, patience, \
                                                               imbalance_correction
        self.x_validation, self.y_validation = validation_data
        self.best_weights = self.wait = self.stopped_epoch = self.best_metric_value = \
            self.compare_function = None
        self.rtol, self.atol = rtol, atol
        self.early_stopping = False

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training to initialize the variables for early stopping.

        Args:
            logs (dict): Currently no data is passed to this argument for this method but that
            may change in the future.
        """
        self.wait = 0
        self.stopped_epoch = 0
        if 'Regression' in str(type(self.class_model)):
            self.best_metric_value = np.inf
            self.compare_function = np.less
        elif 'Classification' in str(type(self.class_model)):
            self.best_metric_value = -np.inf
            self.compare_function = np.greater
        else:
            raise ValueError(
                    'The class of the model is invalid. Only regression and classification models '
                    'are currently available.')

    def on_epoch_end(self, epoch, logs=None):
        """
        Checks, based on the patience value, if the training should stop. After stopping,
        it restores the model's weights from the epoch with the best value of the monitored
        quantity.

        Args:
            epoch (int): Index of epoch.
            logs (dict): Currently no data is passed to this argument for this method but that
            may change in the future.
        """
        current_metric, *_ = self.class_model.score(self.x_validation, self.y_validation,
                                                    imbalance_correction=self.imbalance_correction)
        if self.compare_function(current_metric, self.best_metric_value) and not np.isclose(
                current_metric, self.best_metric_value, rtol=self.rtol, atol=self.atol):
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
        """
        This function is called when the training is finished, and it is used to set a flag for
        early stopping.

        Args:
            logs (dict): Currently no data is passed to this argument for this method but that
            may change in the future.
        """
        if self.stopped_epoch > 0:
            self.early_stopping = True
