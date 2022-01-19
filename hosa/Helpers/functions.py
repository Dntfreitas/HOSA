from itertools import product

import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, balanced_accuracy_score


def sliding_window(X, window_size):
    """
    Creates a sliding window view of `X` according to the window size specified.

    .. note::
        This function is based on the NumPy's function `sliding_window_view`. See `numpy.lib.stride_tricks.sliding_window_view <https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html>`_.

    Args:
        X (numpy.ndarray): Input data.
        window_size (int): Size of the sliding window.

    Returns:
        (numpy.ndarray): Returns a sliding window view of the array.

    """
    window_size = (window_size,)
    X = np.array(X, copy=False, subok=False)
    axis = tuple(range(X.ndim))
    out_strides = X.strides + tuple(X.strides[ax] for ax in axis)
    x_shape_trimmed = list(X.shape)
    for ax, dim in zip(axis, window_size):
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_size
    return as_strided(X, strides=out_strides, shape=out_shape, subok=False, writeable=False)


def create_overlapping(X, y, model, n_overlapping_epochs=0, overlapping_type=None, n_stride=1, n_timesteps=None):
    """

    Depending on the model chosen, prepare the data with segmented windows according to the number of epochs and overlapping type.

    Args:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): Target values (class labels in classification, real numbers in regression).
        model (object): Class of the object to be optimized. Available options are: :class:`.RNNClassification`, :class:`.RNNRegression`, :class:`.CNNClassification` and :class:`.CNNRegression`.
        n_overlapping_epochs (int): Number of epochs to be overlapped (in other words, the overlap duration).
        overlapping_type (str): Type of overlapping to perform on the data. Available options are `central`, where the target value corresponds to the central epoch of the overlapping window; `left`, where the target value corresponds to the rightmost epoch of the overlapping window and `right`, where the target value corresponds to the leftmost epoch of the overlapping window.
        n_stride (int): Number of strides to apply to the data.
        n_timesteps (int): Number of timesteps to apply to the data for recurrent models, in other words, the number of lagged observations to be used in the model. **Only used when `model=RNNClassification` or `model=RNNRegression`.**

    Returns:
        tuple: Returns a tuple with the input data (`X`) and target values (`y`), both in segmented window view.

    """

    def cnn(X, y, overlapping_epochs, overlapping_type, stride):
        if overlapping_epochs < 0:
            raise ValueError('The number of overlapping epochs should be zero or a positive number.')
        epochs, n_points = X.shape
        if overlapping_epochs == 0:
            window_size = n_points
            y_windowed = y[::stride] if y is not None else None
        elif overlapping_type == 'central':
            window_size = n_points * (2 * overlapping_epochs + 1)
            y_windowed = y[overlapping_epochs:-overlapping_epochs:stride] if y is not None else None
        elif overlapping_type == 'left':
            window_size = n_points * (overlapping_epochs + 1)
            y_windowed = y[overlapping_epochs::stride] if y is not None else None
        elif overlapping_type == 'right':
            window_size = n_points * (overlapping_epochs + 1)
            y_windowed = y[:-overlapping_epochs:stride] if y is not None else None
        else:
            raise ValueError(f'`{overlapping_type}` is not a valid type. The available types are: `central`, `left` and `right`.')
        x_flatten = X.flatten()
        if window_size > len(x_flatten):
            raise ValueError('Not enough data to create the overlapping window.')
        idx = np.arange(len(x_flatten))
        idx_win = sliding_window(idx, window_size)[::n_points * stride]
        X_windowed = x_flatten[idx_win]
        return X_windowed, y_windowed

    def rnn(X, y, timesteps, overlapping_epochs, overlapping_type, stride):
        if timesteps is None:
            raise ValueError('`timesteps` must be defined.')
        X_windowed, y_windowed = cnn(X, y, overlapping_epochs, overlapping_type, stride)
        y_windowed = y_windowed[timesteps - 1:] if y_windowed is not None else None
        idx = np.arange(len(X_windowed))
        idx_win = sliding_window(idx, timesteps)
        X_windowed = X_windowed[idx_win]
        return X_windowed, y_windowed

    # According to the model, initialize the overlapping function
    if 'CNN' in str(model):
        return cnn(X, y, overlapping_type, n_overlapping_epochs, n_stride)
    elif 'RNN' in str(model):
        return rnn(X, y, n_timesteps, overlapping_type, n_overlapping_epochs, n_stride)
    else:
        raise TypeError('The type of the model is invalid.')


def metrics_multiclass(y_true, y_probs, n_classes, imbalance_correction=False):
    """Computes the performance metrics for classification problems.
    This function supports multiclass classification, being, in this case, the metrics given in terms of the average value, or weighed average if `imbalance_correction=True`.

    Args:
        y_true (numpy.ndarray): Ground truth (correct) labels.
        y_probs (numpy.ndarray): Probability estimates.
        n_classes (int): Number of classes (or labels) of the classification problem.
        imbalance_correction (bool): `True` if correction for imbalance should be applied to the metrics; `False` otherwise.

    Returns:
        (tuple): Returns a tuple with the metrics for AUC, accuracy, sensitivity, and specificity.

    """
    y_pred = np.argmax(y_probs, axis=1)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]
    if imbalance_correction:
        classes_weight = np.sum(mcm[:, 1, :], axis=1) / np.sum(mcm[:, 1, :])
        sensitivity = np.average(tp / (tp + fn), weights=classes_weight)
        specificity = np.average(tn / (fp + tn), weights=classes_weight)
        accuracy = balanced_accuracy_score(y_true, y_pred)
    else:
        sensitivity = np.mean(tp / (tp + fn))
        specificity = np.mean(tn / (fp + tn))
        accuracy = accuracy_score(y_true, y_pred)
    if n_classes > 2:
        if imbalance_correction:
            auc_value = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        else:
            auc_value = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
    else:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    return auc_value, accuracy, sensitivity, specificity


def create_parameter_grid(param_grid):
    """This function generates an iterator that can be traversed through all the parameter value combinations.
    The order of the generated parameter combinations is deterministic, being done according to the total number of values to try in each parameter in descending order.

    Args:
        param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    """
    for p in param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        keys_sorted = sorted(p, key=lambda key: len(p[key]), reverse=True)
        items = [(key, p[key]) for key in keys_sorted]
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params


def prepare_param_overlapping(specification):
    """
    Prepares, considering the given specification, the parameters for creating the input and output overlapping.

    Args:
        specification (dict): Parameter names mapped to their values.

    Returns:
        tuple: Returns a tuple containing the overlapping type, number of overlapping epochs, strides, and timesteps.

    """
    if 'overlapping_epochs' in specification:
        overlapping_epochs = specification['overlapping_epochs']
    else:
        overlapping_epochs = 0
    if overlapping_epochs > 0 and 'overlapping_type' in specification:
        overlapping_type = specification['overlapping_type']
    else:
        overlapping_type = None
    if 'stride' in specification:
        stride = specification['stride']
    else:
        stride = 1
    if 'timesteps' in specification:
        timesteps = specification['timesteps']
    else:
        timesteps = None
    return overlapping_type, overlapping_epochs, stride, timesteps
