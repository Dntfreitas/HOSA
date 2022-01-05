import operator
from functools import partial, reduce
from itertools import product

import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, balanced_accuracy_score


def sliding_window(x, window_shape):
    window_shape = (window_shape,)
    x = np.array(x, copy=False, subok=False)
    axis = tuple(range(x.ndim))
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape, subok=False, writeable=False)


def metrics_multiclass(y_true, y_probs, n_classes, inbalance_correction=False):
    y_pred = np.argmax(y_probs, axis=1)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]
    if inbalance_correction:
        classes_weight = np.sum(mcm[:, 1, :], axis=1) / np.sum(mcm[:, 1, :])
        sensitivity = np.average(tp / (tp + fn), weights=classes_weight)
        specificity = np.average(tn / (fp + tn), weights=classes_weight)
        accuracy = balanced_accuracy_score(y_true, y_pred)
    else:
        sensitivity = np.mean(tp / (tp + fn))
        specificity = np.mean(tn / (fp + tn))
        accuracy = accuracy_score(y_true, y_pred)
    if n_classes > 2:
        if inbalance_correction:
            auc_value = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        else:
            auc_value = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
    else:
        auc_value = roc_auc_score(y_true, y_probs[:, 1])
    return auc_value, accuracy, sensitivity, specificity


def create_overlapping(X, y, model, overlapping_epochs, overlapping_type=None, stride=1, timesteps=None):
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
        return cnn(X, y, overlapping_type, overlapping_epochs, stride)
    elif 'RNN' in str(model):
        return rnn(X, y, timesteps, overlapping_type, overlapping_epochs, stride)
    else:
        raise TypeError('The type of the model is invalid.')


def create_parameter_grid(param_grid):
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


def n_points(param_grid):
    product = partial(reduce, operator.mul)
    n_total_parameters = sum(product(len(v) for v in p.values()) if p else 1 for p in param_grid)
    # Select the step
    step = 0
    for p in param_grid:
        for v in p:
            current_step = len(p[v])
            if current_step > step:
                step = current_step
    return n_total_parameters, step
