import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


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


def create_overlapping(X, y, overlapping_type, overlapping_epochs, stride=1, apply_data_standardization=True):
    # Check if the parameter `apply_data_standardization` is boolean
    if type(apply_data_standardization) != bool:
        raise ValueError('The parameter `apply_data_standardization` must be boolean.')
    # Check if the number of overlapping epochs is even
    if (overlapping_epochs > 0 and overlapping_epochs % 2 == 0) or overlapping_epochs < 0:
        raise ValueError('The number of overlapping epochs should be an odd positive number.')
    epochs, n_points = X.shape
    x_flatten = X.flatten()
    window_size = n_points * (overlapping_epochs + 1)
    # Check if it has enough data points to create the overlapping
    if window_size > len(x_flatten):
        raise ValueError('Not enough data to create the overlapping window.')
    if overlapping_epochs == 0:
        annotations = y[1:]
    elif overlapping_type == 'central':
        annotations = y[int(overlapping_epochs / 2):-int(overlapping_epochs / 2)]
    elif overlapping_type == 'left':
        annotations = y[overlapping_epochs:]
    elif overlapping_type == 'right':
        annotations = y[:-overlapping_epochs]
    else:
        raise ValueError(f'`{overlapping_type}` is not a valid type. The available types are: `central`, `left` and `right`.')
    idx = np.arange(len(x_flatten))
    idx_win = sliding_window(idx, window_size)[::n_points * stride]
    X_windowed = x_flatten[idx_win]
    if apply_data_standardization:
        X_windowed = StandardScaler().fit_transform(X_windowed)
    return X_windowed, annotations
