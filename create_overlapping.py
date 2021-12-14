import numpy as np
from sklearn.preprocessing import StandardScaler


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
    idx_win = np.lib.stride_tricks.sliding_window_view(idx, window_size)[::n_points * stride]
    X_windowed = x_flatten[idx_win]
    if apply_data_standardization:
        X_windowed = StandardScaler().fit_transform(X_windowed)
    return X_windowed, annotations
