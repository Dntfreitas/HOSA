import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from project.LSTM import LSTMRegression
from project.aux import create_overlapping

dataset = read_csv('../datasets/pollution.csv', header=0, index_col=0)
values = dataset.values[:, 4:]
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')
X = values[:, 1:]
y = values[:, 0]
overlapping_type = 'right'
overlapping_epochs = 5
X, y = create_overlapping(X, y, overlapping_type, overlapping_epochs, stride=1, apply_data_standardization=False)
np.nan_to_num(X, copy=False)
np.nan_to_num(y, copy=False)
X_train, X_test, y_train, y_test = train_test_split(X, y)

number_outputs = 1
is_bidirectional = False
n_units = 2
n_subs_layers = 2
n_neurons_last_dense_layer = 10

reg = LSTMRegression(number_outputs, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, patientece=2, epochs=5, verbose=1)
reg.prepare(X_train, y_train)
reg.compile()
reg.model.summary()
# reg.fit(X_train, y_train)
