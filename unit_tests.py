import unittest

import numpy as np
from pandas import read_csv, concat, DataFrame
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from project.CNN import CNNClassification, CNNRegression
from project.LSTM import LSTMClassification, LSTMRegression


def run_binary_classification_cnn():
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, 10, [3], epochs=5, patientece=10, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return True
    except:
        return False


def run_multiclass_classification_cnn():
    try:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = train_images.reshape((-1, 28 * 28))
        test_images = test_images.reshape((-1, 28 * 28))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        clf = CNNClassification(len(class_names), 10, [3], epochs=5, verbose=0)
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels)
        clf.predict(test_images)
        return True
    except:
        return False


def run_regression_cnn():
    try:
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        reg = CNNRegression(1, 10, [3], patientece=2, epochs=5, verbose=0)
        reg.prepare(X_train, y_train)
        reg.compile()
        reg.fit(X_train, y_train)
        reg.predict(X_test)
        return True
    except:
        return False


def run_multiclass_classification_lstm():
    try:
        num_distinct_words = 5000
        max_sequence_length = 300
        number_classes = 2
        is_bidirectional = False
        n_units = 2
        n_subs_layers = 2
        n_neurons_last_dense_layer = 10
        (X_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
        X_train = pad_sequences(X_train, maxlen=max_sequence_length, value=0.0)
        X_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)
        clf = LSTMClassification(number_classes, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, patientece=2, epochs=5, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        X_test = np.expand_dims(X_test, axis=-1)
        clf.predict(X_test)
        return True
    except:
        return False


def run_regression_lstm():
    """
    SOURCE: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    """

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    dataset = read_csv('datasets/pollution.csv', header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 8] = encoder.fit_transform(values[:, 8])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    try:
        number_outputs = 1
        is_bidirectional = False
        n_units = 2
        n_subs_layers = 2
        n_neurons_last_dense_layer = 10
        reg = LSTMRegression(number_outputs, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, patientece=2, epochs=5, verbose=0)
        reg.prepare(X_train, y_train)
        reg.compile()
        reg.fit(X_train, y_train)
        X_test = np.expand_dims(X_test, axis=-1)
        reg.predict(X_test)
        return True
    except:
        return False


class CNNTest(unittest.TestCase):

    def test_cnn_binary_classification(self):
        self.assertEqual(run_binary_classification_cnn(), True)

    def test_cnn_multiclass_classification(self):
        self.assertEqual(run_multiclass_classification_cnn(), True)

    def test_cnn_regression(self):
        self.assertEqual(run_regression_cnn(), True)

    def test_lstm_multiclass_classification(self):
        self.assertEqual(run_regression_lstm(), True)

    def test_lstm_regression(self):
        self.assertEqual(run_multiclass_classification_lstm(), True)


if __name__ == '__main__':
    unittest.main()
