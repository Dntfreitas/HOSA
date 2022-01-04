import unittest

import numpy as np
from pandas import read_csv
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras, random
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from project.CNN import CNNClassification, CNNRegression
from project.RNN import RNNClassification, RNNRegression
from project.aux import create_overlapping_cnn


def run_binary_classification_cnn(inbalance_correction):
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, 10, [3, 2], epochs=200, patientece=3, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train, inbalance_correction=inbalance_correction)
        clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_cnn(inbalance_correction):
    try:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images[:500]
        train_labels = train_labels[:500]
        test_images = test_images[:250]
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = train_images.reshape((-1, 28 * 28))
        test_images = test_images.reshape((-1, 28 * 28))
        clf = CNNClassification(10, 10, [3], epochs=5, verbose=0, strides_convolution=2, strides_pooling=2, padding='same')
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels, inbalance_correction=inbalance_correction)
        clf.predict(test_images)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_2dcnn():
    try:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images[:500]
        train_labels = train_labels[:500]
        test_images = test_images[:250]
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        clf = CNNClassification(10, 10, [3], epochs=5, cnn_dim=2, verbose=0)
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels)
        clf.predict(test_images)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_3dcnn():
    try:
        input_shape = (750, 28, 28, 28)
        n_classes = 3
        X = random.normal(input_shape).numpy()
        y = np.random.randint(0, n_classes, input_shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = CNNClassification(n_classes, 10, [3], epochs=5, cnn_dim=3, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_regression_cnn():
    try:
        X, y = fetch_california_housing(return_X_y=True)
        X = X[:500]
        y = y[:500]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        reg = CNNRegression(1, 10, [3, 5], patientece=2, epochs=5, verbose=0, kernel_size=2, pool_size=1, strides_pooling=1)
        reg.prepare(X_train, y_train)
        reg.compile()
        reg.fit(X_train, y_train)
        reg.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_rnn():
    try:
        num_distinct_words = 5000
        max_sequence_length = 300
        number_classes = 2
        is_bidirectional = False
        n_units = 2
        n_subs_layers = 2
        n_neurons_last_dense_layer = 10
        (X_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
        X_train = X_train[:500]
        y_train = y_train[:500]
        x_test = x_test[:250]
        X_train = pad_sequences(X_train, maxlen=max_sequence_length, value=0.0)
        X_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)
        X_test = np.expand_dims(X_test, axis=-1)
        for model in ['lstm', 'gru']:
            clf = RNNClassification(number_classes, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, model_type=model, patientece=2, epochs=5, verbose=0)
            clf.prepare(X_train, y_train)
            clf.compile()
            clf.fit(X_train, y_train)
            clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_regression_rnn(is_bidirectional, overlapping_type, overlapping_epochs=5, data_standardization_strategy='after', stride=1):
    try:
        dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv', header=0, index_col=0)
        dataset = dataset.head(750).copy()
        values = dataset.values[:, 4:]
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        values = values.astype('float32')
        X = values[:, 1:]
        y = values[:, 0]
        X, y = create_overlapping_cnn(X, y, overlapping_type, overlapping_epochs, stride=stride, data_standardization_strategy=data_standardization_strategy)
        np.nan_to_num(X, copy=False)
        np.nan_to_num(y, copy=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        number_outputs = 1
        n_units = 2
        n_subs_layers = 2
        n_neurons_last_dense_layer = 10
        X_test = np.expand_dims(X_test, axis=-1)
        for model in ['lstm', 'gru']:
            reg = RNNRegression(number_outputs, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, model_type=model, patientece=2, epochs=5, verbose=0)
            reg.prepare(X_train, y_train)
            reg.compile()
            reg.fit(X_train, y_train)
            reg.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


class CNNTest(unittest.TestCase):

    def test_cnn_binary_classification(self):
        self.assertEqual(run_binary_classification_cnn(True), True)
        self.assertEqual(run_binary_classification_cnn(False), True)

    def test_cnn_multiclass_classification(self):
        self.assertEqual(run_multiclass_classification_cnn(True), True)
        self.assertEqual(run_multiclass_classification_cnn(False), True)
        self.assertEqual(run_multiclass_classification_2dcnn(), True)
        self.assertEqual(run_multiclass_classification_3dcnn(), True)

    def test_cnn_regression(self):
        self.assertEqual(run_regression_cnn(), True)

    def test_rnn_multiclass_classification(self):
        self.assertEqual(run_multiclass_classification_rnn(), True)

    def test_rnn_regression(self):
        self.assertEqual(run_regression_rnn(True, 'central', stride=1), True)
        self.assertEqual(run_regression_rnn(False, 'right', stride=1), True)
        self.assertEqual(run_regression_rnn(False, 'left', stride=1), True)
        self.assertEqual(run_regression_rnn(True, 'central', stride=2), True)
        self.assertEqual(run_regression_rnn(False, 'right', stride=2), True)
        self.assertEqual(run_regression_rnn(False, 'left', stride=2), True)
        self.assertEqual(run_regression_rnn(True, 'central', data_standardization_strategy=None, stride=2), True)
        self.assertEqual(run_regression_rnn(True, 'central', data_standardization_strategy='before', stride=2), True)
        self.assertEqual(run_regression_rnn(False, 'central', overlapping_epochs=0, data_standardization_strategy='after', stride=1), True)


if __name__ == '__main__':
    unittest.main()
