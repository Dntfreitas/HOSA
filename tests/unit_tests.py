import unittest

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras, random
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hosa.Models.CNN.cnn_models import CNNClassification, CNNRegression
from hosa.Models.RNN import RNNClassification, RNNRegression
from hosa.Optimization.hosa import HOSA
from hosa.aux import create_overlapping


def run_binary_classification_cnn(inbalance_correction):
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X = X[:, :10]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, 10, [3, 2], epochs=200, patience=3)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train, inbalance_correction=inbalance_correction, verbose=0)
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
        clf = CNNClassification(10, 10, [3], epochs=5, strides_convolution=2, strides_pooling=2, padding='same')
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels, inbalance_correction=inbalance_correction, verbose=0)
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
        clf = CNNClassification(10, 10, [3], epochs=5, cnn_dim=2)
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels, verbose=0)
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
        clf = CNNClassification(n_classes, 10, [3], epochs=5, cnn_dim=3)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train, verbose=0)
        clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_regression_cnn():
    try:
        data = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
        X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population']]
        y = data['median_house_value']
        X = X[:500].fillna(0)
        y = y[:500].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train, y_train = create_overlapping(X_train, y_train, CNNRegression, 'central', 3, stride=1, timesteps=2)
        X_test, y_test = create_overlapping(X_test, y_test, CNNRegression, 'central', 3, stride=1, timesteps=2)
        reg = CNNRegression(1, 10, [3, 5], patience=2, epochs=5, kernel_size=2, pool_size=1, strides_pooling=1)
        reg.prepare(X_train, y_train)
        reg.compile()
        reg.fit(X_train, y_train, verbose=0)
        reg.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_rnn(is_bidirectional=False, overlapping_epochs=3):
    try:
        num_distinct_words = 5000
        max_sequence_length = 300
        number_classes = 2
        n_units = 2
        n_subs_layers = 2
        n_neurons_dense_layer = 10
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_distinct_words)
        X_train = X_train[:500]
        y_train = y_train[:500]
        X_test = X_test[:250]
        X_train = pad_sequences(X_train, maxlen=max_sequence_length, value=0.0)
        X_test = pad_sequences(X_test, maxlen=max_sequence_length, value=0.0)
        X_train, y_train = create_overlapping(X_train, y_train, RNNClassification, 'central', overlapping_epochs, stride=1, timesteps=2)
        X_test, y_test = create_overlapping(X_test, y_test, RNNClassification, 'central', overlapping_epochs, stride=1, timesteps=2)
        for model in ['lstm', 'gru']:
            clf = RNNClassification(number_classes, n_neurons_dense_layer, is_bidirectional=is_bidirectional, n_units=n_units, n_subs_layers=n_subs_layers, model_type=model, patience=2, epochs=5)
            clf.prepare(X_train, y_train)
            clf.compile()
            clf.fit(X_train, y_train, verbose=0)
            clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_regression_rnn(is_bidirectional, overlapping_type, overlapping_epochs=5, stride=1, timesteps=1):
    try:
        dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv', header=0, index_col=0)
        dataset = dataset.head(200).copy()
        values = dataset.values[:, 4:]
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        values = values.astype('float32')
        X = values[:, 1:]
        y = values[:, 0]
        X, y = create_overlapping(X, y, RNNRegression, overlapping_type, overlapping_epochs, stride=stride, timesteps=timesteps)
        np.nan_to_num(X, copy=False)
        np.nan_to_num(y, copy=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        number_outputs = 1
        n_units = 2
        n_subs_layers = 2
        n_neurons_dense_layer = 10
        for model in ['lstm', 'gru']:
            reg = RNNRegression(number_outputs, n_neurons_dense_layer, is_bidirectional=is_bidirectional, n_units=n_units, n_subs_layers=n_subs_layers, model_type=model, patience=2, epochs=5)
            reg.prepare(X_train, y_train)
            reg.compile()
            reg.fit(X_train, y_train, verbose=0)
            reg.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_hosa_classification():
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X = X[:, :5]
        param_grid = {
                'n_neurons_dense_layer': [5, 10],
                'gol_sizes':             [[3]],
                'overlapping_type':      ['central', 'right'],
                'overlapping_epochs':    [3],
                'stride':                [1],
                'timesteps':             [1, 2],
                'model_type':            ['lstm', 'gru']
        }
        clf = HOSA(X, y, CNNClassification, 2, param_grid, 0.1, n_splits=2, apply_rsv=True)
        clf.fit_cnn(inbalance_correction=True, validation_size=0.5, verbose=0)
        clf.score(X, y)
        clf = HOSA(X, y, RNNClassification, 2, param_grid, 0.1, apply_rsv=False)
        clf.fit_cnn(inbalance_correction=False, verbose=0)
        clf.score(X, y)
        return True
    except Exception as e:
        print(e)
        return False


def run_hosa_regression():
    try:
        dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv', header=0, index_col=0)
        dataset = dataset.head(200).copy()
        values = dataset.values[:, 4:]
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        values = values.astype('float32')
        X = values[:, 1:]
        y = values[:, 0]
        np.nan_to_num(X, copy=False)
        np.nan_to_num(y, copy=False)
        param_grid = {
                'n_neurons_dense_layer': [5, 10],
                'gol_sizes':             [[3]],
                'overlapping_type':      ['central', 'left'],
                'overlapping_epochs':    [3],
                'model_type':            ['lstm', 'gru']
        }
        clf = HOSA(X, y, CNNRegression, 1, param_grid, 0.1, apply_rsv=False)
        clf.fit_cnn(validation_size=0.5, verbose=0)
        clf.score(X, y)
        param_grid[0]['timesteps'] = [1]
        clf = HOSA(X, y, RNNRegression, 1, param_grid, 0.1, apply_rsv=False)
        clf.fit_cnn(verbose=0)
        clf.predict(X)
        return True
    except Exception as e:
        print(e)
        return False


class ModelTesting(unittest.TestCase):

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
        self.assertEqual(run_multiclass_classification_rnn(is_bidirectional=False), True)
        self.assertEqual(run_multiclass_classification_rnn(is_bidirectional=True, overlapping_epochs=0), True)

    def test_rnn_regression(self):
        self.assertEqual(run_regression_rnn(False, 'left', stride=2, timesteps=1), True)
        self.assertEqual(run_regression_rnn(False, 'right', stride=2, timesteps=1), True)
        self.assertEqual(run_regression_rnn(False, 'central', stride=2, timesteps=1), True)

    def test_hosa_classification(self):
        self.assertEqual(run_hosa_classification(), True)

    def test_hosa_regression(self):
        self.assertEqual(run_hosa_regression(), True)
