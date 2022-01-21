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

from hosa.Helpers.functions import create_overlapping
from hosa.Models.CNN.cnn_models import CNNClassification, CNNRegression
from hosa.Models.RNN import RNNClassification, RNNRegression
from hosa.Optimization.hosa import HOSACNN, HOSARNN


def run_binary_classification_cnn(imbalance_correction):
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X = X[:, :10]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, [3, 2], epochs=200, patience=3)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train, imbalance_correction=imbalance_correction, verbose=0)
        clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_multiclass_classification_cnn(imbalance_correction):
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
        clf = CNNClassification(10, [3], epochs=5, strides_convolution=2, strides_pooling=2, padding='same')
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels, imbalance_correction=imbalance_correction, verbose=0)
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
        clf = CNNClassification(10, [3], epochs=5, cnn_dim=2)
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
        clf = CNNClassification(n_classes, [3], epochs=5, cnn_dim=3)
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
        X_train, y_train = create_overlapping(X_train, y_train, CNNRegression, 3, 'central', n_stride=1, n_timesteps=2)
        X_test, y_test = create_overlapping(X_test, y_test, CNNRegression, 3, 'central', n_stride=1, n_timesteps=2)
        reg = CNNRegression(1, [3, 5], patience=2, epochs=5, kernel_size=2, pool_size=1, strides_pooling=1)
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
        X_train, y_train = create_overlapping(X_train, y_train, RNNClassification, overlapping_epochs, 'central', n_stride=1, n_timesteps=2)
        X_test, y_test = create_overlapping(X_test, y_test, RNNClassification, overlapping_epochs, 'central', n_stride=1, n_timesteps=2)
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
        X, y = create_overlapping(X, y, RNNRegression, overlapping_epochs, overlapping_type, n_stride=stride, n_timesteps=timesteps)
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
        dataset = np.loadtxt('datasets/occupancy.txt', delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle=False)
        # CNN
        param_grid_rnn = {
                'overlapping_type':          ['central', 'left'],
                'overlapping_epochs':        [1],
                'n_kernels_first_gol':       [16, 32],
                'activation_function_dense': ['relu'],
                'mults':                     [1, 2],
                'optimizer':                 ['adam'],
                'batch_size':                [32],
        }
        clf = HOSACNN(X_train, y_train, CNNClassification, 2, param_grid_rnn, 0.01, validation_size=.05, apply_rsv=False)
        clf.fit(max_gol_sizes=4, show_progress=True, verbose=1, shuffle=False, imbalance_correction=True)
        score = clf.score(X_test, y_test)
        clf.get_model().__dict__()
        # RNN
        param_grid_rnn = {
                'overlapping_type':          ['central', 'left'],
                'model_type':                ['lstm', 'gru'],
                'overlapping_epochs':        [1],
                'timesteps':                 [1],
                'activation_function_dense': ['relu'],
                'n_units':                   [10, 12],
                'mults':                     [1, 2],
                'optimizer':                 ['adam'],
                'batch_size':                [32],
        }
        clf = HOSARNN(X_train, y_train, RNNClassification, 2, param_grid_rnn, 0.01, validation_size=.05, apply_rsv=False)
        clf.fit(max_n_subs_layers=4, show_progress=True, verbose=0, shuffle=False, imbalance_correction=True)
        score = clf.score(X_test, y_test)
        clf.get_model().__dict__()
    except Exception as e:
        print(e)
        return False


def run_hosa_regression():
    try:
        dataset = np.loadtxt('datasets/pollution.txt', delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)
        # CNN
        param_grid_rnn = {
                'overlapping_type':          ['central', 'left'],
                'overlapping_epochs':        [1],
                'n_kernels_first_gol':       [16, 32],
                'activation_function_dense': ['relu'],
                'mults':                     [1, 2],
                'optimizer':                 ['adam'],
                'batch_size':                [32],
        }
        regr = HOSACNN(X_train, y_train, CNNRegression, 1, param_grid_rnn, 0.01, apply_rsv=False)
        regr.fit(max_gol_sizes=4, show_progress=True, verbose=1, shuffle=False)
        score = regr.score(X_test, y_test)
        regr.get_model().__dict__()
        # RNN
        param_grid_rnn = {
                'overlapping_type':          ['central', 'left'],
                'model_type':                ['lstm', 'gru'],
                'overlapping_epochs':        [1],
                'timesteps':                 [1],
                'activation_function_dense': ['relu'],
                'n_units':                   [10, 12],
                'mults':                     [1, 2],
                'optimizer':                 ['adam'],
                'batch_size':                [32],
        }
        regr = HOSARNN(X_train, y_train, RNNRegression, 1, param_grid_rnn, 0.01, apply_rsv=False)
        regr.fit(max_n_subs_layers=4, show_progress=True, verbose=1, shuffle=False)
        score = regr.score(X_test, y_test)
        regr.get_model().__dict__()
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

    def test_hosa_regression(self):
        self.assertEqual(run_hosa_regression(), True)

    def test_hosa_classification(self):
        self.assertEqual(run_hosa_classification(), True)
