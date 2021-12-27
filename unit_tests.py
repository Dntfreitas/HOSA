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
from project.LSTM import LSTMClassification, LSTMRegression
from project.aux import create_overlapping


def run_binary_classification_cnn(inbalance_correction):
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, 10, [3], epochs=200, patientece=3, verbose=0)
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
        train_images = train_images[:501]
        train_labels = train_labels[:501]
        test_images = test_images[:501]
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = train_images.reshape((-1, 28 * 28))
        test_images = test_images.reshape((-1, 28 * 28))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        clf = CNNClassification(len(class_names), 10, [3], epochs=5, verbose=0)
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
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        clf = CNNClassification(len(class_names), 10, [3], epochs=5, cnn_dim=2, verbose=0)
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
        reg = CNNRegression(1, 10, [3], patientece=2, epochs=5, verbose=0)
        reg.prepare(X_train, y_train)
        reg.compile()
        reg.fit(X_train, y_train)
        reg.predict(X_test)
        return True
    except Exception as e:
        print(e)
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
        X_train = X_train[:500]
        y_train = y_train[:500]
        x_test = x_test[:250]
        X_train = pad_sequences(X_train, maxlen=max_sequence_length, value=0.0)
        X_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)
        clf = LSTMClassification(number_classes, is_bidirectional, n_units, n_subs_layers, n_neurons_last_dense_layer, patientece=2, epochs=5, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        X_test = np.expand_dims(X_test, axis=-1)
        clf.predict(X_test)
        return True
    except Exception as e:
        print(e)
        return False


def run_regression_lstm(is_bidirectional):
    try:
        dataset = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv', header=0, index_col=0)
        dataset = dataset.head(750).copy()
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

    def test_lstm_multiclass_classification(self):
        self.assertEqual(run_regression_lstm(True), True)
        self.assertEqual(run_regression_lstm(False), True)

    def test_lstm_regression(self):
        self.assertEqual(run_multiclass_classification_lstm(), True)


if __name__ == '__main__':
    unittest.main()
