import unittest

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from project.CNN import CNNClassification, CNNRegression


def run_binary_classification():
    try:
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNClassification(2, 10, [3], patientece=10, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return True
    except:
        return False


def run_multiclass_classification():
    try:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = train_images.reshape((-1, 28 * 28))
        test_images = test_images.reshape((-1, 28 * 28))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        clf = CNNClassification(len(class_names), 10, [3], verbose=0)
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels)
        clf.predict(test_images)
        return True
    except:
        return False


def run_regression():
    try:
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = CNNRegression(1, 10, [3], patientece=2, epochs=200, verbose=0)
        clf.prepare(X_train, y_train)
        clf.compile()
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        return True
    except:
        return False


class CNNTest(unittest.TestCase):

    def test_cnn_binary_classification(self):
        self.assertEqual(run_binary_classification(), True)

    def test_cnn_multiclass_classification(self):
        self.assertEqual(run_multiclass_classification(), True)

    def test_cnn_regression(self):
        self.assertEqual(run_regression(), True)


if __name__ == '__main__':
    unittest.main()
