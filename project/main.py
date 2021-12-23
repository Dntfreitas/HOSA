import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from project.CNN import CNNClassification, CNNRegression

g_max = 4
#
o_max = 35
#
n_start = 50
n_step = 50
n_max = 150
#
m_start = 4
m_max = 7
#
mul_max = 2
#
size = 100
X = np.random.rand(size, 100)
y = np.random.randint(0, 2, size)
#
epsilon = 0.01


# hosa_cnn(X, y, g_max, o_max, n_start, n_step, n_max, m_start, m_max, mul_max, epsilon)

def test_multiclass():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((-1, 28 * 28))
    test_images = test_images.reshape((-1, 28 * 28))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    clf = CNNClassification(len(class_names), 10, [3])
    clf.prepare(train_images, train_labels)
    clf.compile()
    clf.fit(train_images, train_labels)


def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = CNNClassification(2, 10, [3], patientece=10)
    clf.prepare(X_train, y_train)
    clf.compile()
    clf.fit(X_train, y_train)


def test_regression():
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = CNNRegression(1, 10, [3], patientece=2, epochs=200)
    clf.prepare(X_train, y_train)
    clf.compile()
    clf.fit(X_train, y_train)


test_multiclass()
test_regression()
