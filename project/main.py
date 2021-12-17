import numpy as np
from tensorflow import keras

from project.CNN import CNNClassification

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

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((-1, 28 * 28))
test_images = test_images.reshape((-1, 28 * 28))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

clf = CNNClassification(len(class_names), 10, [3])
clf.prepare(train_images, train_labels)
clf.compile()
clf.fit(train_images, train_labels)
