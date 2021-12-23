import unittest

from tensorflow import keras

from project.CNN import CNNClassification


def run_binary_classification():
    try:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = train_images.reshape((-1, 28 * 28))
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        clf = CNNClassification(len(class_names), 10, [3], verbose=0)
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels)
        clf.predict(test_images)
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
        clf = CNNClassification(len(class_names), 10, [3])
        clf.prepare(train_images, train_labels)
        clf.compile()
        clf.fit(train_images, train_labels)
        clf.predict(test_images)
        return True
    except:
        return False


class CNNTest(unittest.TestCase):

    def test_cnn_binary_classification(self):
        self.assertEqual(run_binary_classification(), True)

    def test_cnn_multiclass_classification(self):
        self.assertEqual(run_multiclass_classification(), True)


if __name__ == '__main__':
    unittest.main()
