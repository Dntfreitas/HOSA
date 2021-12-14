import numpy as np

from CNNClassification import CNN
from optimize_cnn import hosa_cnn

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

hosa_cnn(X, y, g_max, o_max, n_start, n_step, n_max, m_start, m_max, mul_max, epsilon)

CNN()
