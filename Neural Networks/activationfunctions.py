import numpy as np


class Activations:

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x: np.ndarray):
        z = np.exp(x)
        m = np.sum(z)
        return z / m

    def relu(self, x: np.ndarray):
        return np.maximum(0, x)

    def leaky_relu(self, x: np.ndarray):
        return np.maximum(0.01 * x, x)

    def tanhx(self, x: np.ndarray):
        return np.tanh(x)

    def der_sigmoid(self, x: np.ndarray):
        s = self.sigmoid(x)
        return s * (1 - s)

    def der_relu(self, x: np.ndarray):
        return (x > 0).astype(float)

    def der_leaky_relu(sekf, x: np.ndarray):
        x[x >= 0] = 1.00
        x[x <= 0] = 0.01
        return x

    def der_tanhx(self, x: np.ndarray):
        return (1 / np.cosh(x)) ** 2
