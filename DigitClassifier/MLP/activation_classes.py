import numpy as np


class softmax:
    """
    Softmax derivative returns jacobian. Should be indexed by the corresponding one hot encoded position.
    """

    @staticmethod
    def function(logits):
        c = max(logits)
        numerator = np.exp(logits - c)
        denominator = sum(numerator)
        return numerator / denominator

    @staticmethod
    def derivative(logits):
        probabilities = softmax.function(logits)
        s = probabilities.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        return jacobian


class sigmoid:
    @staticmethod
    def function(arr):
        return 1.0 / (1.0 + np.exp(-arr))

    @staticmethod
    def derivative(arr):
        return sigmoid.function(arr) * (1.0 - sigmoid.function(arr))


class relu:
    @staticmethod
    def function(arr):
        activated = np.clip(arr, a_min=0, a_max=None)
        return activated

    @staticmethod
    def derivative(arr):
        prime = (arr >= 0).astype(float)
        return prime


class leaky_relu:
    """
    Activations with parameters influencing general behavior should have an instance with those unique parameters.
    """

    def __init__(self, slope):
        self.slope = slope

    def function(self, arr):
        arr_copy = arr.copy()
        arr_copy[arr_copy < 0] *= self.slope
        return arr_copy

    def derivative(self, arr):
        arr_copy = arr.copy()
        arr_copy[arr_copy > 0] = 1
        arr_copy[arr_copy <= 0] = self.slope
        return arr_copy


class linear:
    @staticmethod
    def function(arr):
        return arr

    @staticmethod
    def derivative(arr):
        prime = np.ones_like(arr)
        return prime
