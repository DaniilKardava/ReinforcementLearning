import numpy as np


def ZeroInitializer(dimensions, fan_in, fan_out):
    # To make it work better with cnns, dimensions now accepts the actual expected shape of the weight matrix

    weights = np.zeros(
        dimensions,
    )

    biases = np.zeros((dimensions[0], 1))

    return weights, biases


def XavierNormal(dimensions, fan_in, fan_out):

    weights = np.random.normal(
        0,
        np.sqrt(2 / (fan_in + fan_out)),
        dimensions,
    )

    biases = np.full((dimensions[0], 1), 0.01)

    return weights, biases


def HeUniform(dimensions, fan_in, fan_out):

    weights = np.random.uniform(
        -np.sqrt(6.0 / fan_in),
        np.sqrt(6.0 / fan_in),
        dimensions,
    )

    biases = np.full((dimensions[0], 1), 0.01)

    return weights, biases
