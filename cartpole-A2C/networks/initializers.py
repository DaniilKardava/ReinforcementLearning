import numpy as np


def ZeroInitializer(dimensions):
    weights = []
    biases = []
    for i in range(len(dimensions) - 1):

        fan_in = dimensions[i]
        fan_out = dimensions[i + 1]

        weights.append(
            np.zeros(
                (fan_out, fan_in),
            )
        )
        biases.append(np.zeros((fan_out, 1)))
    return weights, biases


def XavierNormal(dimensions):
    weights = []
    biases = []
    for i in range(len(dimensions) - 1):

        fan_in = dimensions[i]
        fan_out = dimensions[i + 1]

        weights.append(
            (
                np.random.normal(
                    0,
                    np.sqrt(2 / (fan_in + fan_out)),
                    (fan_out, fan_in),
                )
            )
        )
        biases.append(np.full((fan_out, 1), 0.01))
    return weights, biases


def HeUniform(dimensions):
    weights = []
    biases = []
    for i in range(len(dimensions) - 1):

        fan_in = dimensions[i]
        fan_out = dimensions[i + 1]

        weights.append(
            (
                np.random.uniform(
                    -np.sqrt(6.0 / fan_in),
                    np.sqrt(6.0 / fan_in),
                    (fan_out, fan_in),
                )
            )
        )
        biases.append(np.full((fan_out, 1), 0.01))
    return weights, biases
