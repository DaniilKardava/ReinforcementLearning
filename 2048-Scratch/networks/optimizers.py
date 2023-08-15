import numpy as np
from copy import deepcopy
from .initializers import ZeroInitializer


class Adam:
    def __init__(self, first_beta, second_beta):
        self.first_beta = first_beta
        self.second_beta = second_beta

        self.first_correction = self.first_beta
        self.second_correction = self.second_beta

        self.epsilon = 1e-7

    def initialize_weights(self, dimensions):
        weights, biases = ZeroInitializer(dimensions, 0, 0)

        # Copy one to keep separate instances for each moment
        self.first_moment = {"Weights": deepcopy(weights), "Biases": deepcopy(biases)}
        self.second_moment = {"Weights": weights, "Biases": biases}

    def calculate_moments(self, weights_delta, biases_delta):
        """
        Copy, manipulate, and return updated weights and biases.
        """

        # Calculate moments and unbiased terms
        unbiased_first_moment = {}
        unbiased_second_moment = {}

        self.first_moment["Weights"] = (
            self.first_beta * self.first_moment["Weights"]
            + (1 - self.first_beta) * weights_delta
        )
        self.first_moment["Biases"] = (
            self.first_beta * self.first_moment["Biases"]
            + (1 - self.first_beta) * biases_delta
        )

        unbiased_first_moment["Weights"] = self.first_moment["Weights"] / (
            1 - self.first_correction
        )
        unbiased_first_moment["Biases"] = self.first_moment["Biases"] / (
            1 - self.first_correction
        )

        self.second_moment["Weights"] = self.second_beta * self.second_moment[
            "Weights"
        ] + (1 - self.second_beta) * np.square(weights_delta)
        self.second_moment["Biases"] = self.second_beta * self.second_moment[
            "Biases"
        ] + (1 - self.second_beta) * np.square(biases_delta)

        unbiased_second_moment["Weights"] = self.second_moment["Weights"] / (
            1 - self.second_correction
        )

        unbiased_second_moment["Biases"] = self.second_moment["Biases"] / (
            1 - self.second_correction
        )

        # Update bias correction:
        self.first_correction *= self.first_beta
        self.second_correction *= self.second_beta

        return unbiased_first_moment, unbiased_second_moment

    def update_weights(self, weights, biases, weights_delta, biases_delta, step_size):
        unbiased_first_moment, unbiased_second_moment = self.calculate_moments(
            weights_delta, biases_delta
        )

        weights -= step_size * (
            unbiased_first_moment["Weights"]
            / (np.sqrt(unbiased_second_moment["Weights"]) + self.epsilon)
        )
        biases -= step_size * (
            unbiased_first_moment["Biases"]
            / (np.sqrt(unbiased_second_moment["Biases"]) + self.epsilon)
        )


class SGD:
    @staticmethod
    def update_weights(weights, biases, weights_delta, biases_delta, step_size):
        for i in range(len(weights)):
            weights[i] -= step_size * weights_delta[i]
            biases[i] -= step_size * biases_delta[i]
