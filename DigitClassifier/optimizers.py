import numpy as np
from copy import deepcopy

class Adam:
    def __init__(self, first_beta, second_beta, dimensions, initializer):
        self.first_beta = first_beta
        self.second_beta = second_beta

        weights, biases = initializer(dimensions)

        # Copy one to keep separate instances for each moment
        self.first_moment = {"Weights": deepcopy(weights), "Biases": deepcopy(biases)}
        self.second_moment = {"Weights": weights, "Biases": biases}

        self.first_correction = self.first_beta
        self.second_correction = self.second_beta

        self.epsilon = 1e-7

    def calculate_moments(self, weights_delta, biases_delta):
        """
        Copy, manipulate, and return updated weights and biases.
        """

        # Calculate moments and unbiased terms
        unbiased_first_moment = {"Weights": [], "Biases": []}
        unbiased_second_moment = {"Weights": [], "Biases": []}

        for i in range(len(weights_delta)):
            # Calculate first moments
            self.first_moment["Weights"][i] = (
                self.first_beta * self.first_moment["Weights"][i]
                + (1 - self.first_beta) * weights_delta[i]
            )
            self.first_moment["Biases"][i] = (
                self.first_beta * self.first_moment["Biases"][i]
                + (1 - self.first_beta) * biases_delta[i]
            )

            unbiased_first_moment["Weights"].append(
                self.first_moment["Weights"][i] / (1 - self.first_correction)
            )
            unbiased_first_moment["Biases"].append(
                self.first_moment["Biases"][i] / (1 - self.first_correction)
            )

            # Calculate second moments
            self.second_moment["Weights"][i] = self.second_beta * self.second_moment[
                "Weights"
            ][i] + (1 - self.second_beta) * np.square(weights_delta[i])
            self.second_moment["Biases"][i] = self.second_beta * self.second_moment[
                "Biases"
            ][i] + (1 - self.second_beta) * np.square(biases_delta[i])

            unbiased_second_moment["Weights"].append(
                self.second_moment["Weights"][i] / (1 - self.second_correction)
            )
            unbiased_second_moment["Biases"].append(
                self.second_moment["Biases"][i] / (1 - self.second_correction)
            )

        # Update bias correction:
        self.first_correction *= self.first_beta
        self.second_correction *= self.second_beta

        return unbiased_first_moment, unbiased_second_moment

    def update_weights(self, weights, biases, weights_delta, biases_delta, step_size):
        unbiased_first_moment, unbiased_second_moment = self.calculate_moments(
            weights_delta, biases_delta
        )
        for i in range(len(weights)):
            weights[i] -= step_size * (
                unbiased_first_moment["Weights"][i]
                / (np.sqrt(unbiased_second_moment["Weights"][i]) + self.epsilon)
            )
            biases[i] -= step_size * (
                unbiased_first_moment["Biases"][i]
                / (np.sqrt(unbiased_second_moment["Biases"][i]) + self.epsilon)
            )


class SGD:
    @staticmethod
    def update_weights(weights, biases, weights_delta, biases_delta, step_size):
        for i in range(len(weights)):
            weights[i] -= step_size * weights_delta[i]
            biases[i] -= step_size * biases_delta[i]
