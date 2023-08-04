import numpy as np


class Network:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.weights = []
        self.biases = []
        for i in range(len(dimensions) - 1):
            self.weights.append(
                (np.random.rand(dimensions[i + 1], dimensions[i]))
                / dimensions[i] ** 0.5
            )
            self.biases.append(
                np.random.rand(dimensions[i + 1], 1),
            )


class value_network(Network):
    def feedforward(self, input):
        zs = []
        activations = [input]
        for i in range(len(self.weights)):
            z = self.weights[i] @ input + self.biases[i]
            zs.append(z)
            input = sigmoid(z)
            activations.append(input)

        network_output = activations[-1]

        return {"zs": zs, "Activations": activations, "Output": network_output}

    def backprop(self, zs, activations, td_error):

        weight_delta = []
        bias_delta = []
        for i in range(len(self.weights)):
            weight_delta.append(np.zeros_like(self.weights[i]))
            bias_delta.append(np.zeros_like(self.biases[i]))

        # First layer error wrt unactivated nodes (z). Remember that I am minimizing the td error, where the negative is carried outside after differentiation.
        z_gradient = -td_error * sigmoid_prime(zs[-1])
        weight_delta[-1] = z_gradient @ np.transpose(activations[-2])
        bias_delta[-1] = z_gradient

        for i in range(1, len(self.dimensions) - 1):
            z_gradient = (
                np.transpose(self.weights[-i]) @ z_gradient * sigmoid_prime(zs[-1 - i])
            )
            weight_delta[-1 - i] = z_gradient @ np.transpose(activations[-2 - i])
            bias_delta[-1 - i] = z_gradient

        # Do not update, return gradients
        return (weight_delta, bias_delta)

    def calc_gradient(self, network_input, td_error):

        feedforward_dict = self.feedforward(network_input)

        return self.backprop(
            feedforward_dict["zs"], feedforward_dict["Activations"], td_error
        )


class policy_network(Network):
    def feedforward(self, input):
        zs = []
        activations = [input]

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ input + self.biases[i]
            zs.append(z)
            input = sigmoid(z)
            activations.append(input)

        # Softmax on last layer
        z = self.weights[-1] @ input + self.biases[-1]
        zs.append(z)
        network_output = softmax(z)
        activations.append(network_output)

        return {"zs": zs, "Activations": activations, "Output": network_output}

    def backprop(self, zs, activations, td_error, action):

        weight_delta = []
        bias_delta = []
        for i in range(len(self.weights)):
            weight_delta.append(np.zeros_like(self.weights[i]))
            bias_delta.append(np.zeros_like(self.biases[i]))

        ohe = np.zeros_like(activations[-1])
        ohe[action] = 1

        # First layer error wrt unactivated nodes (z). For minimization, the loss is crossentropy * advantage. Crosssentropy gradient is given below.
        z_gradient = (td_error) * (activations[-1] - ohe)
        weight_delta[-1] = z_gradient @ np.transpose(activations[-2])
        bias_delta[-1] = z_gradient

        for i in range(1, len(self.dimensions) - 1):
            z_gradient = (
                np.transpose(self.weights[-i]) @ z_gradient * sigmoid_prime(zs[-1 - i])
            )
            weight_delta[-1 - i] = z_gradient @ np.transpose(activations[-2 - i])
            bias_delta[-1 - i] = z_gradient

        # Do not update, return gradients
        return (weight_delta, bias_delta)

    def calc_gradient(self, network_input, td_error, action):

        feedforward_dict = self.feedforward(network_input)

        return self.backprop(
            feedforward_dict["zs"], feedforward_dict["Activations"], td_error, action
        )


def softmax(preferences):
    c = max(preferences)
    numerator = np.exp(preferences - c)
    denominator = sum(numerator)
    return numerator / denominator


def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-input))


def sigmoid_prime(input):
    return sigmoid(input) * (1.0 - sigmoid(input))
