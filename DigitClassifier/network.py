import numpy as np
from activation_classes import *
from initializers import *
from collections import namedtuple


class Network:
    def __init__(
        self,
        arguments={
            "Dimensions": None,
            "Step Size": None,
            "Layer Activations": None,
            "Network Initializer": None,
            "Optimizer": None,
        },
    ):

        self.step_size = arguments["Step Size"]

        self.dimensions = arguments["Dimensions"]

        self.weights, self.biases = arguments["Network Initializer"](self.dimensions)

        # Create list of layer activations and their parameters for training.
        self.layer_activations = arguments["Layer Activations"]

        self.optimizer = arguments["Optimizer"]

    def forward(self, a):
        zs = []
        activated_nodes = [a]

        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            zs.append(z)
            a = self.layer_activations[i].function(z)
            activated_nodes.append(a)

        output_tuple = namedtuple("output_tuple", ["zs", "active_nodes", "output"])
        return output_tuple(zs, activated_nodes, a)

    def backward(self, zs, activated_nodes, advantage, cross_entropy_index=None):
        weight_delta = []
        bias_delta = []
        for i in range(len(self.weights)):
            weight_delta.append(np.zeros_like(self.weights[i]))
            bias_delta.append(np.zeros_like(self.biases[i]))

        if cross_entropy_index == None:
            network_output = activated_nodes[-1]
            activation_gradient = self.layer_activations[-1].derivative(zs[-1])
        else:
            network_output = activated_nodes[-1][cross_entropy_index]
            activation_gradient = self.layer_activations[-1].derivative(zs[-1])[
                cross_entropy_index
            ]

        z_gradient = advantage(network_output) * activation_gradient
        weight_delta[-1] = z_gradient @ np.transpose(activated_nodes[-2])
        bias_delta[-1] = z_gradient

        for i in range(1, len(self.dimensions) - 1):
            z_gradient = (
                np.transpose(self.weights[-i])
                @ z_gradient
                * self.layer_activations[-1 - i].derivative(zs[-1 - i])
            )
            weight_delta[-1 - i] = z_gradient @ np.transpose(activated_nodes[-2 - i])
            bias_delta[-1 - i] = z_gradient

        # Do not update, return gradients
        return (weight_delta, bias_delta)

    def backpropogate(self, network_input, advantage, cross_entropy_index=None):

        forward_dict = self.forward(network_input)

        return self.backward(
            forward_dict.zs,
            forward_dict.active_nodes,
            advantage,
            cross_entropy_index,
        )

    def train(
        self, network_input, advantages, expected_output, cross_entropy_index=None
    ):

        for i in range(len(network_input)):
            print(network_input[i].shape)
            print(advantages[i])
            weight_gradient, bias_gradient = self.backpropogate(
                network_input[i],
                advantages[i],
                cross_entropy_index,
            )

            print(weight_gradient[0].shape)
            # Update weights
            self.optimizer.update_weights(
                self.weights,
                self.biases,
                weight_gradient,
                bias_gradient,
                self.step_size,
            )

        return self.evaluate(network_input, expected_output)

    def evaluate(self, train_data, expected_output):
        correct = 0
        total_cost = 0
        for i in range(len(train_data)):
            # Calculate cost
            total_cost += sum(
                (self.forward(train_data[i]).output - expected_output[i]) ** 2
            )
            if np.argmax(self.forward(train_data[i]).output) == np.argmax(
                expected_output[i]
            ):
                correct += 1
        return (correct / len(train_data), total_cost / len(train_data))
