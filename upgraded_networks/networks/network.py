import numpy as np
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
            activation_gradient = activation_gradient.reshape(-1, 1)

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

        feedforward_dict = self.forward(network_input)

        return self.backward(
            feedforward_dict.zs,
            feedforward_dict.active_nodes,
            advantage,
            cross_entropy_index,
        )

    def train(self, network_input, advantages, cross_entropy_index=None):

        for i in range(len(network_input)):

            weight_gradient, bias_gradient = self.backpropogate(
                network_input[i],
                advantages[i],
                cross_entropy_index,
            )

            # Update weights
            self.optimizer.update_weights(
                self.weights,
                self.biases,
                weight_gradient,
                bias_gradient,
                self.step_size,
            )
