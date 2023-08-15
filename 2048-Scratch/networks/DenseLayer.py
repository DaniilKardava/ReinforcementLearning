import numpy as np
from copy import deepcopy
from collections import namedtuple


class Dense:
    def __init__(
        self,
        size=None,
        initializer=None,
        optimizer=None,
        activation=None,
        input_dim=None,
    ):

        # Input if applicable, first layer.
        self.network_input_dim = input_dim

        self.initializer = initializer
        self.optimizer = optimizer
        self.activation = activation

        self.size = size

        self.forward_pass_data = None

    def build_network(self, input_dim):

        input_length = np.prod(input_dim)

        # Pass desired shape of weight matrix
        weight_dims = (self.size, input_length)
        self.weights, self.biases = self.initializer(
            weight_dims, fan_in=input_length, fan_out=self.size
        )

        # Initialize adam optimizer
        self.optimizer.initialize_weights(weight_dims)

        # Generate forward pass data
        self.forward(np.zeros(input_dim))
        output_shape = self.forward_pass_data.output.shape

        return output_shape

    def forward(self, input, mask=None, logits=None):

        # If input is convolutional pools, flatten into vector
        if len(input.shape) == 3:
            input = input.flatten().reshape(-1, 1)

        z = self.weights @ input + self.biases

        # If output layer, if apply mask.
        if mask != None:
            if logits:
                z[mask] = -np.inf
                output = self.activation.function(z)
            else:
                output = self.activation.function(z)
                output[mask] = 0
        else:
            output = self.activation.function(z)

        output_tuple = namedtuple("output_tuple", ["output", "z"])

        self.forward_pass_data = output_tuple(output, z)

    def backward(self, next_layer_error, next_layer_weights):
        z = self.forward_pass_data.z
        layer_error = (
            np.transpose(next_layer_weights)
            @ next_layer_error
            * self.activation.derivative(z)
        )
        return layer_error

    def calculate_deltas(self, layer_error, prev_layer_nodes):
        # If input is convolutional pools, flatten into vector
        if len(prev_layer_nodes.shape) == 3:
            prev_layer_nodes = prev_layer_nodes.flatten().reshape(-1, 1)

        weights_delta = layer_error @ np.transpose(prev_layer_nodes)
        bias_delta = layer_error

        return weights_delta, bias_delta

    def update_weights(self, weight_deltas, bias_deltas, step_size):
        self.optimizer.update_weights(
            self.weights, self.biases, weight_deltas, bias_deltas, step_size
        )

    def get_output(self):
        return deepcopy(self.forward_pass_data.output)

    def get_weights(self):
        return deepcopy(self.weights), deepcopy(self.biases)
