import numpy as np


class Network:
    def __init__(
        self,
        arguments={
            "Layers": None,
            "Step Size": None,
        },
    ):

        self.layers = arguments["Layers"]

        self.step_size = arguments["Step Size"]

        # Initialize network weights and calculate input/output dimensions.
        input_shape = self.layers[0].network_input_dim
        for layer in self.layers:
            input_shape = layer.build_network(input_shape)

    def forward(self, a):

        for layer in self.layers:
            layer.forward(a)

            a = layer.get_output()
        return a

    def backward(self, network_input, advantage, cross_entropy_index=None):

        # First layer error
        last_layer = self.layers[-1]
        if cross_entropy_index == None:
            network_output = last_layer.get_output()
            z = last_layer.forward_pass_data.z
            activation_gradient = last_layer.activation.derivative(z)
        else:
            network_output = last_layer.get_output()[cross_entropy_index]
            z = last_layer.forward_pass_data.z
            activation_gradient = last_layer.activation.derivative(z)[
                cross_entropy_index
            ]

        z_gradient = advantage(network_output) * activation_gradient
        previous_layer = self.layers[-2]
        previous_ouput = previous_layer.get_output()
        weight_delta, bias_delta = last_layer.calculate_deltas(
            z_gradient, previous_ouput
        )
        last_layer.update_weights(weight_delta, bias_delta, self.step_size)

        # Previous layer errors
        num_layers = len(self.layers)
        for i in range(num_layers - 2, -1, -1):

            current_layer = self.layers[i]

            # Get weight connections, belonging to class of next layer
            next_layer = self.layers[i + 1]
            weights, biases = next_layer.get_weights()

            # Get previous nodes
            if i != 0:
                previous_layer = self.layers[i - 1]
                previous_nodes = previous_layer.get_output()
            else:
                previous_nodes = network_input

            z_gradient = current_layer.backward(z_gradient, weights)
            weight_delta, bias_delta = current_layer.calculate_deltas(
                z_gradient, previous_nodes
            )
            current_layer.update_weights(weight_delta, bias_delta, self.step_size)


    def backpropogate(self, network_input, advantage, cross_entropy_index=None):

        self.forward(network_input)

        return self.backward(
            network_input,
            advantage,
            cross_entropy_index,
        )

    def train(
        self, network_input, advantages, expected_output, cross_entropy_index=None
    ):

        for i in range(len(network_input)):

            self.backpropogate(
                network_input[i],
                advantages[i],
                cross_entropy_index,
            )

        return self.evaluate(network_input, expected_output)

    def evaluate(self, train_data, expected_output):
        correct = 0
        total_cost = 0
        for i in range(len(train_data)):
            # Calculate cost
            total_cost += sum((self.forward(train_data[i]) - expected_output[i]) ** 2)
            if np.argmax(self.forward(train_data[i])) == np.argmax(expected_output[i]):
                correct += 1
        return (correct / len(train_data), total_cost / len(train_data))
