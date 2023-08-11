from scipy import signal
import numpy as np
from collections import namedtuple
from copy import deepcopy
from datetime import datetime


class ConvPoolLayer:
    def __init__(
        self,
        features=None,
        kernel_width=None,
        stride=None,
        pooling_width=None,
        pooling_method=None,
        activation=None,
        initializer=None,
        optimizer=None,
        input_dim=None,
    ):
        # Input if applicable, first layer. If input is 2d, expand dimension.
        if input_dim != None:
            if len(input_dim) == 2:
                input_dim = (1, input_dim[0], input_dim[1])
        self.network_input_dim = input_dim

        self.output_features = features

        self.kernel_width = kernel_width
        self.stride = stride
        self.pooling_width = pooling_width
        self.pooling_method = pooling_method

        self.activation = activation
        self.initializer = initializer
        self.optimizer = optimizer

        self.forward_pass_data = None

    def build_network(self, input_dim):

        # Expect 3d input, # of feature maps, dimensions of features.
        weight_dims = (
            self.output_features,
            input_dim[0],
            self.kernel_width,
            self.kernel_width,
        )

        self.weights, self.biases = self.initializer(
            weight_dims, fan_in=self.kernel_width ** 2 * input_dim[0], fan_out=1
        )

        self.optimizer.initialize_weights(weight_dims)

        # Check the output shape of the convolutional/pool layer
        self.forward(np.zeros(input_dim))
        self.output_shape = self.forward_pass_data.pooled_layer.shape

        return self.output_shape

    def forward(self, input):
        # Expect same 3d input as above. If 2d network input, expand dimension.
        if len(input.shape) == 2:
            input = np.expand_dims(input, axis=0)

        # Correlate channels
        feature_maps = []
        for feature_out in range(self.output_features):
            # Correlate input and weight kernel to form feature map
            feature_map = signal.correlate(
                input, self.weights[feature_out, :, :, :], mode="valid"
            )
            feature_map = np.squeeze(feature_map, axis=0)

            bias_term = self.biases[feature_out]
            feature_map += bias_term

            feature_maps.append(feature_map)

        active_feature_maps = []
        for feature_map in feature_maps:

            # Activate feature map
            active_feature_map = self.activation.function(feature_map)
            active_feature_maps.append(active_feature_map)

        if self.pooling_method != None:
            pooled_layers = []
            max_indices = []
            for feature_map in feature_maps:
                # Pool features
                pooled_layer, indices = self.pooling_method.pool(
                    active_feature_map, self.pooling_width
                )
                pooled_layers.append(pooled_layer)
                max_indices.append(indices)
        else:
            pooled_layers = active_feature_maps
            max_indices = []

        output_tuple = namedtuple(
            "convolved_output",
            [
                "pooled_layer",
                "feature_maps",
                "activated_feature_maps",
                "max_indices",
            ],
        )

        self.forward_pass_data = output_tuple(
            np.array(pooled_layers),
            np.array(feature_maps),
            np.array(active_feature_maps),
            max_indices,
        )

    def backward(self, next_layer_error, next_layer_weights):
        if len(next_layer_error.shape) == 2:
            return self.backward_dense(next_layer_error, next_layer_weights)
        else:
            return self.backward_conv(next_layer_error, next_layer_weights)

    def backward_conv(self, next_layer_error, next_layer_weights):
        # Next layer error should be a list of 2d-matrices each representing a feature map.
        # Weights should be a list of kernels: number of output features, depth = # of pools, size, size.

        # Calculate pool layer error. Traverse each channel and sum net effects on all next layer output features.
        pooled_layer_error = []
        for pool_channel in range(self.output_shape[0]):
            feature_effects = []
            for feature_out in range(len(next_layer_error)):
                feature_effects.append(
                    signal.convolve2d(
                        next_layer_error[feature_out, :, :],
                        next_layer_weights[feature_out, pool_channel, :, :],
                        mode="full",
                    )
                )
            net_feature_effects = np.sum(feature_effects, axis=0)
            pooled_layer_error.append(net_feature_effects)

        # Calculate feature layer error
        feature_maps = self.forward_pass_data.feature_maps

        if self.pooling_method != None:
            max_indices = self.forward_pass_data.max_indices
            feature_map_size = feature_maps[0].shape[0]

            feature_layer_error = np.zeros_like(feature_maps)
            for feature_index in range(len(feature_maps)):

                num_indices = len(max_indices[feature_index])
                for map_index in range(num_indices):
                    # Check that max index did not come from padding.
                    index_2d = max_indices[feature_index][map_index]
                    if (
                        index_2d[0] >= feature_map_size
                        or index_2d[1] >= feature_map_size
                    ):
                        continue

                    feature_layer_error[feature_index][
                        index_2d[0], index_2d[1]
                    ] = pooled_layer_error[feature_index].flatten()[map_index]

        else:
            feature_layer_error = pooled_layer_error

        feature_layer_error *= self.activation.derivative(feature_maps)

        return feature_layer_error

    def backward_dense(self, next_layer_error, weights):

        feature_maps = self.forward_pass_data.feature_maps
        max_indices = self.forward_pass_data.max_indices
        feature_map_size = feature_maps[0].shape[0]

        # A vector shaped pool error layer
        pooled_layer_error = np.transpose(weights) @ next_layer_error

        # Calculate feature layer error
        if self.pooling_method != None:

            single_pool_length = pooled_layer_error.shape[0] // len(feature_maps)

            feature_layer_error = np.zeros_like(feature_maps)

            for feature_index in range(len(feature_maps)):
                # Index appropriate subsection of pool layer
                start_index = single_pool_length * feature_index
                corresponding_pool = pooled_layer_error[
                    start_index : start_index + single_pool_length
                ]

                num_indices = len(max_indices[feature_index])
                for map_index in range(num_indices):
                    # Check that max index did not come from padding.
                    index_2d = max_indices[feature_index][map_index]
                    if (
                        index_2d[0] >= feature_map_size
                        or index_2d[1] >= feature_map_size
                    ):
                        continue

                    feature_layer_error[feature_index][
                        index_2d[0], index_2d[1]
                    ] = corresponding_pool[map_index]
        else:
            # Reshape vector pool back into matrix
            pooled_layer_error = pooled_layer_error.reshape(
                self.output_features, feature_map_size, feature_map_size
            )
            feature_layer_error = pooled_layer_error

        feature_layer_error *= self.activation.derivative(feature_maps)

        return feature_layer_error

    def calculate_deltas(self, layer_error, prev_layer_nodes):

        # If previous layer are 2d network inputs add dimensions:
        if len(prev_layer_nodes.shape) == 2:
            prev_layer_nodes = np.expand_dims(prev_layer_nodes, axis=0)

        kernel_depth = len(prev_layer_nodes)
        output_features = len(layer_error)

        kernel_deltas = []
        bias_deltas = []
        # Calculate kernel errors for each feature output
        for feature in range(output_features):
            # Calculate channel wise errors
            channel_deltas = []
            for channel in range(kernel_depth):
                channel_error = signal.correlate2d(
                    prev_layer_nodes[channel, :, :],
                    layer_error[feature, :, :],
                    mode="valid",
                )
                channel_deltas.append(channel_error)
            kernel_deltas.append(np.array(channel_deltas))
            bias_deltas.append(np.sum(layer_error[feature, :, :]))

        return np.array(kernel_deltas), np.array(bias_deltas).reshape(-1, 1)

    def update_weights(self, kernel_deltas, bias_deltas, step_size):
        self.optimizer.update_weights(
            self.weights, self.biases, kernel_deltas, bias_deltas, step_size
        )

    def get_output(self):
        return deepcopy(self.forward_pass_data.pooled_layer)

    def get_weights(self):
        return deepcopy(self.weights), deepcopy(self.biases)
