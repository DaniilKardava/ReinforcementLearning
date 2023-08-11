import numpy as np
from copy import deepcopy
from skimage.measure import block_reduce

class MaxPooling:
    @staticmethod
    def pool(grid, window_size, padding=True):

        grid = deepcopy(grid)

        pad_layers = window_size - grid.shape[0] % window_size

        if pad_layers != 0:
            if padding:

                grid = np.pad(
                    grid,
                    ((0, pad_layers), (0, pad_layers)),
                    mode="constant",
                    constant_values=0,
                )

        pooled = block_reduce(grid, block_size=(window_size, window_size), func=np.max)

        max_indices = []

        # Find max inidices
        for max in pooled.flatten():
            loc = np.where(grid == max)

            # In case there are exact duplicates, don't believe its worth the trouble to keep track of occurances and assign correclty.
            if len(loc[0]) > 1:
                max_indices.append((loc[0][0], loc[1][0]))
            else:
                max_indices.append((loc[0], loc[1]))

        return pooled, max_indices
