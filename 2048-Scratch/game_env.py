import numpy as np
from copy import deepcopy


class env_2048:
    def __init__(self):

        self.action_space = np.arange(4)

        # Initialize grid with 2 random starting blocks
        self.grid = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.random_square_spawn()
        self.random_square_spawn()

        self.observation_space = 176

    def neighboring_elements(self):
        # True if mergeable neighboring elements exist
        for row in range(4):
            for col in range(4):
                value = self.grid[row][col]
                if value == 0:
                    continue

                if row > 0 and self.grid[row - 1][col] == value:
                    return True

                if row < 3 and self.grid[row + 1][col] == value:
                    return True

                if col > 0 and self.grid[row][col - 1] == value:
                    return True

                if col < 3 and self.grid[row][col + 1] == value:
                    return True

        return False

    def random_square_spawn(self):

        open_squares = np.argwhere(self.grid == 0)

        # The function can be called any time. However, the program is intended to end by returning none when 1 empty square remains,
        # is filled, and neighboring_elements returns false. If there are no empty squares, but a valid action exists, this function
        # will not be called.

        assert len(open_squares) != 0

        # Probability of 4 or 2 block
        if np.random.rand() < 0.1:
            random_value = 4
        else:
            random_value = 2

        # Randomly choose location to fill from available squares
        spawn_location = open_squares[np.random.randint(0, len(open_squares))]

        self.grid[spawn_location[0]][spawn_location[1]] = random_value

        # Return True if grid is full and no legal actions exist
        if len(open_squares) == 1:
            if not self.neighboring_elements():
                return True

        return False

    def simulate_next_state(self, action):

        temporary_grid = deepcopy(self.grid)

        # A block can only merge once. Grid tracks merged blocks.
        merged_status = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )

        reward = 0

        # MOVE RIGHT
        if action == 0:
            for row in range(4):
                for col in range(3)[::-1]:
                    while col <= 2 and (
                        temporary_grid[row][col + 1] == 0
                        or (
                            temporary_grid[row][col + 1] == temporary_grid[row][col]
                            and not merged_status[row][col + 1]
                        )
                    ):
                        if temporary_grid[row][col + 1] == 0:
                            temporary_grid[row][col + 1] = temporary_grid[row][col]
                            temporary_grid[row][col] = 0
                        elif temporary_grid[row][col + 1] == temporary_grid[row][col]:
                            temporary_grid[row][col + 1] *= 2
                            temporary_grid[row][col] = 0
                            merged_status[row][col + 1] = True
                            reward += temporary_grid[row][col + 1]
                        col += 1

        # MOVE LEFT
        elif action == 1:
            for row in range(4):
                for col in range(1, 4):
                    while col >= 1 and (
                        temporary_grid[row][col - 1] == 0
                        or (
                            temporary_grid[row][col - 1] == temporary_grid[row][col]
                            and not merged_status[row][col - 1]
                        )
                    ):
                        if temporary_grid[row][col - 1] == 0:
                            temporary_grid[row][col - 1] = temporary_grid[row][col]
                            temporary_grid[row][col] = 0
                        elif temporary_grid[row][col - 1] == temporary_grid[row][col]:
                            temporary_grid[row][col - 1] *= 2
                            temporary_grid[row][col] = 0
                            merged_status[row][col - 1] = True
                            reward += temporary_grid[row][col - 1]
                        col -= 1

        # MOVE UP
        elif action == 2:
            for col in range(4):
                for row in range(1, 4):
                    while row >= 1 and (
                        temporary_grid[row - 1][col] == 0
                        or (
                            temporary_grid[row - 1][col] == temporary_grid[row][col]
                            and not merged_status[row - 1][col]
                        )
                    ):
                        if temporary_grid[row - 1][col] == 0:
                            temporary_grid[row - 1][col] = temporary_grid[row][col]
                            temporary_grid[row][col] = 0
                        elif temporary_grid[row - 1][col] == temporary_grid[row][col]:
                            temporary_grid[row - 1][col] *= 2
                            temporary_grid[row][col] = 0
                            merged_status[row - 1][col] = True
                            reward += temporary_grid[row - 1][col]
                        row -= 1

        # MOVE DOWN
        elif action == 3:
            for col in range(4):
                for row in range(3)[::-1]:
                    while row <= 2 and (
                        temporary_grid[row + 1][col] == 0
                        or (
                            temporary_grid[row + 1][col] == temporary_grid[row][col]
                            and not merged_status[row + 1][col]
                        )
                    ):

                        if temporary_grid[row + 1][col] == 0:
                            temporary_grid[row + 1][col] = temporary_grid[row][col]
                            temporary_grid[row][col] = 0
                        elif temporary_grid[row + 1][col] == temporary_grid[row][col]:
                            temporary_grid[row + 1][col] *= 2
                            temporary_grid[row][col] = 0
                            merged_status[row + 1][col] = True
                            reward += temporary_grid[row + 1][col]
                        row += 1

        return (temporary_grid, reward)

    def update_state(self, action):
        new_grid, reward = self.simulate_next_state(action)
        # Invalid actions should be masked
        assert not np.all(new_grid == self.grid)
        self.grid = new_grid
        return (self.random_square_spawn(), reward)

    def reset_game(self):
        # Initialize grid with 2 random starting blocks
        self.grid = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.random_square_spawn()
        self.random_square_spawn()

    def get_binary_feature(self):
        # Do not encode 0.
        grid = self.grid.flatten()
        feature_vector = np.zeros([16, 11])
        for i in range(len(grid)):
            if grid[i] == 0:
               continue
            power = len(bin(grid[i])) - 3
            feature_vector[i][-power] = 1

        return feature_vector.flatten().reshape(-1, 1)

    def get_linear_feature(self):
        grid = self.grid.flatten().reshape(-1, 1).astype("float64")
        grid /= 2024
        return grid

    def get_2d_feature(self):
        grid = self.grid / 2024
        return grid

    def get_3d_feature(self):
        # Calculate binary feature vector. Stack, aligning like terms columnwise and transpose to align like terms row wise. 
        # Reshape to combine like terms into channels. Flip to index small terms first. 
        feature_vector = self.get_binary_feature().reshape(16, 11)
        h_stacked_channels = np.transpose(feature_vector)
        kernel = h_stacked_channels.reshape(11, 4, 4)
        kernel = kernel[::-1]
        return kernel

    def get_invalid_actions(self):
        mask = []
        for i in range(4):
            new_grid, reward = self.simulate_next_state(i)
            if np.all(new_grid == self.grid):
                mask.append(i)
        return mask
