from game_env import env_2048
from a2c import a2c
import matplotlib.pyplot as plt
import pandas as pd
from take_sc import take_sc
import numpy as np

from networks.network import Network
from networks.activation_classes import *
from networks.optimizers import *
from networks.initializers import *
from networks.pools import *

from networks.ConvPoolLayer import ConvPoolLayer
from networks.DenseLayer import Dense

# A2C implementation of 2048
game = env_2048()

observation_space = game.observation_space  # int
action_space = len(game.action_space)  # int

# Value Network
network_Arguments = {
    "Layers": [
        ConvPoolLayer(
            features=20,
            kernel_width=2,
            stride=1,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
            input_dim=(11, 4, 4),
        ),
        ConvPoolLayer(
            features=20,
            kernel_width=2,
            stride=1,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=1,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
    ],
    "Step Size": 0.0001,
}

value_network = Network(network_Arguments)

# Policy Network
network_Arguments = {
    "Layers": [
        ConvPoolLayer(
            features=20,
            kernel_width=2,
            stride=1,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
            input_dim=(11, 4, 4),
        ),
        ConvPoolLayer(
            features=20,
            kernel_width=2,
            stride=1,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=64,
            activation=leaky_relu(0.3),
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
        Dense(
            size=action_space,
            activation=softmax,
            initializer=HeUniform,
            optimizer=Adam(0.9, 0.999),
        ),
    ],
    "Step Size": 0.00001,
}

policy_network = Network(network_Arguments)

reward_step = 0.01
gamma = 1
a2c = a2c(value_network, policy_network, game.action_space, reward_step, gamma)

top_score = 0
top_block = 0

block_history = []

performance = []

sc_path = "C:/Users/Daniil/ReinforcementLearning/latest.png"

for episode in range(1, 10000):

    # Initialize with start state
    game.reset_game()
    state = game.get_3d_feature()
    a2c.agent_init(state)
    invalid_actions = game.get_invalid_actions()

    # Play round
    total_reward = 0
    steps = 0

    while True:

        action = a2c.policy(state, invalid_actions)

        # Observe next state and reward
        terminal, reward = game.update_state(action)
        next_state = game.get_3d_feature()

        total_reward += reward

        if terminal:

            if episode % 100 == 0:
                take_sc(performance, sc_path)

            performance.append(total_reward)

            if total_reward > top_score:
                top_score = total_reward

            largest_block = max(game.grid.flatten())
            block_history.append(largest_block)

            if largest_block > top_block:
                top_block = largest_block

            # Recent frequency stats
            block_counts = {
                8: 0,
                16: 0,
                32: 0,
                64: 0,
                128: 0,
                256: 0,
                512: 0,
                1024: 0,
                2048: 0,
            }
            for block in block_history[-500:]:
                block_counts[block] += 1

            block_frequency = {
                k: round(v / min(500, len(block_history)), 3)
                for k, v in block_counts.items()
            }

            print(
                "Episode: "
                + str(episode)
                + " | Largest Block: "
                + str(largest_block)
                + " | Game Reward: "
                + str(total_reward)
                + " | Top Score: "
                + str(top_score)
                + " | Top Block: "
                + str(top_block)
            )
            print("Recent Frequency: " + str(block_frequency))

            a2c.agent_step(
                action=action,
                reward=reward,
                state=next_state,
                steps=steps,
                mask=invalid_actions,
                terminal=terminal,
            )

            break
        else:

            a2c.agent_step(
                action=action,
                reward=reward,
                state=next_state,
                steps=steps,
                mask=invalid_actions,
                terminal=terminal,
            )

        state = next_state
        invalid_actions = game.get_invalid_actions()
        steps += 1

# Convert to pandas Series
s = pd.Series(performance)
rolling_mean = s.rolling(window=50).mean()

plt.plot(np.arange(len(performance)), performance)
plt.plot(np.arange(len(performance)), rolling_mean)
plt.show()
