from game_env import env_2048
from a2c import a2c
import matplotlib.pyplot as plt
import pandas as pd
from take_sc import take_sc
import numpy as np

import tensorflow as tf
import keras
from keras.models  import Model
from keras.layers import Input, Dense, Conv2D, Add, Flatten

# A2C implementation of 2048
game = env_2048()

observation_space = game.observation_space  # int
action_space = len(game.action_space)  # int

class ValueNetwork():
    def __init__(self):
        input_layer = Input(shape=(4,4,11))
        conv_layer = Conv2D(filters = 128, kernel_size = 2, activation = "relu", data_format = "channels_last")(input_layer)
        conv_layer = Conv2D(filters = 64, kernel_size = 2, activation = "relu", data_format = "channels_last")(conv_layer)
        flattened_layer = Flatten()(conv_layer)
        dense_layer = Dense(256, activation = "relu")(flattened_layer)
        output_layer = Dense(1, activation = "relu")(dense_layer)

        self.model = Model(inputs = input_layer, outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .0001)

        # Compile so the model can be saved. I do not intend to use the defined loss function.
        self.model.compile(optimizer=self.optimizer, loss = "mse")

    @tf.function
    def forward(self, one_input):
        return self.model(one_input)

    @tf.function
    def train(self, one_input, bootstrap):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            prediction = self.model(one_input)
            advantage = bootstrap - prediction
            loss = tf.square(advantage)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Return advantage to pass directly to policy
        return advantage

class PolicyNetwork():
    def __init__(self):
        input_layer = Input(shape = (4,4,11))
        input_mask = Input(shape = (4,))
        conv_layer = Conv2D(filters = 128, kernel_size = 2, activation = "relu", data_format = "channels_last")(input_layer)
        conv_layer = Conv2D(filters = 64, kernel_size = 2, activation = "relu", data_format = "channels_last")(conv_layer)
        flattened_layer = Flatten()(conv_layer)
        dense_layer = Dense(256, activation = "relu")(flattened_layer)
        logits_layer = Dense(4, activation = "linear")(dense_layer)
        masked_logits_layer = Add()([logits_layer, input_mask])
        output_layer = keras.layers.Activation("softmax")(masked_logits_layer)

        self.model = Model(inputs = [input_layer, input_mask], outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .00001)

        # Compile model so it can be saved. A custom loss function will be used below.
        self.model.compile(optimizer=self.optimizer, loss = "categorical_crossentropy")

    @tf.function
    def forward(self, one_input, mask):
        return self.model([one_input, mask])

    @tf.function
    def train(self,one_input, advantage, action_index, mask = np.ones((4,))):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            prediction = self.model([one_input, mask])
            chosen_action = prediction[0][action_index]
            loss = -advantage * tf.math.log(chosen_action) 

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
value_network = ValueNetwork()
policy_network = PolicyNetwork()

if True:
    # Import model
    saved_value_net = keras.models.load_model("saved_models/model final/value_network/")
    saved_policy_net = keras.models.load_model("saved_models/model final/policy_network/")

    # Set model
    value_network.model = saved_value_net
    policy_network.model = saved_policy_net

reward_step = 0.01
gamma = .9
a2c = a2c(value_network, policy_network, reward_step, gamma, False, False)

top_score = 0
top_block = 0

# Block history
block_history = []

# Score history
performance = []

sc_path = "C:/Users/danik/ReinforcementLearning/2048/latest.png"

saved_model_number = 1

for episode in range(1, 20000):

    # Initialize with start state
    game.reset_game()
    state = game.get_3d_feature()

    # Expand dims for batch size dimension
    state = np.expand_dims(state, axis =0)

    a2c.agent_init(state)
    invalid_actions = game.get_invalid_actions()

    # Expand dims for batch size dimension
    invalid_actions = np.expand_dims(invalid_actions, axis =0)

    # Play round
    total_reward = 0
    steps = 0

    while True:
        
        action = a2c.policy(state, invalid_actions)

        # Observe next state and reward
        terminal, reward = game.update_state(action)
        next_state = game.get_3d_feature()

        # Expand dims for batch size dimension
        next_state = np.expand_dims(next_state, axis =0)


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

            # Check if the model should be saved:
            if min(block_history[-3:]) >= 1024:
                value_network.model.save("saved_models/model "+str(saved_model_number)+"/value_network/")
                policy_network.model.save("saved_models/model "+str(saved_model_number)+"/policy_network/")
                saved_model_number += 1

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

        # Expand dims for batch size dimension
        invalid_actions = np.expand_dims(invalid_actions, axis =0)

        steps += 1

# Save last model:
value_network.model.save("saved_models/model final/value_network/")
policy_network.model.save("saved_models/model final/policy_network/")

# Convert to pandas Series
s = pd.Series(performance)
rolling_mean = s.rolling(window=50).mean()

plt.plot(np.arange(len(performance)), performance)
plt.plot(np.arange(len(performance)), rolling_mean)
plt.show()
