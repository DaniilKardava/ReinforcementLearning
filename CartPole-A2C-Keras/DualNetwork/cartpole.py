import gym
from a2c import a2c
import numpy as np 

import keras
import tensorflow as tf

from keras.models  import Model
from keras.layers import Input, Dense

env = gym.make("CartPole-v1")  # render_mode="rgb_array_list")

observation_space = env.observation_space.shape[0]  # int
action_space = env.action_space.n  # int


# Value Network
class ValueNetwork():
    def __init__(self):
        input_layer = Input(shape=(4, ))
        dense_layer = Dense(100, activation = "leaky_relu")(input_layer)
        dense_layer = Dense(50, activation = "leaky_relu")(dense_layer)
        output_layer = Dense(1, activation = "linear")(dense_layer)

        self.model = Model(inputs = input_layer, outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .00025)

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
        input_layer = Input(shape = (4,))
        dense_layer = Dense(100, activation = "leaky_relu")(input_layer)
        dense_layer = Dense(50, activation = "leaky_relu")(dense_layer)
        output_layer = Dense(2, activation = "softmax")(dense_layer)

        self.model = Model(inputs = input_layer, outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .000125)

    @tf.function
    def forward(self, one_input):
        return self.model(one_input)

    @tf.function
    def train(self,one_input, advantage, action_index, ):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            prediction = self.model(one_input)
            chosen_action = prediction[0][action_index]
            loss = -advantage * tf.math.log(chosen_action) 

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

value_network = ValueNetwork()
policy_network = PolicyNetwork()

avg_rew_step = 0.01
gamma = .9

a2c = a2c(value_network, policy_network, action_space, gamma, avg_rew_step)

for x in range(3000):
    state = env.reset()[0]
    state = state.reshape(-1, 1)

    state = np.expand_dims(state, axis = 0)

    a2c.agent_init(state)
    step = 0
    while True:
        step += 1
        action = a2c.policy(state)
        state_next, reward, terminal, info = env.step(action)[:-1]
        state_next = state_next.reshape(-1, 1)
    
        state_next = np.expand_dims(state_next, axis = 0)

        reward = reward if not terminal else -reward

        if terminal:
            a2c.agent_step(
                action=action, reward=reward, state=state_next, terminal=terminal
            )
            print(
                "Run: "
                + str(x)
                + ", score: "
                + str(step)
                + ", loss: "
                + str(a2c.avg_value_loss / 1)
            )
            break

        a2c.agent_step(action=action, reward=reward, state=state_next)
        state = state_next
