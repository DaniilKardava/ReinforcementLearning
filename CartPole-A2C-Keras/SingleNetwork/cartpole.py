import gym
from a2c import a2c
import numpy as np 
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

from keras.models  import Model
from keras.layers import Input, Dense, Conv2D, Add, Flatten

env = gym.make("CartPole-v1")  # render_mode="rgb_array_list")

observation_space = env.observation_space.shape[0]  # int
action_space = env.action_space.n  # int

class SharedNetwork():
    def __init__(self, learning_rate = .0007, vf_coef = .25, ent_coef = .01):

        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        input_layer = Input(shape = (4,))
        dense_layer = Dense(100, activation = "leaky_relu")(input_layer)
        dense_layer = Dense(50, activation = "leaky_relu")(dense_layer)
        actor = Dense(action_space, activation = "softmax")(dense_layer)
        critic = Dense(1, activation = "linear")(dense_layer)

        self.model = Model(inputs = input_layer, outputs = [actor, critic])

        self.optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate)

    @tf.function
    def forward(self, one_input):
        return self.model(one_input)

    @tf.function
    def train(self, one_input, target, action_index):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            actor_probs, critic_val = self.model(one_input)

            advantage = target - critic_val
            vf_loss = tf.square(advantage)

            # Treat advantage as a scalar
            chosen_action = actor_probs[0][action_index]
            pg_loss = -advantage * tf.math.log(chosen_action)
            
            ent_loss = -tf.reduce_sum(actor_probs * tf.math.log(actor_probs + 1e-9))

            global_loss = pg_loss + vf_loss * self.vf_coef + ent_loss * self.ent_coef

        gradients = tape.gradient(global_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Return advantage to pass directly to policy
        return advantage
    
shared_network = SharedNetwork()

gamma = .9

_a2c_ = a2c(shared_network, action_space, gamma)

performance = []

for x in range(500):
    state = env.reset()[0]
    state = state.reshape(-1, 1)

    state = np.expand_dims(state, axis = 0)

    _a2c_.agent_init(state)
    step = 0
    while True:
        step += 1
        action = _a2c_.policy(state)
        state_next, reward, terminal, info = env.step(action)[:-1]
        state_next = state_next.reshape(-1, 1)
    
        state_next = np.expand_dims(state_next, axis = 0)

        reward = reward if not terminal else -reward

        if terminal:
            _a2c_.agent_step(
                action=action, reward=reward, state=state_next, terminal=terminal
            )
            print(
                "Run: "
                + str(x)
                + ", score: "
                + str(step)
            )

            performance.append(step)

            break

        _a2c_.agent_step(action=action, reward=reward, state=state_next)
        state = state_next

plt.plot(np.arange(len(performance)), performance)
plt.show()