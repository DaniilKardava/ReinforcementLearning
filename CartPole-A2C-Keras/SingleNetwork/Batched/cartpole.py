import gym
from a2c import a2c
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime as dt
import matplotlib.cm as cm

import keras
import tensorflow as tf

from keras.models  import Model
from keras.layers import Input, Dense

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
    def train(self, inputs, targets, action_indices):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:

            actor_probs, critic_vals = self.model(inputs)

            # Calculate batch vf_loss
            advantage = targets - critic_vals
            vf_loss = tf.square(advantage)
            vf_loss = tf.reduce_mean(vf_loss) 

            # Index action probabilities and calculate batch pg_loss
            row_indices = tf.range(tf.shape(actor_probs)[0])
            chosen_actions = tf.gather_nd(actor_probs, tf.stack([row_indices, action_indices], axis=1))
            chosen_actions = tf.reshape(chosen_actions, [-1, 1])
            
            pg_loss = -tf.stop_gradient(advantage) * tf.math.log(chosen_actions)
            pg_loss = tf.reduce_mean(pg_loss) 

            ent_loss = -tf.reduce_mean(actor_probs * tf.math.log(actor_probs + 1e-9)) 
            
            global_loss = pg_loss + vf_loss * self.vf_coef - ent_loss * self.ent_coef

        gradients = tape.gradient(global_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
shared_network = SharedNetwork()

gamma = .9

_a2c_ = a2c(shared_network, action_space, gamma)

performance = []
speed = []

batch_size = 5

for x in range(1000):

    start_time = dt.now()

    batch_targets = []
    batch_states = []
    batch_actions = []

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

        # Keep history of game trajectory for batch. Remove batch dimension in state

        target = _a2c_.agent_step(reward=reward, state=state_next, terminal = terminal)

        batch_targets.append(target)
        batch_actions.append(action)
        batch_states.append(state[0])

        # ----- # 

        if terminal:
            
            _a2c_.agent_train(batch_states, batch_targets, batch_actions)

            print(
                "Run: "
                + str(x)
                + ", score: "
                + str(step)
            )

            performance.append(step)

            break

        # Perform a training step every "batch_size" steps
        if step != 0 and step % batch_size == 0:

            _a2c_.agent_train(batch_states, batch_targets, batch_actions)
            
            batch_targets = []
            batch_states = []
            batch_actions = []

        state = state_next
    
    end_time = dt.now()
    elapsed = end_time - start_time

    speed.append(elapsed.total_seconds())

colors = np.linspace(0, 1, len(performance[25:]))
colormap = cm.get_cmap('viridis')
plot_colors = colormap(colors)

plt.scatter(performance[25:], speed[25:], c=plot_colors)
plt.show()

plt.plot(np.arange(len(performance)), performance)
plt.show()
