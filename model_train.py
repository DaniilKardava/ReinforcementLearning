from agenv.a2c import a2c
import matplotlib.pyplot as plt
import pandas as pd
from utils.take_sc import take_sc
import numpy as np
from finta import TA
import copy 
import json
from datetime import datetime

import tensorflow as tf
import keras
from keras.models  import Model
from keras.layers import Input, Dense, LSTM
from keras.initializers import Orthogonal

tf.keras.backend.set_floatx('float64')

def format_data(window_size, start_index, stop_index):

    # Import data:
    price_data = pd.read_csv("spy_data/spy_padded.csv")
    price_data.set_index("Date", inplace=True)
    price_data.index = pd.to_datetime(price_data.index)

    price_data = price_data[:stop_index]

    # Make space for lookback period. Includes current point, so subtract 1.
    warm_up = start_index - (window_size - 1)


    rsi = TA.RSI(price_data).to_numpy()[warm_up:]
    # rsi = np.interp(rsi, (min(rsi), max(rsi)), (-1,1))
    rsi = np.interp(rsi, (0, 100), (-1,1))

    sma = TA.SMA(price_data, 200).to_numpy()
    sma_displacement = (sma / price_data["Close"].to_numpy())[warm_up:]
    # sma_displacement = np.interp(sma_displacement, (min(sma_displacement), max(sma_displacement)), (-1,1))
    sma_displacement = np.interp(sma_displacement, (.97, 1.03), (-1,1))


    macd_feature = TA.MACD(price_data)[warm_up:]

    macd = macd_feature["MACD"].to_numpy()
    # macd = np.interp(macd, (min(macd), max(macd)), (-1,1))
    macd = np.interp(macd, (-2, 2), (-1,1))

    signal = macd_feature["SIGNAL"].to_numpy()
    # signal = np.interp(signal, (min(signal), max(signal)), (-1,1))
    signal = np.interp(signal, (-2, 2), (-1,1))

    time_series = (price_data.index.hour * 60 +
                        price_data.index.minute) - 570
    time = time_series.to_numpy()[warm_up:]
    time = np.interp(time, (0, 389), (-1,1))

    features = [rsi, sma_displacement, macd, signal, time]
    stacked_features = np.column_stack(features)

    prices = price_data["Close"].to_numpy()[warm_up:]

    return prices, stacked_features


class SharedNetwork():
    def __init__(self, learning_rate = .0007, vf_coef = .25, ent_coef = .01, rho = .99, epsilon = 1e-5, global_clip = .5):

        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.rho = rho
        self.epsilon = epsilon
        self.global_clip = global_clip

        input_layer = Input(shape=(1, window_size * observation_space))
        initial_h = Input(shape=(256,))
        initial_c = Input(shape=(256,))
        dense_layer = Dense(units = 64, activation = "tanh", kernel_initializer=Orthogonal(gain=np.sqrt(2)))(input_layer)
        dense_layer = Dense(units = 64, activation = "tanh", kernel_initializer=Orthogonal(gain = np.sqrt(2)))(dense_layer)
        lstm_layer, state_h, state_c = LSTM(units = 256, return_state = True, return_sequences=True, kernel_initializer=Orthogonal, recurrent_initializer=Orthogonal)(dense_layer, initial_state=[initial_h, initial_c])
        logits = Dense(action_space, activation="linear", kernel_initializer=Orthogonal)(lstm_layer)
        actor = keras.layers.Activation("softmax")(logits)
        critic = Dense(1, activation = "linear", kernel_initializer=Orthogonal)(lstm_layer)

        self.model = Model(inputs = [input_layer, [initial_h, initial_c]], outputs = [actor, critic, logits, [state_h, state_c]])

        self.optimizer = keras.optimizers.legacy.RMSprop(learning_rate=self.learning_rate, rho = self.rho, epsilon = self.epsilon, global_clipnorm = self.global_clip)

        # Compile so the model can be saved. I do not intend to use the defined loss function.
        self.model.compile(optimizer=self.optimizer, loss = "mse")

    @tf.function
    def forward(self, one_input, hidden_state):
        return self.model(inputs = [one_input, hidden_state])

    @tf.function
    def train(self, inputs, targets, action_indices, hidden_state_initial):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            
            # Pass list of observations gathered in batch as a timeseries and gather outputs on each step.
            actor_probs, critic_vals, _,  _ = self.model(inputs = [inputs, hidden_state_initial])

            actor_probs = tf.cast(actor_probs[0], dtype = tf.float64)
            critic_vals = tf.cast(critic_vals[0], dtype = tf.float64)

            # Calculate batch vf_loss
            advantage = targets - critic_vals
            vf_loss = tf.square(advantage)
            ret_vf_loss = vf_loss
            vf_loss = tf.reduce_mean(vf_loss) 

            # Index action probabilities and calculate batch pg_loss
            row_indices = tf.range(tf.shape(actor_probs)[0])
            chosen_actions = tf.gather_nd(actor_probs, tf.stack([row_indices, action_indices], axis=1))
            chosen_actions = tf.reshape(chosen_actions, [-1, 1])
            
            pg_loss = -tf.stop_gradient(advantage) * tf.math.log(chosen_actions)
            ret_pg_loss = pg_loss
            pg_loss = tf.reduce_mean(pg_loss) 

            # Perform elementwise entropy calculations
            batch_entropy_components = -actor_probs * tf.math.log(actor_probs + 1e-9)
            batch_entropies = tf.reduce_sum(batch_entropy_components, axis = 1)
            ret_ent_loss = batch_entropies
            ent_loss = tf.reduce_mean(batch_entropies)

            global_loss = pg_loss + vf_loss * self.vf_coef - ent_loss * self.ent_coef

        gradients = tape.gradient(global_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return ret_vf_loss, ret_pg_loss, ret_ent_loss

def use_json(path, network):
    with open(path, "r") as f:
        params = json.load(f)

    for key, value in params.items():
        params[key] = np.array(value)

    keys = list(params.keys())

    trainable_variables = network.model.trainable_variables

    for i in range(len(keys)):
        data  = params[keys[i]]
        trainable_variables[i].assign(data)
        
# Hyperparams
window_size = 30
gamma = .99
batch_size = 5
lr = .0007
vf_coef = .25
ent_coef = .01
rho = .99
epsilon = 1e-5
global_clip = .5
from agenv.trading_env_2_2_2 import trading_env

prices, stacked_features = format_data(window_size, 390, 666000)

env = trading_env(window_size= window_size, price_data=prices, feature_data = stacked_features)

observation_space = env.observation_space  # int
action_space = env.action_space  # int

shared_network = SharedNetwork(learning_rate=lr, vf_coef=vf_coef, ent_coef=ent_coef, rho = rho, epsilon=epsilon, global_clip=global_clip)

_a2c_ = a2c(shared_network, gamma, apply_gamma=True)

cumulative_rewards = []
episode_rewards = []


# path = "sb_weights/win_30_model_weights_reordered.json"
# use_json(path, shared_network)

total_steps = 0
start = datetime.now()
num_features = window_size * observation_space
for cycle in range(1, 100):        

    batch_rewards = []
    batch_obs = []
    batch_actions = []

    # Initialize with start state
    env.reset()
    obs = env.get_feature()
    
    # Expand dims for batch size dimension
    obs = np.expand_dims(obs, axis =0)

    # Empty memory state at the start of the episode
    hidden_state = [np.zeros((1,256)), np.zeros((1,256))]
    # The first hidden state for train calculations
    hidden_state_train = [np.zeros((1,256)), np.zeros((1,256))]

    # Cycle once through data
    total_reward = 0
    steps = 0

    terminal = False

    while True:
                
        steps += 1
        total_steps += 1

        if total_steps % 10000 == 0:
            end = datetime.now()
            print(end - start)
            print("TIMESTEPS: " + str(total_steps))
            start = end
            ema_beta = .99
            take_sc(_a2c_.vf_hist, ema_beta, "log", "images/vf_loss_1.png")
            take_sc(_a2c_.pg_hist, ema_beta, "symlog", "images/pg_loss_1.png")
            take_sc(_a2c_.ent_hist, ema_beta, "log","images/ent_loss_1.png")
            

        if total_steps % 250000 == 0:
            print("Model Saved")
            shared_network.model.save("saved_models/spread_.999925/model "+str(total_steps)+"/")
            
        # Make prediction using model.
        action, action_prob, hidden_state, value = _a2c_.policy(obs, hidden_state)

        next_obs, reward, terminal = env.next_obs(action)

        # Expand dims for batch size dimension
        next_obs = np.expand_dims(next_obs, axis =0)
        
        total_reward += reward
        cumulative_rewards.append(reward)

        batch_rewards.append(reward)
        batch_actions.append(action)
        batch_obs.append(obs[0][0])

        if terminal:
            
            batch_targets = _a2c_.n_bootstrap(batch_rewards)

            # Reshape observations into timeseries, observation format and add batch dimension:
            batch_obs = np.array(batch_obs).reshape(-1, num_features)
            batch_obs = np.expand_dims(batch_obs, axis = 0)

            _a2c_.agent_train(batch_obs, batch_targets, batch_actions, hidden_state_train)
            episode_rewards.append(total_reward)

            break


        # Perform a training step every "batch_size" steps
        if steps != 0 and steps % batch_size == 0:
            
            # Bootstrap next state:
            _, _, _, value = _a2c_.policy(next_obs, hidden_state)
            

            # Append the bootstrapped value to list of rewards and perform n-step backward loop to calculate bootstrapped discounted
            # rewards. Exclude the added value
            batch_targets = _a2c_.n_bootstrap(batch_rewards + [value])[:-1]

            # Reshape observations into timeseries, observation format and add batch dimension:
            batch_obs = np.array(batch_obs).reshape(batch_size, num_features)
            batch_obs = np.expand_dims(batch_obs, axis = 0)

            _a2c_.agent_train(batch_obs, batch_targets, batch_actions, hidden_state_train)
            
            batch_rewards = []
            batch_obs = []
            batch_actions = []

            hidden_state_train = copy.deepcopy(hidden_state)

        obs = next_obs


plt.plot(np.arange(len(cumulative_rewards)), np.cumsum(cumulative_rewards))
plt.show()

plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.show()