from agenv.a2c import a2c
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from finta import TA


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
    rsi = np.interp(rsi, (0, 100), (-1,1))

    sma = TA.SMA(price_data, 200).to_numpy()
    sma_displacement = (sma / price_data["Close"].to_numpy())[warm_up:]
    sma_displacement = np.interp(sma_displacement, (.97, 1.03), (-1,1))


    macd_feature = TA.MACD(price_data)[warm_up:]

    macd = macd_feature["MACD"].to_numpy()
    macd = np.interp(macd, (-2, 2), (-1,1))

    signal = macd_feature["SIGNAL"].to_numpy()
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
    def __init__(self):

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


    @tf.function
    def forward(self, one_input, hidden_state):
        return self.model(inputs = [one_input, hidden_state])

# Hyperparams
window_size = 30
gamma = .99
path = "sb_weights/win_30_model_weights_reordered.json"

from agenv.trading_env_2_2_2 import trading_env

# prices, stacked_features = format_data(window_size, 666000, 1373000)
prices, stacked_features = format_data(window_size, 666000, 1373000)
env = trading_env(window_size= window_size, price_data=prices, feature_data = stacked_features)

observation_space = env.observation_space  # int
action_space = env.action_space  # int

shared_network = SharedNetwork()

# Load keras network:
saved_model = keras.models.load_model("saved_models/sum_log_return/two_actions/fixed_bounds/legacy_rms/long.short/action_obs/no_gumbel/ent_.01/spread_.999975/model 4500000/")
# saved_model = keras.models.load_model("saved_models/s/model 2250000/")
shared_network.model = saved_model

_a2c_ = a2c(shared_network, gamma, apply_gamma=True)

sc_path = "images/evaluation_rewards.png"
cumulative_rewards = []
episode_rewards = []

long_equity = []
short_equity = []

trades = 0
for cycle in range(1, 2):

    # Initialize with start state
    env.reset()
    obs = env.get_feature()

    # Expand dims for batch size dimension
    obs = np.expand_dims(obs, axis =0)

    # Empty memory state at the start of the episode
    hidden_state = [np.zeros((1,256)), np.zeros((1,256))]

    # Cycle once through data
    steps = 0

    while True:

        steps += 1    

        if steps % 10000 == 0:
            print("TIMESTEPS: " + str(steps))

        # Make prediction using model.
        action, action_prob, hidden_state, value = _a2c_.policy(obs, hidden_state)

        # print(action_prob)
        action = np.argmax(action_prob)

        trade = False
        if action != env.prev_action:
            trades += 1
            trade = True

        next_obs, reward, terminal = env.next_obs(action)

        if trade:
            if action == env.long:
                short_equity.append(reward)
            elif action == env.short:
                long_equity.append(reward)

        # Expand dims for batch size dimension
        next_obs = np.expand_dims(next_obs, axis =0)

        cumulative_rewards.append(np.exp(reward))
    
        # Gather the observations along the timeseries form.

        if terminal:
            
            break

        obs = next_obs

print("Trades: " + str(trades))
env.render()

plt.title("Long Rewards")
plt.plot(np.arange(len(long_equity)), np.cumsum(long_equity))
plt.show()

plt.title("Short Rewards")
plt.plot(np.arange(len(short_equity)), np.cumsum(short_equity))
plt.show()

plt.title("Long Profits")
plt.plot(np.arange(len(env.profit_history["Long"])), env.profit_history["Long"])
plt.show()

plt.title("Short Profits")
plt.plot(np.arange(len(env.profit_history["Short"])), env.profit_history["Short"])
plt.show()

plt.title("Profits")
plt.plot(np.arange(len(env.profit_history["Total"])), env.profit_history["Total"])
plt.show()
