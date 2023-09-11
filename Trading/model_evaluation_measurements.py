from agenv.a2c import a2c
import matplotlib.pyplot as plt
import pandas as pd
from utils.take_sc import take_sc
import numpy as np
from finta import TA
import scipy
import statistics

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

from agenv.trading_env_2_2_1 import trading_env

# 1373000
prices, stacked_features = format_data(window_size, 390, 666000)
env = trading_env(window_size= window_size, price_data=prices, feature_data = stacked_features)

observation_space = env.observation_space  # int
action_space = env.action_space  # int

shared_network = SharedNetwork()

# Load keras network:
saved_model = keras.models.load_model("saved_models/sum_log_return/two_actions/fixed_bounds/legacy_rms/long.short/action_obs/no_gumbel/ent_.0005/lr_.0007/model 3250000/")
shared_network.model = saved_model

_a2c_ = a2c(shared_network, gamma, apply_gamma=True)

sc_path = "images/evaluation_rewards.png"
cumulative_rewards = []
episode_rewards = []

# Statistics gathering:

# Correlate reward and action probabilities
rew_v_conf_long = {"rewards" : [], "average_confidence" : [], "entry_confidence": [], "exit_confidence": []}
rew_v_conf_short = {"rewards" : [], "average_confidence" : [], "entry_confidence": [], "exit_confidence": []}

rew_v_length_long = {"rewards" : [], "length" : []}
rew_v_length_short = {"rewards" : [], "length" : []}

tod_contribution = []
tod_short_contribution = []
tod_long_contribution = []
for i in range(390):
    tod_contribution.append([])
    tod_short_contribution.append([])
    tod_long_contribution.append([])

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

        next_obs, reward, terminal = env.next_obs(action)

        
        # Measure the cumulative return for each individual minute
        price_delta = (env.price_data[env.data_index] - env.price_data[env.data_index - 1]) / env.price_data[env.data_index - 1] 
        reshaped_obs = np.squeeze(next_obs).reshape(window_size, -1)
        time = np.interp(reshaped_obs[-1][4], [-1,1], [0,389])
        time = int(time + .5)
        if action == env.long:
            tod_contribution[time].append(price_delta)
            tod_long_contribution[time].append(price_delta)
        elif action == env.short:
            tod_contribution[time].append(-price_delta)
            tod_short_contribution[time].append(-price_delta)

        # Correlate probability and rewards
        if reward != 0:
            try:
                current_confidence = action_prob[action]
                average_confidence = (previous_confidence + current_confidence) / 2

                if action == env.long:
                    rew_v_conf_long["average_confidence"].append(average_confidence)
                    rew_v_conf_long["entry_confidence"].append(previous_confidence)
                    rew_v_conf_long["exit_confidence"].append(current_confidence)

                    rew_v_conf_long["rewards"].append(reward)

                    rew_v_length_long["rewards"].append(reward)
                    rew_v_length_long["length"].append(steps - last_trade_step)

                elif action == env.short:
                    rew_v_conf_short["average_confidence"].append(average_confidence)
                    rew_v_conf_short["entry_confidence"].append(previous_confidence)
                    rew_v_conf_short["exit_confidence"].append(current_confidence)

                    rew_v_conf_short["rewards"].append(reward)

                    rew_v_length_short["rewards"].append(reward)
                    rew_v_length_short["length"].append(steps - last_trade_step)

                previous_confidence = current_confidence
                last_trade_step = steps
            except:
                previous_confidence = action_prob[action]
                last_trade_step = steps

        # Expand dims for batch size dimension
        next_obs = np.expand_dims(next_obs, axis =0)

        cumulative_rewards.append(np.exp(reward))

        # Gather the observations along the timeseries form.

        if terminal:
            
            break


        obs = next_obs

env.render()

plt.title("Long Profits")
plt.plot(np.arange(len(env.profit_history["Long"])), env.profit_history["Long"])
plt.show()

plt.title("Short Profits")
plt.plot(np.arange(len(env.profit_history["Short"])), env.profit_history["Short"])
plt.show()

plt.title("Profits")
plt.plot(np.arange(len(env.profit_history["Total"])), env.profit_history["Total"])
plt.show()


# Concert list of lists of minute returns into np arrays:
tod_mean = [np.mean(lst) for lst in tod_contribution]
tod_short_mean = [np.mean(lst) for lst in tod_short_contribution]
tod_long_mean = [np.mean(lst) for lst in tod_long_contribution]

plt.title("Time of Day Contributions")
plt.bar(np.arange(390), tod_mean)
plt.show()

plt.title("Time of Day Short Contributions")
plt.bar(np.arange(390), tod_short_mean)
plt.show()

plt.title("Time of Day Long Contributions")
plt.bar(np.arange(390), tod_long_mean)
plt.show()
# ---- # 

# First minute aggregate performance:
plt.title("Opening Short")
plt.plot(np.cumprod(np.array(tod_short_contribution[0]) + 1))
plt.show()

plt.title("Opening Long")
plt.plot(np.cumprod(np.array(tod_long_contribution[0]) + 1))
plt.show()
# ---- # 

print("LONG TRADE STATISTICS")
print("<------------------>")

# Correlate rewards and average probabilities of the entry and exit
corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_long["average_confidence"], rew_v_conf_long["rewards"])
print("Correlation Between Rewards and Average Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Average Probability v Reward (LONG)")
plt.scatter(rew_v_conf_long["average_confidence"], rew_v_conf_long["rewards"])
plt.show()

print("")

corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_long["entry_confidence"], rew_v_conf_long["rewards"])
print("Correlation Between Rewards and Entry Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Entry Probability v Reward (LONG)")
plt.scatter(rew_v_conf_long["entry_confidence"], rew_v_conf_long["rewards"])
plt.show()

print("")

corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_long["exit_confidence"], rew_v_conf_long["rewards"])
print("Correlation Between Rewards and Exit Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Exit Probability v Reward (LONG)")
plt.scatter(rew_v_conf_long["exit_confidence"], rew_v_conf_long["rewards"])
plt.show()

print("")

# Analyze long trade reward distribution:
stat, p_two_sided = scipy.stats.ttest_1samp(rew_v_conf_long["rewards"], 0)
print("Long trade mean: " + str(statistics.mean(rew_v_conf_long["rewards"])))
print("Long trade std-dev: " + str(statistics.pstdev(rew_v_conf_long["rewards"])))
print("Two sided significance of mean != 0: " + str(p_two_sided))

print("")

# Analyze relationship between trade length and return
corr_coef, p_value = scipy.stats.pearsonr(rew_v_length_long["length"], rew_v_length_long["rewards"])
print("Correlation Between Rewards and Trade Length: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))
plt.title("Trade Length v Reward")
plt.scatter(rew_v_length_long["length"], rew_v_length_long["rewards"])
plt.show()

print("")

print("SHORT TRADE STATISTICS")
print("<------------------>")

# Correlate rewards and average probabilities of the entry and exit
corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_short["average_confidence"], rew_v_conf_short["rewards"])
print("Correlation Between Rewards and Average Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Average Probability v Reward (SHORT)")
plt.scatter(rew_v_conf_short["average_confidence"], rew_v_conf_short["rewards"])
plt.show()

print("")

corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_short["entry_confidence"], rew_v_conf_short["rewards"])
print("Correlation Between Rewards and Entry Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Entry Probability v Reward (SHORT)")
plt.scatter(rew_v_conf_short["entry_confidence"], rew_v_conf_short["rewards"])
plt.show()

print("")

corr_coef, p_value = scipy.stats.pearsonr(rew_v_conf_short["exit_confidence"], rew_v_conf_short["rewards"])
print("Correlation Between Rewards and Exit Action Probabilities: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))

plt.title("Exit Probability v Reward (SHORT)")
plt.scatter(rew_v_conf_short["exit_confidence"], rew_v_conf_short["rewards"])
plt.show()

# Analyze short trade reward distribution:
stat, p_two_sided = scipy.stats.ttest_1samp(rew_v_conf_short["rewards"], 0)
print("Short trade mean: " + str(statistics.mean(rew_v_conf_short["rewards"])))
print("Short trade std-dev: " + str(statistics.pstdev(rew_v_conf_short["rewards"])))
print("Two sided significance of mean != 0: " + str(p_two_sided))

# Analyze relationship between trade length and return
corr_coef, p_value = scipy.stats.pearsonr(rew_v_length_short["length"], rew_v_length_short["rewards"])
print("Correlation Between Rewards and Trade Length: " + str(corr_coef))
print("Significance of relationship: "  + str(p_value))
plt.title("Trade Length v Reward")
plt.scatter(rew_v_length_short["length"], rew_v_length_short["rewards"])
plt.show()
