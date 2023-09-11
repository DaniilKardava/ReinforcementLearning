import numpy as np
import matplotlib.pyplot as plt

'''
V2.2 trading env class-
Modifications: Rewards now include spreads.

'''

class trading_env:
    def __init__(self, window_size, price_data, feature_data):

        # 0 Short, 1 Long
        self.action_space = 2
        self.short = 0
        self.long = 1
        self.neutral = None

        self.observation_space = feature_data.shape[1] + 1
        self.window_size = window_size
        self.data_index = window_size - 1

        self.prev_action = 0
        self.price_data = price_data

        # Change entry price to none if starting with neutral position
        self.entry_price = self.price_data[self.data_index]

        self.feature_data = feature_data

        self.position_history = ["red"]
        self.action_history = self.window_size * [self.prev_action]

        self.profit = 1
        self.long_profit = 1
        self.short_profit = 1

        self.profit_history = {"Long": [], "Short": [], "Total" : []}

        self.spread = .999975

    def reset(self):
        
        self.prev_action = 0

        self.position_history = ["red"]
        self.action_history = self.window_size * [self.prev_action]

        self.data_index = self.window_size - 1

        # Change entry price to none if starting with neutral position
        self.entry_price = self.price_data[self.data_index]

        self.profit = 1
        self.long_profit = 1
        self.short_profit = 1

        self.profit_history = {"Long": [], "Short": [], "Total" : []}

    def calculate_reward(self, action):
        if action == self.prev_action:
            return 0
        
        if self.prev_action == self.short:
            exit_price = self.price_data[self.data_index]
            reward = (2 * self.entry_price - exit_price) / self.entry_price

            self.profit *= reward
            self.short_profit *= reward

            self.entry_price = self.price_data[self.data_index]
        elif self.prev_action == self.long:
            exit_price = self.price_data[self.data_index]
            reward = exit_price / self.entry_price

            self.profit *= reward
            self.long_profit *= reward

            self.entry_price = self.price_data[self.data_index]
        
        reward = np.log(reward + 1e-9) + np.log(self.spread)
        
        return reward

    def get_feature(self):
        # Observations are not fed as timeseries. Window variable does not control unrollment of the network, but the individual
        # observation itself.
        feature_vector = self.feature_data[self.data_index - self.window_size + 1: self.data_index + 1, :]
        prev_actions = np.array(self.action_history[-self.window_size:]).reshape(-1,1)

        feature_vector = np.hstack((feature_vector,prev_actions))
        feature_vector = feature_vector.flatten()

        feature_vector = np.expand_dims(feature_vector, axis = 0)
        
        return feature_vector
        
    def next_obs(self, action):
        
        if action == 0:
            self.position_history.append("red")
        else:
            self.position_history.append("green")

        self.data_index += 1

        # Normalize for feature vector
        self.action_history.append(action / (self.action_space - 1))

        feature_vector = self.get_feature()
        reward = self.calculate_reward(action)
        
        self.profit_history["Total"].append(self.profit)
        self.profit_history["Long"].append(self.long_profit)
        self.profit_history["Short"].append(self.short_profit)

        self.prev_action = action

        terminal = False
        if self.data_index == self.feature_data.shape[0] - 1:
            terminal = True

        return feature_vector, reward, terminal

    
    def render(self):
        plt.plot(np.arange(len(self.price_data[self.window_size - 1:])), self.price_data[self.window_size - 1:])
        plt.scatter(np.arange(len(self.price_data[self.window_size - 1:])), self.price_data[self.window_size - 1:], c = self.position_history, s = 15)
        plt.show()


        



    
    

