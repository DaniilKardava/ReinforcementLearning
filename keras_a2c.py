import numpy as np


class a2c:
    def __init__(
        self,
        value_network,
        policy_network,
        actions,
        hyperparameters={
            "Policy Step": 0.0001,
            "Reward Step": 0.001,
        },
    ):

        self.policy_step = hyperparameters["Policy Step"]
        self.reward_step = hyperparameters["Reward Step"]

        self.value_network = value_network
        self.policy_network = policy_network

        self.avg_reward = 0
        self.avg_value_loss = 0

        self.actions = actions

        self.last_state = None

    def agent_init(self, state):
        self.last_state = state
        self.avg_value_loss = 0

    def agent_step(self, action, reward, state, terminal=False):

        # Expand dimension to represent batch of 1
        self.last_state = np.expand_dims(self.last_state, axis=0)
        state = np.expand_dims(state, axis=0)

        if terminal:
            td_error = (
                reward - self.avg_reward - self.value_network.predict(self.last_state)
            )
        else:
            td_error = (
                reward
                - self.avg_reward
                + self.value_network.predict(state)
                - self.value_network.predict(self.last_state)
            )

        if terminal:
            self.avg_value_loss += td_error ** 2

        # Update value network weights. The mse loss follows true - pred form. Therefore true value should be first terms of the td_error.
        if terminal:
            next_state_value = 0
        else:
            next_state_value = self.value_network.predict(state)

        self.value_network.fit(
            self.last_state, reward - self.avg_reward + next_state_value
        )

        # Contract dimesions after using keras
        self.last_state = self.last_state[0]
        state = state[0]

        # Update policy network weights
        # print("Action Taken: " + str(action))
        # print("Reward: " + str(reward))
        # print("Advantage: " + str(td_error))
        # print("Before")
        # print(self.policy_network.feedforward(self.last_state)["Output"])
        policy_weight_d, policy_bias_d = self.policy_network.calc_gradient(
            self.last_state, td_error, action
        )

        for i in range(len(policy_weight_d)):
            self.policy_network.weights[i] -= self.policy_step * policy_weight_d[i]
            self.policy_network.biases[i] -= self.policy_step * policy_bias_d[i]

        # print("After")
        # print(self.policy_network.feedforward(self.last_state)["Output"])

        # Update average reward:
        self.avg_reward += self.reward_step * (td_error)

        # Update state
        self.last_state = state

    def policy(self, state):

        action_prob = self.policy_network.feedforward(state)["Output"]
        action = np.random.choice(len(action_prob), p=action_prob.ravel())

        return action
