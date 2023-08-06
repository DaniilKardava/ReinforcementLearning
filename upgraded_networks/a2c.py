import numpy as np


class a2c:
    def __init__(
        self,
        value_network,
        policy_network,
        actions,
        hyperparameters={
            "Reward Step": 0.001,
        },
    ):

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

        if terminal:
            td_error = reward - self.avg_reward
            advantage = td_error - self.value_network.forward(self.last_state).output
        else:
            td_error = (
                reward - self.avg_reward + self.value_network.forward(state).output
            )
            advantage = td_error - self.value_network.forward(self.last_state).output

        if terminal:
            self.avg_value_loss += advantage ** 2

        # Construct expanded loss function
        x = [lambda y: -advantage]

        self.value_network.train([self.last_state], x)

        # Construct expanded loss function
        x = [lambda y: -advantage / y]

        self.policy_network.train([self.last_state], x, action)

        # Update average reward:
        self.avg_reward += self.reward_step * (advantage)

        # Update state
        self.last_state = state

    def policy(self, state):

        action_prob = self.policy_network.forward(state).output
        action = np.random.choice(len(action_prob.ravel()), p=action_prob.ravel())

        return action
