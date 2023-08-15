import numpy as np


class a2c:
    def __init__(self, value_network, policy_network, actions, reward_step, gamma):

        self.reward_step = reward_step
        self.gamma = gamma

        self.value_network = value_network
        self.policy_network = policy_network

        self.avg_reward = 0
        self.avg_value_loss = 0

        self.actions = actions

        self.last_state = None

    def agent_init(self, state):
        self.last_state = state

    def agent_step(self, action, reward, state, steps, mask=None, terminal=False):

        if terminal:
            advantage = (
                reward - self.value_network.forward(self.last_state)
            )
        else:
            advantage = (
                reward
                + self.gamma * self.value_network.forward(state)
                - self.value_network.forward(self.last_state)
            )

        # Construct expanded loss function
        x = [lambda y: -advantage]

        self.value_network.train([self.last_state], x)

        # Construct expanded loss function
        x = [lambda y: -(self.gamma ** steps) * advantage / y]

        self.policy_network.train([self.last_state], x, action, mask, logits=True)

        # Update average reward:
        self.avg_reward += self.reward_step * (advantage)

        # Update state
        self.last_state = state

    def policy(self, state, mask=None):

        action_prob = self.policy_network.forward(state, mask, logits=True)
        action = np.random.choice(len(action_prob.ravel()), p=action_prob.ravel())

        return action
