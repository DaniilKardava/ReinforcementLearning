import numpy as np


class a2c:
    def __init__(
        self,
        value_network,
        policy_network,
        actions,
        gamma,
        reward_step,

    ):

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
        self.avg_value_loss = 0

    def agent_step(self, action, reward, state, terminal=False):

        if terminal:
            bootstrap = reward 
        else:
            bootstrap = (
                reward  + self.gamma * self.value_network.forward(state)
            )


        bootstrap = np.float32(bootstrap)

        advantage = self.value_network.train(self.last_state, bootstrap)

        advantage = np.float32(advantage)

        self.policy_network.train(self.last_state, advantage, action)

        # Update average reward:
        self.avg_reward += self.reward_step * (advantage)

        # Update state
        self.last_state = state

    def policy(self, state):

        action_prob = self.policy_network.forward(state)

        action = np.random.choice(2, p=np.squeeze(action_prob))

        return action
