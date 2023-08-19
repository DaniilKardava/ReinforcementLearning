import numpy as np


class a2c:
    def __init__(self, value_network, policy_network, reward_step, gamma, apply_avg_rew, apply_gamma):

        self.reward_step = reward_step
        self.gamma = gamma

        self.apply_avg_rew = apply_avg_rew
        self.apply_gamma = apply_gamma

        self.value_network = value_network
        self.policy_network = policy_network

        self.avg_reward = 0
        self.avg_value_loss = 0

        self.last_state = None

    def agent_init(self, state):
        self.last_state = state

    def agent_step(self, action, reward, state, steps, mask=np.ones((4,)), terminal=False):

        if terminal:
            # Make compatible with tensor type
            bootstrap = np.float32(reward) 
        else:
            bootstrap = reward - (self.avg_reward * self.apply_avg_rew)  +  (self.gamma ** self.apply_gamma) *  self.value_network.forward(state)

        # Drop all tensor variable properties and treat as constant.
        bootstrap = np.float32(bootstrap)
        
        advantage = self.value_network.train(self.last_state, bootstrap)

        # Update average reward:
        self.avg_reward += self.reward_step * (advantage)

        advantage *= (self.gamma ** self.apply_gamma) ** steps

        # Drop all tensor variable properties and treat as constant.
        advantage = np.float32(advantage)

        self.policy_network.train(self.last_state, advantage, action, mask)

        # Update state
        self.last_state = state

    def policy(self, state, mask=np.ones((4,))):
        
        action_prob = self.policy_network.forward(state, mask)
        action = np.random.choice(4, p=np.squeeze(action_prob))

        return action
