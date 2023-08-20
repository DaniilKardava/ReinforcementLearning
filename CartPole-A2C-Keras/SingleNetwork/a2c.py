import numpy as np


class a2c:
    def __init__(
        self,
        shared_network,
        actions,
        gamma,

    ):

        self.gamma = gamma

        self.shared_network = shared_network

        self.avg_reward = 0

        self.actions = actions

        self.last_state = None

    def agent_init(self, state):
        self.last_state = state

    def agent_step(self, action, reward, state, terminal=False):

        if terminal:
            target = reward 
        else:
            actor_probs, boostrapped_state = self.shared_network.forward(state)
            target = (
                reward  + self.gamma * boostrapped_state
            )


        target = np.float32(target)

        self.shared_network.train(self.last_state, target, action)

        # Update state
        self.last_state = state

    def policy(self, state):

        action_prob, actor_val = self.shared_network.forward(state)

        action = np.random.choice(2, p=np.squeeze(action_prob))

        return action
