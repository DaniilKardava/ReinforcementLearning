import numpy as np
import tensorflow as tf

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

    def agent_step(self, reward, state, terminal):

        if terminal:
            target = np.float32(np.array(reward).reshape(1))
            
        else:
            actor_probs, boostrapped_state = self.shared_network.forward(state)
            target = (
                reward  + self.gamma * boostrapped_state
            )
            target = np.float32(target[0])


        # Update state
        self.last_state = state

        return target

    def agent_train(self, batch_states, btach_targets, batch_actions):
                
        # Convert everything to tensor:

        batch_states = tf.constant(batch_states)
        btach_targets = tf.constant(btach_targets)
        batch_actions = tf.constant(batch_actions)

        self.shared_network.train(batch_states, btach_targets, batch_actions)

        

    def policy(self, state):

        action_prob, actor_val = self.shared_network.forward(state)

        action = np.random.choice(2, p=np.squeeze(action_prob))

        return action
