import numpy as np
import tensorflow as tf

class a2c:
    def __init__(self, shared_network, gamma, apply_gamma):

        self.gamma = gamma

        self.apply_gamma = apply_gamma

        self.shared_network = shared_network

        self.avg_value_loss = 0

        self.vf_hist = []
        self.pg_hist = []
        self.ent_hist = []


    def n_bootstrap(self, rewards):
        # Formatting nuances inside
        returns = []
        ret = 0
        for reward in rewards[::-1]:
            ret = reward + self.gamma * ret
            ret = np.float32(np.array(ret).reshape(1))
            returns.append(ret)

        return returns[::-1]

    def agent_train(self, batch_obs, batch_targets, batch_actions, hidden_state_initial):
                
        # Convert everything to tensor:

        batch_obs = tf.constant(batch_obs, dtype=tf.float64)
        batch_targets = tf.constant(batch_targets, dtype=tf.float64)
        batch_actions = tf.constant(batch_actions)

        hidden_state_initial = [tf.constant(arr) for arr in hidden_state_initial]
        
        vf, pg, ent = self.shared_network.train(batch_obs, batch_targets, batch_actions, hidden_state_initial)

        self.vf_hist.extend(vf.numpy().flatten().tolist())
        self.pg_hist.extend(pg.numpy().flatten().tolist())
        self.ent_hist.extend(ent.numpy().flatten().tolist())

    def gumbel_sample(self, logits):
        uniform = tf.random.uniform(tf.shape(logits))
        noise = tf.cast(tf.math.log(-tf.math.log(uniform)), dtype = tf.float64)
        distribution = tf.squeeze(logits - noise) 
        return tf.argmax(distribution)
    
    def policy(self, obs, hidden_state):

        action_prob, value, logits, hidden_state = self.shared_network.forward(obs, hidden_state)
        action_prob = np.squeeze(action_prob)
        action = np.random.choice(len(action_prob), p=action_prob)
        # action = self.gumbel_sample(logits)

        return action, action_prob, hidden_state, value
    