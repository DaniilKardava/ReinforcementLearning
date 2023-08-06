import gym
import pickle

from a2c import a2c
from networks.network import Network
from networks.activation_classes import *
from networks.optimizers import *
from networks.initializers import *

with open("saved_weights.pkl", "rb") as f:
    saved_policies = pickle.load(f)

env = gym.make("CartPole-v1", render_mode="human")

observation_space = env.observation_space.shape[0]  # int
action_space = env.action_space.n  # int


# Value Network
dimensions = [observation_space, 100, 50, 1]
step_size = 0.00025
activation_functions = [leaky_relu(0.3), leaky_relu(0.3), linear]
initializer = HeUniform
adam = Adam(0.9, 0.999, dimensions, ZeroInitializer)
arguments = {
    "Dimensions": dimensions,
    "Step Size": step_size,
    "Layer Activations": activation_functions,
    "Network Initializer": initializer,
    "Optimizer": adam,
}
value_network = Network(arguments)

# Policy Network
dimensions = [observation_space, 100, 50, action_space]
step_size = 0.000125
activation_functions = [leaky_relu(0.3), leaky_relu(0.3), softmax]
initializer = HeUniform
adam = Adam(0.9, 0.999, dimensions, ZeroInitializer)
arguments = {
    "Dimensions": dimensions,
    "Step Size": step_size,
    "Layer Activations": activation_functions,
    "Network Initializer": initializer,
    "Optimizer": adam,
}

policy_network = Network(arguments)

avg_rew_step = 0.01
a2c = a2c(value_network, policy_network, action_space, avg_rew_step)

model_number = 167
for x in range(10):

    policy_network.biases = saved_policies[model_number]["Biases"]
    policy_network.weights = saved_policies[model_number]["Weights"]

    state = env.reset()[0]
    state = state.reshape(-1, 1)
    a2c.agent_init(state)
    step = 0
    while True:
        step += 1
        action = a2c.policy(state)
        state_next, reward, terminal, info = env.step(action)[:-1]
        state_next = state_next.reshape(-1, 1)
        reward = reward if not terminal else -reward

        if terminal:
            print("Policy Number: " + str(model_number) + ", score: " + str(step))
            break

        state = state_next
