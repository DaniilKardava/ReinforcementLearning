import gym
from a2c import a2c
from networks.network import Network
from networks.activation_classes import *
from networks.optimizers import *
from networks.initializers import *


env = gym.make("CartPole-v1")  # render_mode="rgb_array_list")

observation_space = env.observation_space.shape[0]  # int
action_space = env.action_space.n  # int


# Value Network
dimensions = [observation_space, 24, 12, 1]
step_size = 0.0005
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
dimensions = [observation_space, 24, 12, action_space]
step_size = 0.00025
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

hyperparameters = {
    "Reward Step": avg_rew_step,
}

a2c = a2c(value_network, policy_network, action_space, hyperparameters)

for x in range(3000):
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
            a2c.agent_step(
                action=action, reward=reward, state=state_next, terminal=terminal
            )
            print(
                "Run: "
                + str(x)
                + ", score: "
                + str(step)
                + ", loss: "
                + str(a2c.avg_value_loss / 1)
            )
            break

        a2c.agent_step(action=action, reward=reward, state=state_next)
        state = state_next
