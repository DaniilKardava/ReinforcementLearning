import gym
from keras_a2c import a2c
from networks import policy_network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ENV_NAME = "CartPole-v1"
env = gym.make("CartPole-v1")  # , render_mode="human")

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Scratch networks
policy_network = policy_network([observation_space, 24, 24, action_space])

# I will replace the value function using a production level keras model:
value_network = Sequential()
value_network.add(Dense(units=24, activation="sigmoid", input_dim=observation_space))
value_network.add(Dense(units=24, activation="sigmoid"))
value_network.add(Dense(units=1, activation="softmax"))
value_network.compile(loss="mse", optimizer="adam")

avg_rew_step = 0.001
policy_step = 0.0001

hyperparameters = {
    "Policy Step": policy_step,
    "Reward Step": avg_rew_step,
}

a2c = a2c(value_network, policy_network, action_space, hyperparameters)

run = 0
while True:
    run += 1
    state = env.reset()[0]
    state = state.reshape(-1, 1)
    a2c.agent_init(state)
    step = 0
    while True:
        step += 1
        # env.render()
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
                + str(run)
                + ", score: "
                + str(step)
                + ", loss: "
                + str(a2c.avg_value_loss / 1)
            )
            break

        a2c.agent_step(action=action, reward=reward, state=state_next)
        state = state_next
