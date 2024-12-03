import numpy as np
import gym
from Environment import Game2048Env
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

# Set parameters
EPISODES = 10
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount rate
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Create and wrap the environment
env = Game2048Env()


# Define the neural network model
def create_model(input_shape, action_space):
    model = Sequential()
    model.add(Dense(256, input_dim=input_shape, activation='relu'))  # Make sure input_shape is correct
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


input_shape = 16
action_space = env.action_space.n
model = create_model(input_shape, action_space)

def choose_action(state):
    if np.random.rand() <= EPSILON:
        return random.randrange(action_space)
    q_values = model.predict(state)
    return np.argmax(q_values[0])  # Exploit learned values


# Training loop
scores = []
for e in range(EPISODES):
    state = env.reset()

    # Reshape state correctly for ANN input (1 sample with 16 features)
    state = np.reshape(state, [1, 16])
    total_reward = 0

    while True:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1, 16])

        target = reward + GAMMA * np.max(model.predict(next_state)[0]) * (1 - done)
        target_f = model.predict(state)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)

    # Reduce exploration rate as training progresses
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Print score after each episode
    print(f"Episode: {e + 1}/{EPISODES}, Score: {total_reward}")

# Save the trained model
model.save("dqn_2048.h5")