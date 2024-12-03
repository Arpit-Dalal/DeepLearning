import numpy as np
from keras.models import load_model
from Environment import Game2048Env

# Load the trained model
model = load_model("dqn_2048.h5")

# Initialize the game environment
env = Game2048Env()


def test_model(model, env, episodes=10):
    total_score = 0
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 16])
        done = False
        score = 0

        print(f"Episode {episode + 1}/{episodes}")

        while not done:
            # Choose action based on model's prediction
            action = np.argmax(model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 16])
            state = next_state
            score += reward

            env.render()

        total_score += score
        print(f"Score: {score}")

    average_score = total_score / episodes
    print(f"Average Score over {episodes} episodes: {average_score}")


# Run the test
test_model(model, env)