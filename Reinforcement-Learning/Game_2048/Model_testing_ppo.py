from Environment import Game2048Env
from stable_baselines3 import PPO
import pygame

env = Game2048Env()
model = PPO.load("ppo_2048")

# Evaluate the trained agent
obs = env.reset()
done = False

# Loop until the game is done
while not done:
    # Predict the action based on the current observation
    action, _states = model.predict(obs)

    # Step the environment forward using the action
    obs, reward, done, info = env.step(action)

    # Render the environment (visualize the game)
    if __name__ == "__main__":
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            env.render()
            pygame.time.delay(100)



print("Game Over!")

