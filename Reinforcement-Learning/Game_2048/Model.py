import gym
from Enviroment import Game2048Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# Create and wrap the environment
env = Game2048Env()
env = DummyVecEnv([lambda: env])

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_2048")

model = PPO.load("ppo_2048")  # If you want to load the model again, otherwise remove this line

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
    env.render()

    time.sleep(0.05)



print("Game Over!")

