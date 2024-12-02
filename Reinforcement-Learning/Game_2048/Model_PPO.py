import gym
from Environment import Game2048Env
from stable_baselines3 import PPO

# Create and wrap the environment
env = Game2048Env()

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_2048")