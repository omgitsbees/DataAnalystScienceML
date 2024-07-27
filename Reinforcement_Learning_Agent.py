import tkinter as tk
from tkinter import messagebox
import gym
from stable_baselines3 import PPO

# Define the environment and RL agent
env = gym.make('CartPole-v1')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

def run_agent():
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()

# Create the GUI window
window = tk.Tk()
window.title("RL Agent GUI")

# Create a button to run the agent
run_button = tk.Button(window, text="Run Agent", command=run_agent)
run_button.pack(pady=20)

# Start the GUI event loop
window.mainloop()
