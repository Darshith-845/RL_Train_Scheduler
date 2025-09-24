import gymnasium as gym
import numpy as np
import pickle
from rl_based import TrainSchedulerEnv, evaluate_model
from stable_baselines3 import PPO

# Load model
model = PPO.load("ppo_train_scheduler_generalized.zip")

# Load timetables
with open("random_timetables.pkl", "rb") as f:
    timetables = pickle.load(f)

results_summary = []

for idx, tt in enumerate(timetables):
    env = TrainSchedulerEnv()
    env.trains = tt["trains"]
    env.platforms = tt["platforms"]
    env.num_trains = len(env.trains)
    env.observation_space = env.observation_space = gym.spaces.Box(
        low=0, high=200, shape=(env.num_trains*3,), dtype=np.float32
    )
    
    print(f"--- Evaluating Timetable {idx+1} ---")
    results = evaluate_model(model, env, n_eval_episodes=1)
    results_summary.append(results)
