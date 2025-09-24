import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from rl_based import TrainSchedulerEnv, evaluate_model
import pickle

# Load random timetables
with open("random_timetables.pkl", "rb") as f:
    timetables = pickle.load(f)

# Generalized Environment
class GeneralizedTrainSchedulerEnv(TrainSchedulerEnv):
    def reset(self, *, seed=None, options=None):
        # Pick a random timetable for this episode
        tt = np.random.choice(timetables)
        self.trains = tt["trains"]
        self.platforms = tt["platforms"]
        self.num_trains = len(self.trains)
        self.observation_space = gym.spaces.Box(
            low=0, high=200, shape=(self.num_trains * 3,), dtype=np.float32
        )
        return super().reset(seed=seed, options=options)

# Main Training
if __name__ == "__main__":
    env = GeneralizedTrainSchedulerEnv()

    # Train PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)  # Increase for better generalization
    model.save("ppo_train_scheduler_generalized")

    # Evaluate
    results = evaluate_model(model, env, n_eval_episodes=50)
