import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from timetable import trains, platforms


class TrainSchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.trains = sorted(trains, key=lambda x: x["arrival"])
        self.platforms = platforms
        self.num_trains = len(self.trains)

        # Action: choose a platform or wait at signal
        # Action = platform index OR wait (last action)
        self.action_space = spaces.Discrete(len(self.platforms) + 1)

        # Observation: arrival, departure, priority for all trains
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(self.num_trains * 3,), dtype=np.float32
        )

        self.current_idx = 0
        self.schedule = {}
        self.wait_times = {p: 0 for p in self.platforms}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        self.schedule = {}
        self.wait_times = {p: 0 for p in self.platforms}
        return self._get_state(), {}

    def _get_state(self):
        obs = []
        for t in self.trains:
            obs += [t["arrival"], t["departure"], t["priority"]]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        train = self.trains[self.current_idx]

        # If last action = "wait at signal"
        if action == len(self.platforms):
            wait_time = 1  # force 1 time-unit wait
            train["arrival"] += wait_time
            train["departure"] += wait_time
            reward = -1  # small penalty for waiting
            info = {
                "conflict": 0,
                "wait_time": wait_time,
                "priority": train["priority"],
                "platform": "WAIT",
            }
            return self._get_state(), reward, False, False, info

        chosen_platform = self.platforms[action]

        # ----------------------------
        # Conflict + Waiting Logic
        # ----------------------------
        conflict = False
        wait_time = 0

        for t_id, p in self.schedule.items():
            t = next(x for x in self.trains if x["id"] == t_id)
            if p == chosen_platform:
                # If overlapping in time
                if not (t["departure"] <= train["arrival"] or train["departure"] <= t["arrival"]):
                    conflict = True
                    # 🚦 Signal: train must wait until platform free
                    wait_time = max(0, t["departure"] - train["arrival"])
                    # Update actual arrival & departure with wait
                    train["arrival"] += wait_time
                    train["departure"] += wait_time

        # ----------------------------
        # Reward System
        # ----------------------------
        reward = 0
        if conflict and wait_time > 5:
            reward -= 20  # heavy penalty for big delays
        elif conflict and wait_time > 0:
            reward -= 5  # small penalty for short wait
        else:
            reward += 10  # reward for conflict-free scheduling

        # Encourage spreading across platforms
        used_platforms = set(self.schedule.values())
        if chosen_platform not in used_platforms:
            reward += 5

        # Penalize overcrowding
        platform_load = list(self.schedule.values()).count(chosen_platform)
        reward -= platform_load * 2

        # Encourage compact scheduling
        duration = train["departure"] - train["arrival"]
        reward += max(0, 5 - duration)

        # Priority handling (bonus if high-priority train avoids waiting)
        if train["priority"] == 1 and wait_time == 0:
            reward += 10
        elif train["priority"] == 1 and wait_time > 0:
            reward -= 10

        # ----------------------------
        # Update schedule
        # ----------------------------
        self.schedule[train["id"]] = chosen_platform
        self.current_idx += 1

        terminated = self.current_idx >= len(self.trains)
        truncated = False

        # ----------------------------
        # Info for evaluation
        # ----------------------------
        info = {
            "conflict": int(conflict),
            "wait_time": wait_time,
            "priority": train["priority"],
            "platform": chosen_platform,
        }

        return self._get_state(), reward, terminated, truncated, info


# ----------------------------------
# Evaluation Function (Standalone)
# ----------------------------------
def evaluate_model(model, env, n_eval_episodes=20):
    rewards = []
    conflicts = []
    priority_success = []
    waiting_times = []
    platform_balance = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        conflict_count = 0
        high_priority = 0
        high_priority_success = 0
        total_wait_time = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if "conflict" in info:
                conflict_count += info["conflict"]
            if "wait_time" in info:
                total_wait_time += info["wait_time"]

            # Track high-priority trains
            train = env.trains[env.current_idx - 1] if env.current_idx > 0 else None
            if train and train["priority"] <= 2:
                high_priority += 1
                if "conflict" in info and info["conflict"] == 0:
                    high_priority_success += 1

        # Episode metrics
        rewards.append(total_reward)
        conflicts.append(conflict_count)
        waiting_times.append(total_wait_time)
        if high_priority > 0:
            priority_success.append(high_priority_success / high_priority)
        else:
            priority_success.append(1.0)

        # Platform utilization
        usage = list(env.schedule.values())
        balance = len(set(usage)) / len(env.platforms)
        platform_balance.append(balance)

    # Final summary
    print("📊 Evaluation Results over", n_eval_episodes, "episodes")
    print("Mean Reward:", np.mean(rewards))
    print("Conflicts per Episode:", np.mean(conflicts))
    print("High Priority Success Rate:", np.mean(priority_success) * 100, "%")
    print("Avg Waiting Time:", np.mean(waiting_times))
    print("Platform Utilization Balance:", np.mean(platform_balance))

    return {
        "mean_reward": np.mean(rewards),
        "conflicts": np.mean(conflicts),
        "priority_success_rate": np.mean(priority_success),
        "avg_wait_time": np.mean(waiting_times),
        "platform_balance": np.mean(platform_balance),
    }


# ----------------------------------
# Main
# ----------------------------------
if __name__ == "__main__":
    env = TrainSchedulerEnv()

    # To train new model:
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=500000)
    # model.save("ppo_train_scheduler_with_signals")

    # To load existing model:
    model = PPO.load("ppo_train_scheduler_with_signals", env=env)

    # Evaluate loaded model
    results = evaluate_model(model, env, n_eval_episodes=50)
