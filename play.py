import argparse
import json
import os
import time

import numpy as np

from environment import RehabCoachEnv
from environment import patient_model as pm
from training.evaluate import load_policy

BEST_MODELS_FILE = "models/best_models.json"
EPISODE_DIR = "logs/episodes"


def choose_model(algo=None):
    """Path of the best model for `algo`, or of the best algorithm overall."""
    if not os.path.exists(BEST_MODELS_FILE):
        return None, None
    with open(BEST_MODELS_FILE) as f:
        best = json.load(f)
    algo = algo or best["overall"]
    return algo, best[algo]["path"]


def describe_observation(obs):
    raw = dict(zip(pm.OBS_NAMES, obs.tolist()))
    return (f"ROM {raw['rom']:.0%} | form {raw['form']:.2f} | pain {raw['pain'] * 10:.1f} | "
            f"fatigue {raw['fatigue'] * 10:.1f} | motivation {raw['motivation']:.2f} | "
            f"adherence {raw['adherence_7d']:.0%} | diff {round(raw['difficulty'] * 4) + 1} | "
            f"sets {round(raw['sets'] * 4) + 1}")


def play_episode(act, seed, profile_mode, render_mode, frames_per_day):
    env = RehabCoachEnv(render_mode=render_mode, profile_mode=profile_mode)
    obs, info = env.reset(seed=seed)
    record = {"seed": seed, "profile_mode": profile_mode, "steps": []}
    if render_mode == "human":
        env.render()                                   # show day 0 before the first action
        env.renderer.frames_per_day = frames_per_day
    print(f"\n=== New patient (seed {seed}, {profile_mode} profiles) ===")
    print(f"Day  0 | start          | {describe_observation(obs)}")
    done = False
    while not done:
        action = act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Day {info['day']:>2} | {pm.ACTION_NAMES[action]:<21} -> {info['status']:<8} | "
              f"reward {reward:+6.2f} | {describe_observation(next_obs)}")
        record["steps"].append({
            "day": info["day"], "observation": [round(float(x), 4) for x in obs],
            "action": action, "action_name": pm.ACTION_NAMES[action], "reward": round(reward, 4),
            "info": {"status": info["status"], "outcome": info["outcome_name"], "flags_used": info["flags_used"]},
        })
        obs, done = next_obs, terminated or truncated
        if env.renderer is not None and env.renderer.window_closed:
            break
    print(f"--> {info['outcome_name']} on day {info['day']}, total reward {env.total_reward:+.1f}")
    record.update({"outcome": info["outcome_name"], "days": info["day"], "total_reward": round(env.total_reward, 3)})
    env.close()
    return record


def save_episode(record, algo):
    os.makedirs(EPISODE_DIR, exist_ok=True)
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{algo}_seed{record['seed']}.json"
    path = os.path.join(EPISODE_DIR, name)
    with open(path, "w") as f:
        json.dump({"agent": algo, **record}, f, indent=2)
    print(f"episode saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run the trained rehab coach.")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], help="default: best overall")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="patient seed (10000+ are held-out patients); default: a random held-out patient")
    parser.add_argument("--profiles", choices=["train", "unseen"], default="train")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--frames-per-day", type=int, default=90, help="frames per simulated day at 30 FPS (90 = 3 s per day); lower = faster")
    args = parser.parse_args()
    if args.seed is None:
        # Random held-out patient (never used for training or model selection).
        # The seed is printed with each episode, so any run can be replayed with --seed.
        args.seed = int(np.random.default_rng().integers(10_000, 100_000))

    algo, path = choose_model(args.algo)
    if path is None:
        print("No trained model found (models/best_models.json). Running a RANDOM agent instead.")
        rng = np.random.default_rng(0)

        def act(obs):
            return int(rng.integers(pm.N_ACTIONS))
        algo = "random"
    else:
        print(f"Loaded {algo.upper()} agent from {path}")
        act = load_policy(path)

    for i in range(args.episodes):
        record = play_episode(act, args.seed + i, args.profiles,
                              None if args.no_gui else "human", args.frames_per_day)
        save_episode(record, algo)


if __name__ == "__main__":
    main()