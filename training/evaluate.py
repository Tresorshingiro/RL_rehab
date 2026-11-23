import glob
import json
import os

import numpy as np
import torch

from environment import RehabCoachEnv
from environment import patient_model as pm
from training.dqn_training import build_q_network
from training.pg_modules import build_policy_network

VALIDATION_SEEDS = range(5_000, 5_050)   # used to pick the best config per algorithm
TEST_SEEDS = range(10_000, 10_200)       # held out: never used for training or selection
ALGOS = ["dqn", "reinforce", "a2c", "ppo"]


def load_policy(model_path):
    """Return act(observation) -> action, always choosing the best action (greedy)."""
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    if saved["algo"] == "dqn":
        network = build_q_network(saved["hidden"], "cpu")      # outputs Q-values
    else:
        network = build_policy_network(saved["hidden"], "cpu")  # outputs logits
    network.load_state_dict(saved["state_dict"])
    network.eval()

    def act(observation):
        with torch.no_grad():
            scores = network(torch.as_tensor(observation, dtype=torch.float32))
        return int(torch.argmax(scores).item())

    return act


def run_episode(act, seed, profile_mode="train"):
    env = RehabCoachEnv(profile_mode=profile_mode)
    obs, info = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(act(obs))
        total += reward
        done = terminated or truncated
    return total, env.day, int(info["outcome"])


def evaluate(act, seeds, profile_mode="train"):
    results = [run_episode(act, seed, profile_mode) for seed in seeds]
    returns = np.array([r[0] for r in results])
    outcomes = np.array([r[2] for r in results])
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "success_rate": float((outcomes == pm.OUTCOME_SUCCESS).mean()),
        "injury_rate": float((outcomes == pm.OUTCOME_INJURY).mean()),
        "dropout_rate": float((outcomes == pm.OUTCOME_DROPOUT).mean()),
        "mean_days": float(np.mean([r[1] for r in results])),
    }


def model_paths(algo, out_root="."):
    folder = "dqn" if algo == "dqn" else "pg"
    return sorted(glob.glob(os.path.join(out_root, "models", folder, f"{algo}_*.pt")))


def select_best_models(out_root="."):
    """Score every finished run on the validation seeds; keep the best per algorithm."""
    best = {}
    for algo in ALGOS:
        scored = []
        for path in model_paths(algo, out_root):
            score = evaluate(load_policy(path), VALIDATION_SEEDS)["mean_return"]
            scored.append((score, path))
            print(f"{os.path.basename(path):<22} validation return {score:7.2f}")
        if scored:
            score, path = max(scored)
            best[algo] = {"path": path.replace(os.sep, "/"), "validation_return": round(score, 3)}
    if best:
        best["overall"] = max(ALGOS, key=lambda a: best[a]["validation_return"] if a in best else -1e9)
    with open(os.path.join(out_root, "models", "best_models.json"), "w") as f:
        json.dump(best, f, indent=2)
    return best


def generalisation_table(best, out_root="."):
    """Evaluate each best model on held-out seeds, in- and out-of-distribution."""
    rows = []
    for algo in ALGOS:
        if algo not in best:
            continue
        path = best[algo]["path"]
        act = load_policy(path)
        for mode in ("train", "unseen"):
            rows.append({"algo": algo, "model": os.path.basename(path),
                         "profiles": "training range" if mode == "train" else "unseen range",
                         **evaluate(act, TEST_SEEDS, mode)})
    path = os.path.join(out_root, "logs", "generalisation.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    best_models = select_best_models()
    for row in generalisation_table(best_models):
        print(f"{row['algo']:<10} {row['profiles']:<15} return {row['mean_return']:6.1f}  "
              f"success {row['success_rate']:.0%}  injury {row['injury_rate']:.0%}  dropout {row['dropout_rate']:.0%}")