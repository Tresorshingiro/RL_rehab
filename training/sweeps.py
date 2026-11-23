import argparse
import os
import time

from training.common import default_device
from training.dqn_training import train_dqn
from training.pg_training import train_pg

TOTAL_STEPS = 150_000   # identical environment-step budget for every run

BASE = {
    "dqn": {},  # DEFAULT_CONFIG in dqn_training.py
    "ppo": {"algo": "ppo", "lr": 3e-4, "steps_per_env": 64, "ppo_epochs": 4},
    "a2c": {"algo": "a2c", "lr": 7e-4, "steps_per_env": 16},
    "reinforce": {"algo": "reinforce", "lr": 1e-3, "steps_per_env": 128, "critic_coef": 0.5},
}

SWEEPS = {
    "dqn": [
        ("dqn_01", "baseline", {}),
        ("dqn_02", "lower learning rate", {"lr": 3e-4}),
        ("dqn_03", "higher learning rate", {"lr": 3e-3}),
        ("dqn_04", "short horizon", {"gamma": 0.9}),
        ("dqn_05", "fast epsilon decay", {"eps_decay_fraction": 0.1}),
        ("dqn_06", "slow decay, more exploration", {"eps_decay_fraction": 0.6, "eps_end": 0.1}),
        ("dqn_07", "small replay buffer", {"buffer_size": 5_000}),
        ("dqn_08", "large replay buffer", {"buffer_size": 150_000}),
        ("dqn_09", "small batch", {"batch_size": 32}),
        ("dqn_10", "large batch", {"batch_size": 256}),
        ("dqn_11", "hard target update every 1000", {"target_update": "hard", "hard_update_interval": 1000}),
        ("dqn_12", "slow soft target + double DQN", {"tau": 0.001, "double_dqn": True}),
    ],
    "ppo": [
        ("ppo_01", "baseline", {}),
        ("ppo_02", "lower learning rate", {"lr": 1e-4}),
        ("ppo_03", "higher learning rate", {"lr": 1e-3}),
        ("ppo_04", "short horizon", {"gamma": 0.95}),
        ("ppo_05", "short rollout (256)", {"steps_per_env": 32}),
        ("ppo_06", "long rollout (1024)", {"steps_per_env": 128}),
        ("ppo_07", "no entropy bonus", {"entropy_coef": 0.0}),
        ("ppo_08", "strong entropy bonus", {"entropy_coef": 0.05}),
        ("ppo_09", "low GAE lambda, no adv. norm", {"gae_lambda": 0.8, "normalize_advantage": False}),
        ("ppo_10", "tight clip", {"clip_epsilon": 0.1}),
        ("ppo_11", "loose clip", {"clip_epsilon": 0.3}),
        ("ppo_12", "more epochs, heavier value loss", {"ppo_epochs": 10, "critic_coef": 1.0}),
    ],
    "a2c": [
        ("a2c_01", "baseline", {}),
        ("a2c_02", "lower learning rate", {"lr": 3e-4}),
        ("a2c_03", "higher learning rate", {"lr": 2e-3}),
        ("a2c_04", "short horizon", {"gamma": 0.95}),
        ("a2c_05", "very short rollout (64)", {"steps_per_env": 8}),
        ("a2c_06", "longer rollout (512)", {"steps_per_env": 64}),
        ("a2c_07", "no entropy bonus", {"entropy_coef": 0.0}),
        ("a2c_08", "strong entropy bonus", {"entropy_coef": 0.05}),
        ("a2c_09", "GAE lambda 1.0 (n-step returns)", {"gae_lambda": 1.0}),
        ("a2c_10", "no advantage normalisation", {"normalize_advantage": False}),
        ("a2c_11", "light value loss", {"critic_coef": 0.25}),
        ("a2c_12", "heavy value loss", {"critic_coef": 1.0}),
    ],
    "reinforce": [
        ("reinforce_01", "baseline (pure Monte-Carlo)", {}),
        ("reinforce_02", "lower learning rate", {"lr": 3e-4}),
        ("reinforce_03", "higher learning rate", {"lr": 3e-3}),
        ("reinforce_04", "short horizon", {"gamma": 0.95}),
        ("reinforce_05", "short rollout (512)", {"steps_per_env": 64}),
        ("reinforce_06", "long rollout (2048)", {"steps_per_env": 256}),
        ("reinforce_07", "no entropy bonus", {"entropy_coef": 0.0}),
        ("reinforce_08", "strong entropy bonus", {"entropy_coef": 0.05}),
        ("reinforce_09", "no return normalisation", {"normalize_advantage": False}),
        ("reinforce_10", "learned baseline V(s)", {"reinforce_baseline": True}),
        ("reinforce_11", "baseline, heavy value loss", {"reinforce_baseline": True, "critic_coef": 1.0}),
        ("reinforce_12", "bigger network", {"hidden": (128, 128)}),
    ],
}


def build_config(algo, config_id, changes, total_steps):
    return {**BASE[algo], **changes, "config_id": config_id, "total_steps": total_steps}


def run_sweep(algo, total_steps, device, out_root=".", parallel=False, only=None):
    for config_id, description, changes in SWEEPS[algo]:
        if only and config_id not in only:
            continue
        folder = "dqn" if algo == "dqn" else "pg"
        if os.path.exists(os.path.join(out_root, "models", folder, f"{config_id}.pt")):
            print(f"skip {config_id}: already finished")
            continue
        config = build_config(algo, config_id, changes, total_steps)
        print(f"=== {config_id}: {description} ===", flush=True)
        start = time.time()
        if algo == "dqn":
            train_dqn(config, device=device, out_root=out_root)
        else:
            train_pg({**config, "parallel_envs": parallel}, device=device, out_root=out_root)
        print(f"    took {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="all", choices=["all", "dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", default=".", help="root folder for models/ and logs/")
    parser.add_argument("--parallel", action="store_true", help="ParallelEnv for policy-gradient runs")
    parser.add_argument("--only", nargs="*", help="run just these config ids")
    args = parser.parse_args()
    algos = ["dqn", "ppo", "a2c", "reinforce"] if args.algo == "all" else [args.algo]
    for algo in algos:
        run_sweep(algo, args.steps, args.device, args.out, args.parallel, args.only)