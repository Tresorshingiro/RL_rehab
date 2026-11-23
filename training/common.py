import csv
import os

import torch
from torchrl.envs import(
    Compose, GymWrapper, ParallelEnv, RewardSum, SerialEnv, StepCounter, TransformedEnv,
)
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment import RehabCoachEnv

OBS_DIM, N_ACTIONS = 10, 9
CSV_FIELDS = ["episode", "env_steps", "episode_return", "episode_length", "outcome", "loss", "entropy", "epsilon"]

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_single_env(profile_mode="train"):
    env = GymWrapper(RehabCoachEnv(profile_mode = profile_mode), categorical_action_encoding=True)
    env.set_info_dict_reader(default_info_dict_reader(["outcome"]))
    return TransformedEnv(env, Compose(RewardSum(), StepCounter()))

def make_batched_env(num_envs, device, seed, parallel=False):
    env_class = ParallelEnv if parallel else SerialEnv
    env = env_class(num_envs, make_single_env, device=device)
    env.set_seed(seed)
    return env

def collect_steps(env, policy, td, steps_per_env):
    """Run the policy for steps_per_env steps in every env copy"""
    collected = []
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        for _ in range(steps_per_env):
            td = policy(td)
            step_td, td = env.step_and_maybe_reset(td)
            collected.append(step_td.clone())
    batch = torch.stack(collected, dim=1).contiguous()
    return batch, td

def finished_episodes(batch):
    """Return (return, length, outcome) for every episode that ended in batch."""
    done = batch["next", "done"].squeeze(-1)
    returns = batch["next", "episode_reward"].squeeze(-1)[done]
    lengths = batch["next", "step_count"].squeeze(-1)[done]
    outcomes = batch["next", "outcome"].reshape(done.shape)[done]
    return list(zip(returns.tolist(), lengths.tolist(), outcomes.tolist()))

class RunLogger:
    """Writes one CSV row per finished episode"""
    def __init__(self, path, episodes_done=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kept_rows = []
        if episodes_done is not None and os.path.exists(path):
            with open(path, newline="") as f:
                kept_rows = [row for row in csv.DictReader(f)
                if int(row["episode"]) <= episodes_done]
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.writer.writerows(kept_rows)


    def log(self, row):
        self.writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
        self.file.flush()                   # a killed Kaggle session keeps everything logged so far

    def close(self):
        self.file.close()

def save_checkpoint(path, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)   # atomic: a crash never leaves half a file

def load_checkpoint(path, device):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def save_final_model(path, network, algo, config):
    """Save only what play.py / the API need: the network weight and its shape"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"algo": algo, "config": config, "hidden": list(config["hidden"]),
                "state_dict": {k: v.cpu() for k, v in network.state_dict().items()}}, path)
    
