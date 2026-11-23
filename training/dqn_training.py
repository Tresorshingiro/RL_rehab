import argparse
import os

import torch
from tensordict.nn import TensorDictSequential
from torchrl.data import Categorical, LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate, ValueEstimators

from training.common import (N_ACTIONS, OBS_DIM, RunLogger, collect_steps,
                             default_device, finished_episodes,
                             load_checkpoint, make_batched_env, save_checkpoint, save_final_model)


# defaults
DEFAULT_CONFIG = {
    "config_id": "dqn_default",
    "total_steps": 150_000,       # environment steps (same budget for every algorithm)
    "num_envs": 8,
    "steps_per_env": 8,           # 8 envs x 8 steps = 64 new transitions per iteration
    "lr": 1e-3,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_fraction": 0.3,    # epsilon reaches eps_end after 30% of the budget
    "buffer_size": 50_000,
    "batch_size": 128,
    "updates_per_iter": 16,        # one gradient step per 4 new transitions
    "target_update": "soft",      # "soft" (Polyak, uses tau) or "hard" (copy every N updates)
    "tau": 0.01,
    "hard_update_interval": 500,
    "double_dqn": False,
    "warmup_steps":2_000,
    "hidden": (128, 128),
    "max_grad_norm": 10.0,
    "seed": 0,
    "checkpoint_every": 20_000     # env steps between checkpoints
}


def build_q_network(hidden, device):
    """10 observations in -> one Q-value per action out."""
    return MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=list(hidden),
               activation_class=torch.nn.ReLU, device=device)


def build_dqn(config, device):
    q_net = build_q_network(config["hidden"], device)
    action_spec = Categorical(N_ACTIONS)
    actor = QValueActor(q_net, in_keys=["observation"], spec=action_spec).to(device)
    decay_steps = int(config["eps_decay_fraction"] * config["total_steps"])
    egreedy = EGreedyModule(spec=action_spec, eps_init=config["eps_start"],
                            eps_end=config["eps_end"],
                            annealing_num_steps=decay_steps).to(device)
    explore_policy = TensorDictSequential(actor, egreedy)

    loss_module = DQNLoss(actor, action_space=action_spec, delay_value=True,
                          double_dqn=config["double_dqn"],
                          loss_function="l2")
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config["gamma"])
    if config["target_update"] == "soft":
        target_updater = SoftUpdate(loss_module, tau=config["tau"])
    else:
        target_updater = HardUpdate(loss_module, value_network_update_interval=config["hard_update_interval"])
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config["lr"])
    return q_net, explore_policy, egreedy, loss_module, target_updater, optimizer


def train_dqn(config, device="cpu", out_root=".", resume=True):
    config = {**DEFAULT_CONFIG, **config}
    run_id = config["config_id"]
    ckpt_path = os.path.join(out_root, "models", "dqn", "checkpoints", f"{run_id}.ckpt")
    model_path = os.path.join(out_root, "models", "dqn", f"{run_id}.pt")
    torch.manual_seed(config["seed"])

    env = make_batched_env(config["num_envs"], device, config["seed"])
    q_net, policy, egreedy, loss_module, target_updater, optimizer = build_dqn(config, device)
    buffer = ReplayBuffer(storage=LazyTensorStorage(config["buffer_size"], device=device),
                          batch_size=config["batch_size"])

    env_steps, episodes = 0, 0
    checkpoint = load_checkpoint(ckpt_path, device) if resume else None
    if checkpoint is not None:
        loss_module.load_state_dict(checkpoint["loss_module"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        egreedy.load_state_dict(checkpoint["egreedy"])
        env_steps, episodes = checkpoint["env_steps"], checkpoint["episodes"]
        print(f"[{run_id}] resumed at {env_steps} steps (replay buffer starts empty)")
    logger = RunLogger(os.path.join(out_root, "logs", "dqn", f"{run_id}.csv"), episodes_done=episodes if checkpoint else None)

    td = env.reset()
    last_loss, next_checkpoint = float("nan"), env_steps + config["checkpoint_every"]
    while env_steps < config["total_steps"]:
        # 1) collect data with the epsilon-greedy policy and store it
        batch, td = collect_steps(env, policy, td, config["steps_per_env"])
        buffer.extend(batch.reshape(-1))
        env_steps += batch.numel()
        egreedy.step(batch.numel())

        # 2) learn from random minibatches of past transitions
        if len(buffer) >= config["warmup_steps"]:
            losses = []
            for _ in range(config["updates_per_iter"]):
                sample = buffer.sample()
                loss = loss_module(sample)["loss"]                       # TD error vs the target network
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config["max_grad_norm"])
                optimizer.step()
                target_updater.step()                                     # slowly move the target network
                losses.append(loss.item())
            last_loss = sum(losses) / len(losses)

        # 3) log every episode that finished in this batch
        for ep_return, ep_length, outcome in finished_episodes(batch):
            episodes += 1
            logger.log({"episode": episodes, "env_steps": env_steps, "episode_return": round(ep_return, 4), "episode_length": ep_length, "outcome": int(outcome), "loss": round(last_loss, 6), "epsilon": round(egreedy.eps.item(), 4)})

        # 4) checkpoint so an interrupted kaggle session can resume
        if env_steps >= next_checkpoint:
            save_checkpoint(ckpt_path, loss_module=loss_module.state_dict(), optimizer=optimizer.state_dict(),
                            egreedy=egreedy.state_dict(), env_steps=env_steps, episodes=episodes)
            next_checkpoint += config["checkpoint_every"]

    logger.close()
    env.close()
    save_final_model(model_path, q_net, "dqn", config)
    print(f"[{run_id}] done: {env_steps} steps, {episodes} episodes -> {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["total_steps"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--config-id", default="dqn_default")
    args = parser.parse_args()
    train_dqn({"config_id": args.config_id, "total_steps": args.steps}, device=args.device)
