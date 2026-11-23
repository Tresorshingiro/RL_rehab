import argparse
import os

import torch

from training.common import (
    RunLogger, collect_steps, default_device, finished_episodes, load_checkpoint,
    make_batched_env, save_checkpoint, save_final_model,
)
from training.pg_modules import (
    build_actor, build_critic, build_gae, build_loss, build_policy_network,
    monte_carlo_returns, normalize, policy_entropy, total_loss,
)

# ---------------------------------------------------------------- defaults
DEFAULT_CONFIG = {
    "config_id": "pg_default",
    "algo": "ppo",               # "reinforce", "a2c" or "ppo"
    "total_steps": 150_000,      # environment steps (same budget as DQN)
    "num_envs": 8,
    "steps_per_env": 64,         # rollout length: 8 envs x 64 = 512 steps per update
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,          # A2C/PPO: bias-variance trade-off of the advantage
    "normalize_advantage": True,
    "entropy_coef": 0.01,        # bonus for keeping the policy exploratory
    "critic_coef": 0.5,          # A2C/PPO: weight of the value loss
    "reinforce_baseline": False, # REINFORCE only: subtract a learned V(s) from the return
    "clip_epsilon": 0.2,         # PPO only
    "ppo_epochs": 4,             # PPO only: passes over each batch
    "minibatch_size": 128,       # PPO only
    "hidden": (64, 64),
    "max_grad_norm": 0.5,
    "parallel_envs": False,      # True on Kaggle: one process per env copy
    "seed": 0,
    "checkpoint_every": 20_000,
}


def gradient_step(loss, params, optimizer, max_grad_norm):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()


def update_reinforce(batch, actor, critic, loss_module, optimizer, config):
    """One policy-gradient step using Monte-Carlo returns G_t.

    reinforce_baseline=False: advantage = G_t (classic REINFORCE).
    reinforce_baseline=True:  advantage = G_t - V(s_t), and V is trained on G_t.
    """
    returns, valid = monte_carlo_returns(batch, config["gamma"])
    if not valid.any():
        return float("nan")
    data = batch[valid]                                   # only steps with a finished episode
    data["value_target"] = returns[valid].unsqueeze(-1)
    advantage = returns[valid].unsqueeze(-1)
    if config["reinforce_baseline"]:
        with torch.no_grad():
            advantage = advantage - critic(data.clone())["state_value"]
    if config["normalize_advantage"]:
        advantage = normalize(advantage)
    data["advantage"] = advantage
    loss_td = loss_module(data)
    loss = loss_td["loss_actor"] - config["entropy_coef"] * actor.get_dist(data).entropy().mean()
    if config["reinforce_baseline"]:
        loss = loss + config["critic_coef"] * loss_td["loss_value"]
    gradient_step(loss, loss_module.parameters(), optimizer, config["max_grad_norm"])
    return loss.item()


def update_a2c(batch, loss_module, gae, optimizer, config):
    """One synchronous actor-critic step on the whole rollout."""
    with torch.no_grad():
        gae(batch)                                        # writes "advantage" and "value_target"
    if config["normalize_advantage"]:
        batch["advantage"] = normalize(batch["advantage"])
    loss = total_loss(loss_module(batch.reshape(-1)))
    gradient_step(loss, loss_module.parameters(), optimizer, config["max_grad_norm"])
    return loss.item()


def update_ppo(batch, loss_module, gae, optimizer, config):
    """Several epochs of clipped updates over shuffled minibatches."""
    with torch.no_grad():
        gae(batch)
    if config["normalize_advantage"]:
        batch["advantage"] = normalize(batch["advantage"])
    flat = batch.reshape(-1)
    losses = []
    for _ in range(config["ppo_epochs"]):
        for indices in torch.randperm(flat.numel(), device=flat.device).split(config["minibatch_size"]):
            loss = total_loss(loss_module(flat[indices]))
            gradient_step(loss, loss_module.parameters(), optimizer, config["max_grad_norm"])
            losses.append(loss.item())
    return sum(losses) / len(losses)


def train_pg(config, device="cpu", out_root=".", resume=True):
    config = {**DEFAULT_CONFIG, **config}
    algo, run_id = config["algo"], config["config_id"]
    ckpt_path = os.path.join(out_root, "models", "pg", "checkpoints", f"{run_id}.ckpt")
    model_path = os.path.join(out_root, "models", "pg", f"{run_id}.pt")
    torch.manual_seed(config["seed"])

    env = make_batched_env(config["num_envs"], device, config["seed"], parallel=config["parallel_envs"])
    policy_net = build_policy_network(config["hidden"], device)
    actor = build_actor(policy_net)
    critic = build_critic(config["hidden"], device)
    loss_module = build_loss(algo, actor, critic, config)
    gae = None if algo == "reinforce" else build_gae(critic, config)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config["lr"])

    env_steps, episodes = 0, 0
    checkpoint = load_checkpoint(ckpt_path, device) if resume else None
    if checkpoint is not None:
        loss_module.load_state_dict(checkpoint["loss_module"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        env_steps, episodes = checkpoint["env_steps"], checkpoint["episodes"]
        print(f"[{run_id}] resumed at {env_steps} steps")
    logger = RunLogger(os.path.join(out_root, "logs", algo, f"{run_id}.csv"), episodes_done=episodes if checkpoint else None)

    td = env.reset()
    next_checkpoint = env_steps + config["checkpoint_every"]
    while env_steps < config["total_steps"]:
        # 1) collect a rollout with the current (stochastic) policy
        batch, td = collect_steps(env, actor, td, config["steps_per_env"])
        env_steps += batch.numel()
        entropy = policy_entropy(actor, batch)

        # 2) compute the loss for this algorithm and update the networks
        if algo == "reinforce":
            loss = update_reinforce(batch, actor, critic, loss_module, optimizer, config)
        elif algo == "a2c":
            loss = update_a2c(batch, loss_module, gae, optimizer, config)
        else:
            loss = update_ppo(batch, loss_module, gae, optimizer, config)

        # 3) log finished episodes
        for ep_return, ep_length, outcome in finished_episodes(batch):
            episodes += 1
            logger.log({"episode": episodes, "env_steps": env_steps, "episode_return": round(ep_return, 4),
                        "episode_length": ep_length, "outcome": int(outcome),
                        "loss": round(loss, 6), "entropy": round(entropy, 5)})

        # 4) checkpoint
        if env_steps >= next_checkpoint:
            save_checkpoint(ckpt_path, loss_module=loss_module.state_dict(), optimizer=optimizer.state_dict(),
                            env_steps=env_steps, episodes=episodes)
            next_checkpoint += config["checkpoint_every"]

    logger.close()
    env.close()
    save_final_model(model_path, policy_net, algo, config)
    print(f"[{run_id}] done: {env_steps} steps, {episodes} episodes -> {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "a2c", "ppo"], default="ppo")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["total_steps"])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--parallel", action="store_true", help="one process per env copy")
    args = parser.parse_args()
    train_pg({"algo": args.algo, "config_id": f"{args.algo}_default", "total_steps": args.steps,
              "parallel_envs": args.parallel}, device=args.device)