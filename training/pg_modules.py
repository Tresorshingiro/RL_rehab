import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from training.common import N_ACTIONS, OBS_DIM


def build_policy_network(hidden, device):
    """10 observations in -> 9 action logits out (same shape as the DQN network)."""
    return MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=list(hidden),
               activation_class=torch.nn.Tanh, device=device)


def build_actor(policy_net):
    """Wrap the logits network as a stochastic policy over the 9 actions."""
    logits_module = TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        logits_module, in_keys=["logits"], out_keys=["action"],
        distribution_class=torch.distributions.Categorical, return_log_prob=True,
    )


def build_critic(hidden, device):
    value_net = MLP(in_features=OBS_DIM, out_features=1, num_cells=list(hidden),
                    activation_class=torch.nn.Tanh, device=device)
    return ValueOperator(value_net, in_keys=["observation"])  # writes "state_value"


def build_loss(algo, actor, critic, config):
    """The TorchRL loss module for each algorithm."""
    if algo == "reinforce":
        # TorchRL requires a critic here. We always supply the advantage ourselves
        # (Monte-Carlo returns), so the critic is only trained as an optional baseline.
        return ReinforceLoss(actor_network=actor, critic_network=critic)
    if algo == "a2c":
        return A2CLoss(actor_network=actor, critic_network=critic,
                       entropy_coeff=config["entropy_coef"], critic_coeff=config["critic_coef"])
    if algo == "ppo":
        return ClipPPOLoss(actor_network=actor, critic_network=critic,
                           clip_epsilon=config["clip_epsilon"], entropy_coeff=config["entropy_coef"],
                           critic_coeff=config["critic_coef"], normalize_advantage=False)
    raise ValueError(f"unknown algorithm {algo}")


def build_gae(critic, config):
    return GAE(gamma=config["gamma"], lmbda=config["gae_lambda"], value_network=critic)


def monte_carlo_returns(batch, gamma):
    """Reward-to-go G_t for every step, computed backwards through time.

    Returns (returns, valid). Steps whose episode had not finished by the end of
    the batch have no complete return, so valid=False and REINFORCE skips them.
    """
    rewards = batch["next", "reward"].squeeze(-1)           # [num_envs, T]
    dones = batch["next", "done"].squeeze(-1).float()
    returns = torch.zeros_like(rewards)
    valid = torch.zeros_like(rewards, dtype=torch.bool)
    running = torch.zeros_like(rewards[:, 0])
    seen_end = torch.zeros_like(rewards[:, 0], dtype=torch.bool)
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running * (1.0 - dones[:, t])
        seen_end = seen_end | dones[:, t].bool()
        returns[:, t] = running
        valid[:, t] = seen_end
    return returns, valid


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def policy_entropy(actor, batch):
    """Average entropy of the policy on this batch (higher = more exploration)."""
    with torch.no_grad():
        dist = actor.get_dist(batch)
        return dist.entropy().mean().item()


def total_loss(loss_td):
    """Sum every term whose name starts with 'loss_' (actor, critic, entropy)."""
    return sum(value for key, value in loss_td.items() if key.startswith("loss_"))