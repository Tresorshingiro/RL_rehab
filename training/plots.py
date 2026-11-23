import glob
import json
import os

import matplotlib
matplotlib.use("Agg")                    # no display needed (works on Kaggle)
import matplotlib.pyplot as plt
import pandas as pd

from training.convergence import convergence_episode, rolling_mean

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "a2c": "#1baf7a", "ppo": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOW = 50                              # episodes in the rolling mean

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})


def load_log(algo, config_id, root="."):
    return pd.read_csv(os.path.join(root, "logs", algo, f"{config_id}.csv"))


def best_config_ids(root="."):
    with open(os.path.join(root, "models", "best_models.json")) as f:
        best = json.load(f)
    return {algo: os.path.basename(best[algo]["path"])[:-3] for algo in ALGOS}


def label_line_end(ax, x, y, text, color):
    """Direct label at the right end of a line, in text ink with a colored marker."""
    ax.plot([x], [y], "o", color=color, markersize=5)
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)


def save(fig, name, root):
    os.makedirs(os.path.join(root, "assets"), exist_ok=True)
    path = os.path.join(root, "assets", name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_reward_comparison(best, root):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in ALGOS:
        df = load_log(algo, best[algo], root)
        smooth = rolling_mean(df["episode_return"], WINDOW)
        ax.plot(df["episode"], df["episode_return"], color=COLORS[algo], alpha=0.12, linewidth=0.8)
        ax.plot(df["episode"], smooth, color=COLORS[algo], label=f"{LABELS[algo]} ({best[algo]})")
        label_line_end(ax, df["episode"].iloc[-1], smooth.iloc[-1], LABELS[algo], COLORS[algo])
    ax.set(title=f"Episode return, best config per method ({WINDOW}-episode rolling mean)",
           xlabel="Episode", ylabel="Cumulative reward per episode")
    ax.legend(loc="lower right")
    save(fig, "reward_comparison.png", root)


def plot_dqn_loss(best, root):
    df = load_log("dqn", best["dqn"], root).dropna(subset=["loss"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df["episode"], rolling_mean(df["loss"], WINDOW), color=COLORS["dqn"])
    ax.set(title=f"DQN TD loss ({best['dqn']}, rolling mean)", xlabel="Episode", ylabel="MSE TD loss")
    ax2 = ax.inset_axes([0.62, 0.55, 0.35, 0.4])      # epsilon schedule for context
    ax2.plot(df["episode"], df["epsilon"], color=MUTED, linewidth=1.2)
    ax2.set_title("epsilon", fontsize=8, color=MUTED)
    ax2.tick_params(labelsize=7)
    save(fig, "dqn_loss.png", root)


def plot_entropy(best, root):
    fig, ax = plt.subplots(figsize=(8, 4))
    for algo in ["reinforce", "a2c", "ppo"]:
        df = load_log(algo, best[algo], root)
        ax.plot(df["env_steps"], df["entropy"], color=COLORS[algo], label=LABELS[algo])
        label_line_end(ax, df["env_steps"].iloc[-1], df["entropy"].iloc[-1], LABELS[algo], COLORS[algo])
    ax.axhline(2.197, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("uniform policy, ln 9", (0, 2.197), xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set(title="Policy entropy during training (best config per method)",
           xlabel="Environment steps", ylabel="Entropy (nats)")
    ax.legend(loc="lower left")
    save(fig, "pg_entropy.png", root)


def sweep_summary(root):
    """One row per run: final performance and convergence. Also used for report tables."""
    rows = []
    for algo in ALGOS:
        for path in sorted(glob.glob(os.path.join(root, "logs", algo, f"{algo}_*.csv"))):
            df = pd.read_csv(path)
            last = df.tail(100)
            rows.append({"algo": algo, "config_id": os.path.basename(path)[:-4], "episodes": len(df),
                         "final_return": round(last["episode_return"].mean(), 2),
                         "final_success_rate": round((last["outcome"] == 1).mean(), 3),
                         "converged_at_episode": convergence_episode(df["episode_return"], WINDOW)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(root, "logs", "sweep_summary.csv"), index=False)
    return table


def plot_convergence(best, table, root):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    best_rows = table[table["config_id"].isin(best.values())].set_index("algo").loc[ALGOS]
    bars = ax.bar([LABELS[a] for a in ALGOS], best_rows["converged_at_episode"],
                  color=[COLORS[a] for a in ALGOS], width=0.6)
    ax.bar_label(bars, color=INK, fontsize=9, padding=2)
    ax.set(title="Episodes until the return stabilises (best config)", ylabel="Episode")
    ax.grid(axis="x", visible=False)
    save(fig, "convergence.png", root)


def plot_generalisation(root):
    with open(os.path.join(root, "logs", "generalisation.json")) as f:
        rows = pd.DataFrame(json.load(f))
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, algo in enumerate(ALGOS):
        seen = rows[(rows.algo == algo) & (rows.profiles == "training range")]["success_rate"].iloc[0]
        unseen = rows[(rows.algo == algo) & (rows.profiles == "unseen range")]["success_rate"].iloc[0]
        b1 = ax.bar(i - width / 2 - 0.01, seen, width, color=COLORS[algo])
        b2 = ax.bar(i + width / 2 + 0.01, unseen, width, color=COLORS[algo], alpha=0.45, hatch="//", edgecolor="white")
        ax.bar_label(b1, labels=[f"{seen:.0%}"], fontsize=8, color=INK)
        ax.bar_label(b2, labels=[f"{unseen:.0%}"], fontsize=8, color=INK)
    ax.set_xticks(range(len(ALGOS)), [LABELS[a] for a in ALGOS])
    ax.set(title="Success rate on 200 held-out patients", ylabel="Success rate", ylim=(0, 1.1))
    ax.bar(0, 0, color=MUTED, label="held-out seeds, training profile range")
    ax.bar(0, 0, color=MUTED, alpha=0.45, hatch="//", edgecolor="white", label="held-out seeds, unseen (harder) range")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "generalisation.png", root)


def make_all_figures(root="."):
    best = best_config_ids(root)
    plot_reward_comparison(best, root)
    plot_dqn_loss(best, root)
    plot_entropy(best, root)
    table = sweep_summary(root)
    plot_convergence(best, table, root)
    plot_generalisation(root)


if __name__ == "__main__":
    make_all_figures()