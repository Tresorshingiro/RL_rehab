import glob
import json
import os

import matplotlib
matplotlib.use("Agg")                    # no display needed (works on Kaggle)
import matplotlib.pyplot as plt
import pandas as pd

from training.convergence import TOLERANCE, convergence_band, convergence_episode, rolling_mean

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLORS = {"dqn": "#2a78d6", "reinforce": "#eb6834", "a2c": "#1baf7a", "ppo": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOW = 50                              # episodes in the rolling mean

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.axisbelow": True,          # gridlines behind bars, not across them
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})


def load_log(algo, config_id, root="."):
    return pd.read_csv(os.path.join(root, "logs", algo, f"{config_id}.csv"))


def best_config_ids(root="."):
    with open(os.path.join(root, "models", "best_models.json")) as f:
        best = json.load(f)
    return {algo: os.path.basename(best[algo]["path"])[:-3] for algo in ALGOS}


def label_line_ends(ax, ends, min_gap=0.06):
    """Direct labels at the right end of each line, in text ink with a colored marker.

    `ends` is a list of (x, y, text, color). Labels whose lines end close together
    are pushed apart vertically (min_gap = fraction of the y-axis height) and placed
    in one column just right of the last line end, so they never overlap.
    """
    low, high = ax.get_ylim()
    gap = min_gap * (high - low)
    x_text = max(x for x, _, _, _ in ends)
    placed = []
    for x, y, text, color in sorted(ends, key=lambda e: e[1]):
        ax.plot([x], [y], "o", color=color, markersize=5)
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (x_text, y_text), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9, annotation_clip=False)
    ax.set_ylim(low, high)


def save(fig, name, root):
    os.makedirs(os.path.join(root, "assets"), exist_ok=True)
    path = os.path.join(root, "assets", name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def method_grid(title):
    """2 x 2 subplots, one per method, sharing both axes so the panels compare directly."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    fig.suptitle(title, color=INK, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative reward per episode")
    return fig, axes.ravel()


def plot_reward_comparison(best, root):
    """One panel per method: raw episode returns (faint) and their rolling mean."""
    fig, axes = method_grid(f"Cumulative reward per episode, best config per method "
                            f"(line = {WINDOW}-episode rolling mean)")
    for ax, algo in zip(axes, ALGOS):
        df = load_log(algo, best[algo], root)
        smooth = rolling_mean(df["episode_return"], WINDOW)
        ax.plot(df["episode"], df["episode_return"], color=COLORS[algo], alpha=0.15, linewidth=0.6)
        ax.plot(df["episode"], smooth, color=COLORS[algo])
        ax.plot([df["episode"].iloc[-1]], [smooth.iloc[-1]], "o", color=COLORS[algo], markersize=5)
        ax.annotate(f"{smooth.iloc[-1]:.1f}", (df["episode"].iloc[-1], smooth.iloc[-1]),
                    xytext=(0, 8), textcoords="offset points", ha="right", color=INK, fontsize=8)
        ax.set_title(f"{LABELS[algo]} ({best[algo]})", color=INK, fontsize=10, loc="left")
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
    ends = []
    for algo in ["reinforce", "a2c", "ppo"]:
        df = load_log(algo, best[algo], root)
        ax.plot(df["env_steps"], df["entropy"], color=COLORS[algo], label=LABELS[algo])
        ends.append((df["env_steps"].iloc[-1], df["entropy"].iloc[-1], LABELS[algo], COLORS[algo]))
    label_line_ends(ax, ends)
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


def plot_convergence(best, root):
    """One panel per method: smoothed return, the final-level band and the convergence episode."""
    fig, axes = method_grid("Episodes until the return stabilises (best config per method)")
    for ax, algo in zip(axes, ALGOS):
        df = load_log(algo, best[algo], root)
        smooth = rolling_mean(df["episode_return"], WINDOW)
        episode, level, band = convergence_band(df["episode_return"], WINDOW)
        ax.axhspan(level - band, level + band, color=COLORS[algo], alpha=0.12, linewidth=0)
        ax.plot(df["episode"], smooth, color=COLORS[algo])
        ax.axvline(episode, color=MUTED, linestyle="--", linewidth=1)
        # x in data units, y in axes units: the label sits in the empty bottom strip, by the line
        ax.text(episode, 0.05, f" stable from\n episode {episode}", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", color=INK, fontsize=8)
        ax.set_title(f"{LABELS[algo]} ({best[algo]})", color=INK, fontsize=10, loc="left")
    fig.text(0.5, -0.01, f"Shaded band: final level (mean of the last 10% of the curve) "
             f"± {int(TOLERANCE * 100)}% of the curve's range. Dashed line: last time the curve leaves the band.",
             ha="center", color=MUTED, fontsize=8)
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
    # Extra headroom above 100% holds the legend, so it never covers a bar label.
    ax.set(title="Success rate on 200 held-out patients", ylabel="Success rate", ylim=(0, 1.28))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.bar(0, 0, color=MUTED, label="held-out seeds, training profile range")
    ax.bar(0, 0, color=MUTED, alpha=0.45, hatch="//", edgecolor="white", label="held-out seeds, unseen (harder) range")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "generalisation.png", root)


def make_all_figures(root="."):
    best = best_config_ids(root)
    plot_reward_comparison(best, root)
    plot_dqn_loss(best, root)
    plot_entropy(best, root)
    sweep_summary(root)
    plot_convergence(best, root)
    plot_generalisation(root)


if __name__ == "__main__":
    make_all_figures()