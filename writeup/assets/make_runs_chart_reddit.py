"""Generates runs_chart_reddit.png — the English runs chart with two story callouts,
tuned as a social image for the Reddit discussion post.

Same real data as make_runs_chart_en.py (MLE-bench-agentbeats leaderboard repo).
Adds two honest annotations so the image carries the hook on its own:
  - run #2 is the gold run whose submission.csv was lost from /tmp;
  - runs #7 and #8 are the same Docker image, 14 minutes apart, yet differ.
Run:  python make_runs_chart_reddit.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# chronological order (13 April 2026, UTC)
runs   = [0.81609, 0.82069, 0.80460, 0.80230, 0.81149, 0.80345, 0.80460, 0.80230]
times  = ["02:19", "02:44", "06:07", "06:25", "07:47", "16:58", "17:27", "17:41"]
labels = [f"#{i+1}\n{t}" for i, t in enumerate(times)]
gold, silver, bronze = 0.82066, 0.81388, 0.80967
gold_line_y = 0.82045  # drawn a touch below the true bar so the gold column clearly crosses it


def color(s):
    if s >= gold:   return "#E1A100"
    if s >= silver: return "#9AA0A6"
    if s >= bronze: return "#A0522D"
    return "#4C72B0"


fig, ax = plt.subplots(figsize=(9.6, 5.2))
ax.bar(range(8), runs, color=[color(s) for s in runs],
       edgecolor="black", linewidth=0.5, zorder=3, width=0.62)
for i, s in enumerate(runs):
    ax.annotate(f"{s:.5f}", (i, s), ha="center", va="bottom", fontsize=8)

for y, name, c in [(gold_line_y, "gold bar 0.82066", "#C98A00"),
                   (silver, "silver 0.81388", "#6B7075"),
                   (bronze, "bronze 0.80967", "#A0522D")]:
    ax.axhline(y, ls="--", lw=1.1, color=c, zorder=2)
    ax.text(7.55, y + 0.00012, name, va="bottom", ha="right", fontsize=8, color=c,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2), zorder=4)

# callout 1: the gold run whose artifact was lost
ax.annotate("gold run (0.82069) — and the winning\nsubmission.csv was lost from /tmp, unreproducible",
            xy=(1, 0.82035), xytext=(2.15, 0.8158), ha="left", va="center", fontsize=8.5,
            color="#8a6500", fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#8a6500"), zorder=6)

# callout 2: runs 7 & 8 are the same Docker image, 14 min apart
ax.text(6.2, 0.8122, "same Docker image, 14 min apart", ha="center", va="bottom",
        fontsize=8.5, color="#333333", fontweight="bold", zorder=6)
ax.annotate("", xy=(6, 0.8052), xytext=(6.1, 0.8118),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"), zorder=6)
ax.annotate("", xy=(7, 0.8029), xytext=(6.6, 0.8118),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"), zorder=6)

ax.set_xticks(range(8)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.800, 0.8225)
ax.set_xlim(-0.7, 7.7)
ax.set_ylabel("score (accuracy, MLE-bench eval)")
ax.set_title("Same agent, same code, eight runs: the score crossed three tiers",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
fig.text(0.5, -0.02, "Kaggle Spaceship Titanic, 13 Apr UTC, in run order. Tiers are MLE-bench thresholds (not Kaggle medals).",
         ha="center", fontsize=8, style="italic")
fig.text(0.5, -0.06, "Y-axis clipped (0.800-0.822) so the thresholds are visible.",
         ha="center", fontsize=8, style="italic")
fig.tight_layout()
fig.savefig("runs_chart_reddit.png", dpi=140, bbox_inches="tight")
print("saved runs_chart_reddit.png")
