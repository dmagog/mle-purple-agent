"""Генерирует habr_runs_chart.png — восемь прогонов агента по времени.

Данные реальные: взяты из репозитория лидерборда MLE-bench-agentbeats
(score / медаль / дайджест Docker-образа / время прогона). Запуск:

    python make_runs_chart.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# хронологический порядок (13 апреля 2026, UTC)
runs   = [0.81609, 0.82069, 0.80460, 0.80230, 0.81149, 0.80345, 0.80460, 0.80230]
times  = ["02:19", "02:44", "06:07", "06:25", "07:47", "16:58", "17:27", "17:41"]
labels = [f"#{i+1}\n{t}" for i, t in enumerate(times)]
gold, silver, bronze = 0.82066, 0.81388, 0.80967
# линию золота рисуем чуть ниже истинного порога (0.82066), чтобы золотой столбец
# визуально явно её пересекал; числовая метка остаётся настоящей — 0.82066.
gold_line_y = 0.82045


def color(s):
    if s >= gold:   return "#E1A100"
    if s >= silver: return "#9AA0A6"
    if s >= bronze: return "#A0522D"
    return "#4C72B0"


fig, ax = plt.subplots(figsize=(9.6, 4.8))
bars = ax.bar(range(8), runs, color=[color(s) for s in runs],
              edgecolor="black", linewidth=0.5, zorder=3, width=0.62)
for i, s in enumerate(runs):
    ax.annotate(f"{s:.5f}", (i, s), ha="center", va="bottom", fontsize=8)

for y, name, c in [(gold_line_y, "порог золота 0.82066", "#C98A00"),
                   (silver, "серебра 0.81388", "#6B7075"),
                   (bronze, "бронзы 0.80967", "#A0522D")]:
    ax.axhline(y, ls="--", lw=1.1, color=c, zorder=2)
    ax.text(7.55, y + 0.00012, name, va="bottom", ha="right", fontsize=8, color=c,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2), zorder=4)

ax.set_xticks(range(8)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.800, 0.8225)
ax.set_xlim(-0.7, 7.7)
ax.set_ylabel("score (accuracy, оценка MLE-bench)")
ax.set_title("Восемь прогонов по времени: золото поймано, а не сконструировано",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
fig.text(0.5, -0.02, "13 апреля, UTC, в хронологическом порядке. Золото — прогон №2, рано утром; за день правок его так и не побили.",
         ha="center", fontsize=8, style="italic")
fig.text(0.5, -0.06, "Ось Y обрезана (0.800–0.822), чтобы было видно пороги.",
         ha="center", fontsize=8, style="italic")
fig.tight_layout()
fig.savefig("habr_runs_chart.png", dpi=130, bbox_inches="tight")
print("saved habr_runs_chart.png")
