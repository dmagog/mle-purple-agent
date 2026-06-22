"""Генерирует habr_cover.png — обложку статьи (1200x630).

Оригинальная иллюстрация (без чужих ассетов): терминал с реальной записью
золотого прогона, потерянный /tmp-файл и мотив green -> A2A -> purple. Запуск:

    python make_cover.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

W, H = 1200, 630
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

gx, gy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
g = (gx * 0.45 + gy * 0.55)
cmap = mc.LinearSegmentedColormap.from_list("bg", ["#070810", "#141833", "#0b0d1a"])
ax.imshow(g, extent=[0, W, 0, H], aspect="auto", cmap=cmap, origin="lower", zorder=0)

ax.text(64, 575, "AgentX · AgentBeats  /  трек MLE-bench", fontsize=14,
        color="#8b93b8", family="DejaVu Sans", zorder=5)
ax.text(62, 500, "Золото взято.", fontsize=48, fontweight="bold",
        color="#f5f6fa", family="DejaVu Sans", zorder=5)
ax.text(64, 432, "Код — в /tmp", fontsize=48, fontweight="bold",
        color="#E1A100", family="DejaVu Sans", zorder=5)
ax.text(66, 330, "Автономный агент прошёл золотой\nпорог MLE-bench — и тот прогон\nуже не повторить.",
        fontsize=16.5, color="#aeb6d6", family="DejaVu Sans", linespacing=1.5, zorder=5)

gy0 = 95
ax.add_patch(Circle((88, gy0), 15, color="#3fb950", zorder=5))
ax.text(112, gy0 - 6, "green", fontsize=13.5, color="#cdd3ea", family="DejaVu Sans", zorder=5)
ax.add_patch(FancyArrowPatch((188, gy0), (278, gy0), arrowstyle="-|>",
            mutation_scale=15, color="#6b7299", lw=1.6, zorder=5))
ax.text(212, gy0 + 12, "A2A", fontsize=10.5, color="#8b93b8", family="DejaVu Sans", zorder=5)
ax.add_patch(Circle((300, gy0), 15, color="#a371f7", zorder=5))
ax.text(324, gy0 - 6, "purple", fontsize=13.5, color="#cdd3ea", family="DejaVu Sans", zorder=5)

ax.add_patch(Circle((1108, 508), 28, color="#E1A100", zorder=6))
ax.add_patch(Circle((1108, 508), 28, fill=False, lw=2, ec="#ffd95e", zorder=7))
ax.text(1108, 506, "★", fontsize=28, color="#3a2c00", ha="center", va="center", zorder=8)

card = FancyBboxPatch((632, 150), 478, 318, boxstyle="round,pad=6,rounding_size=18",
        fc="#0d1117", ec="#2a2f50", lw=1.5, zorder=4)
ax.add_patch(card)
for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.add_patch(Circle((666 + i * 24, 446), 6.5, color=c, zorder=6))

mono = "DejaVu Sans Mono"
tx = 666
ax.text(tx, 414, "$ cat results/019d84ac.json", fontsize=14, color="#7ee787", family=mono, zorder=6)
ax.text(tx, 370, '"score": 0.82069,', fontsize=14.5, color="#e6edf3", family=mono, zorder=6)
ax.text(tx, 335, '"gold_medal": ', fontsize=14.5, color="#e6edf3", family=mono, zorder=6)
ax.text(tx + 174, 335, 'true', fontsize=14.5, color="#E1A100", family=mono, zorder=6)
ax.text(tx + 221, 335, ',', fontsize=14.5, color="#e6edf3", family=mono, zorder=6)
ax.text(tx, 300, '"submission_path":', fontsize=14.5, color="#e6edf3", family=mono, zorder=6)
ax.text(tx + 22, 265, '"/tmp/tmpf205ekw1.csv"', fontsize=14.5, color="#ff7b72", family=mono, zorder=6)
ax.text(tx, 224, "# файла больше нет", fontsize=13, color="#6e7681", family=mono, style="italic", zorder=6)

fig.savefig("habr_cover.png", dpi=100)
print("saved habr_cover.png")
