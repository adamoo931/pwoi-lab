"""Ćwiczenie 2, opcja 2 – rysunki do sprawozdania (Matplotlib), na podstawie plików z lab2_s3dis.py.

Wczytuje results/s3dis/plane_k.xyz, remaining.xyz oraz results/lab2_s3dis_results.json.
W CloudCompare te same pliki otwieramy poleceniem File -> Open.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT, FIG = HERE / "results" / "s3dis", HERE / "figures"
FIG.mkdir(exist_ok=True)
info = json.loads((HERE / "results" / "lab2_s3dis_results.json").read_text())
planes = info["planes"]

rng = np.random.default_rng(1)


def load_sub(path, n):
    a = np.loadtxt(path, usecols=(0, 1, 2))
    return a[rng.choice(len(a), min(len(a), n), replace=False)]


P = {p["k"]: load_sub(OUT / f"plane_{p['k']}.xyz", 14000) for p in planes}
R = load_sub(OUT / "remaining.xyz", 20000)
ALL = np.vstack(list(P.values()) + [R])
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
GREY = "#bdbdbd"
span = ALL.max(0) - ALL.min(0)
ceil_k = next((p["k"] for p in planes if p["label"] == "sufit"), None)


def legend_handles():
    h = [plt.Line2D([], [], marker="o", ls="", color=palette[p["k"] - 1], label=f"{p['k']}. {p['label']}") for p in planes]
    h.append(plt.Line2D([], [], marker="o", ls="", color=GREY, label="punkty pozostałe"))
    return h


# ---- Rys. A: widok 3D (całość) i widok po ukryciu sufitu --------------------------
fig = plt.figure(figsize=(12.5, 5.6))
for j, (hide_ceiling, elev, azim, title) in enumerate([(False, 24, -62, "a) wszystkie wyodrębnione płaszczyzny"),
                                                       (True, 48, -62, "b) widok z góry po ukryciu sufitu")]):
    ax = fig.add_subplot(1, 2, j + 1, projection="3d")
    ax.scatter(*R.T, s=0.5, c=GREY, alpha=0.35, depthshade=False)
    for p in planes:
        if hide_ceiling and p["k"] == ceil_k:
            continue
        ax.scatter(*P[p["k"]].T, s=0.5, c=palette[p["k"] - 1], alpha=0.55, depthshade=False)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_box_aspect(span / span.max()); ax.view_init(elev, azim); ax.set_title(title, fontsize=10)
fig.legend(handles=legend_handles(), loc="lower center", ncol=7, fontsize=8, markerscale=1.4, frameon=False)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12, wspace=0.05)
fig.savefig(FIG / "lab2_s3dis_3d.png", dpi=150)
plt.close(fig)

# ---- Rys. B: rzuty prostokątne ----------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
for a, (u, v, tt) in zip(axs, [(0, 1, "Rzut z góry (XY)"), (0, 2, "Rzut z przodu (XZ)"), (1, 2, "Rzut z boku (YZ)")]):
    a.scatter(R[:, u], R[:, v], s=0.4, c=GREY, alpha=0.35)
    for p in planes:
        a.scatter(P[p["k"]][:, u], P[p["k"]][:, v], s=0.4, c=palette[p["k"] - 1], alpha=0.5)
    a.set_xlabel("XYZ"[u] + " [m]"); a.set_ylabel("XYZ"[v] + " [m]"); a.set_title(tt, fontsize=10)
    a.set_aspect("equal"); a.grid(alpha=0.3)
fig.legend(handles=legend_handles(), loc="lower center", ncol=7, fontsize=8, markerscale=1.4, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(FIG / "lab2_s3dis_projections.png", dpi=150)
plt.close(fig)

# ---- Rys. C: każda płaszczyzna osobno (rzut na płaszczyznę, w której leży) -------------
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.4))
for p, a in zip(planes, axs.ravel()):
    pts = P[p["k"]]
    dom = "XYZ".index(p["dominant_axis"])
    u, v = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[dom]
    a.scatter(pts[:, u], pts[:, v], s=0.5, c=palette[p["k"] - 1], alpha=0.7)
    a.set_title(f"{p['k']}. {p['label']}\n{p['n_points']:,} pkt ({p['share_percent']}%)".replace(",", " "), fontsize=9)
    a.set_xlabel("XYZ"[u] + " [m]"); a.set_ylabel("XYZ"[v] + " [m]"); a.set_aspect("equal"); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "lab2_s3dis_each_plane.png", dpi=150)
plt.close(fig)

# ---- Rys. D: kolejne iteracje – ile punktów zostaje w chmurze ---------------------------
N = info["n_points"]
left = [N]
for p in planes:
    left.append(left[-1] - p["n_points"])
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(range(len(left)), [100 * x / N for x in left], color=["#9e9e9e"] + [palette[i] for i in range(len(planes))])
for i, x in enumerate(left):
    ax.text(i, 100 * x / N + 1.5, f"{100 * x / N:.1f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(left)))
ax.set_xticklabels(["start"] + [f"po iter. {p['k']}" for p in planes], fontsize=8)
ax.set_ylabel("punkty pozostałe w chmurze [%]"); ax.set_ylim(0, 112)
fig.tight_layout()
fig.savefig(FIG / "lab2_s3dis_progress.png", dpi=170)
plt.close(fig)
print("Zapisano rysunki w", FIG)
