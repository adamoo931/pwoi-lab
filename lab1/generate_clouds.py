import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from point_clouds import horizontal_plane, vertical_plane, cylinder, save_xyz

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = 42
NOISE = 0.02  # [m] odchylenie standardowe szumu pomiarowego (2 cm)

# --- parametry powierzchni (jednostki: metry) --------------------------------
H = dict(width=10.0, length=10.0, n_points=5000, z=0.0, center=(0.0, 0.0))
V = dict(width=10.0, height=10.0, n_points=5000, y=0.0, center_x=20.0, z_min=0.0)
C = dict(radius=3.0, height=10.0, n_points=6000, center=(10.0, 20.0), z_min=0.0)

rng = np.random.default_rng(SEED)
cloud_h = horizontal_plane(**H, noise_std=NOISE, rng=rng)
cloud_v = vertical_plane(**V, noise_std=NOISE, rng=rng)
cloud_c = cylinder(**C, noise_std=NOISE, rng=rng)
scene = np.vstack([cloud_h, cloud_v, cloud_c])

save_xyz(RES / "horizontal_plane.xyz", cloud_h)
save_xyz(RES / "vertical_plane.xyz", cloud_v)
save_xyz(RES / "cylinder.xyz", cloud_c)
save_xyz(RES / "scene_3_surfaces.xyz", scene)

# --- statystyki kontrolne (czy wygenerowane punkty spełniają równania powierzchni) ---
radial = np.hypot(cloud_c[:, 0] - C["center"][0], cloud_c[:, 1] - C["center"][1])
stats = {
    "seed": SEED, "noise_std": NOISE,
    "horizontal": {"n": len(cloud_h), "z_mean": float(cloud_h[:, 2].mean()), "z_std": float(cloud_h[:, 2].std()),
                   "x_range": [float(cloud_h[:, 0].min()), float(cloud_h[:, 0].max())],
                   "y_range": [float(cloud_h[:, 1].min()), float(cloud_h[:, 1].max())]},
    "vertical": {"n": len(cloud_v), "y_mean": float(cloud_v[:, 1].mean()), "y_std": float(cloud_v[:, 1].std()),
                 "x_range": [float(cloud_v[:, 0].min()), float(cloud_v[:, 0].max())],
                 "z_range": [float(cloud_v[:, 2].min()), float(cloud_v[:, 2].max())]},
    "cylinder": {"n": len(cloud_c), "r_mean": float(radial.mean()), "r_std": float(radial.std()),
                 "z_range": [float(cloud_c[:, 2].min()), float(cloud_c[:, 2].max())]},
    "scene_n": int(len(scene)),
}
(RES / "lab1_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))


# --- wizualizacja -------------------------------------------------------------
def style_axes(ax, pts, title, azim=-58):
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(title, fontsize=10)
    mins, maxs = pts.min(0), pts.max(0)
    # oś "płaska" (zakres rzędu szumu) rozszerzamy do 25% największego wymiaru, aby podziałka była czytelna
    min_span = 0.25 * (maxs - mins).max()
    centre = (mins + maxs) / 2
    half = np.maximum(maxs - mins, min_span) / 2
    flat = (maxs - mins) < min_span          # które osie są "płaskie"
    mins, maxs = centre - half, centre + half
    ax.set_box_aspect((maxs - mins) / (maxs - mins).max())
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    for i, setter in enumerate((ax.set_xticks, ax.set_yticks, ax.set_zticks)):
        if flat[i]:
            setter([round(float(centre[i]), 1)])
    ax.view_init(elev=22, azim=azim)


def single(pts, title, color, fname, azim=-58):
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*pts.T, s=1.2, c=color, alpha=0.6, depthshade=False)
    # lekko poszerzone granice dla płaszczyzn (zerowa grubość osi)
    style_axes(ax, pts, title, azim)
    fig.savefig(FIG / fname, dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


single(cloud_h, "Płaszczyzna pozioma 10 m × 10 m (z = 0)", "#1f77b4", "lab1_horizontal.png")
single(cloud_v, "Płaszczyzna pionowa 10 m × 10 m (y = 0)", "#2ca02c", "lab1_vertical.png", azim=-30)
single(cloud_c, "Powierzchnia walcowa (r = 3 m, h = 10 m)", "#d62728", "lab1_cylinder.png")

fig = plt.figure(figsize=(7.4, 6.0))
ax = fig.add_subplot(111, projection="3d")
for pts, col, lab in [(cloud_h, "#1f77b4", "pozioma"), (cloud_v, "#2ca02c", "pionowa"), (cloud_c, "#d62728", "walec")]:
    ax.scatter(*pts.T, s=1.0, c=col, alpha=0.55, label=lab, depthshade=False)
style_axes(ax, scene, "Scena złożona z trzech powierzchni (plik scene_3_surfaces.xyz)")
ax.legend(markerscale=6, loc="upper left")
fig.savefig(FIG / "lab1_scene.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print("OK – zapisano pliki do", RES, "oraz", FIG)
