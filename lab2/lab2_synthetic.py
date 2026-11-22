import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from ransac import (read_xyz_csv, ransac_plane, point_plane_distance,
                    classify_orientation, angle_to_vertical_axis)

HERE = Path(__file__).parent
SCENE = HERE.parent / "lab1" / "results" / "scene_3_surfaces.xyz"
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

THRESHOLD = 0.05          # [m] próg inliera w RANSAC (~2,5 σ szumu 0,02 m)
PLANE_MEAN_DIST = 0.05    # [m] cluster uznajemy za płaszczyznę, gdy średnia odległość WSZYSTKICH punktów od płaszczyzny < 5 cm
SEED = 0

# --- zadanie 2: wczytanie pliku xyz pakietem csv -------------------------------
pts = read_xyz_csv(SCENE)
print(f"Wczytano {len(pts)} punktów z {SCENE.name}, kształt {pts.shape}")

# prawdziwe etykiety (znamy kolejność łączenia chmur w ćw. 1: 5000 poz., 5000 pion., 6000 walec)
truth = np.repeat([0, 1, 2], [5000, 5000, 6000])

# --- zadanie 3: k-średnich (k = 3) ---------------------------------------------
km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(pts)
labels = km.labels_
# zgodność z prawdą: dla każdego klastra najczęstsza klasa prawdziwa
purity = sum(np.bincount(truth[labels == c]).max() for c in range(3)) / len(pts)
print(f"Środki klastrów:\n{np.round(km.cluster_centers_, 2)}\nCzystość podziału k-średnich: {purity:.4f}")

# --- zadanie 4: RANSAC dla każdego klastra --------------------------------------
results = []
for c in range(3):
    cl = pts[labels == c]
    m = ransac_plane(cl, threshold=THRESHOLD, max_iterations=1000, rng=SEED)
    dist = point_plane_distance(cl, m["normal"], m["d"])
    mean_dist, inl_ratio = float(dist.mean()), m["n_inliers"] / len(cl)
    is_plane = mean_dist < PLANE_MEAN_DIST
    orient = classify_orientation(m["normal"]) if is_plane else "—"
    true_cls = ["pozioma", "pionowa", "walec"][np.bincount(truth[labels == c]).argmax()]
    results.append(dict(cluster=c, n_points=len(cl), centroid=km.cluster_centers_[c].round(3).tolist(),
                        normal=m["normal"].round(4).tolist(), d=round(m["d"], 4),
                        n_inliers=m["n_inliers"], inlier_ratio=round(inl_ratio, 4),
                        mean_dist=round(mean_dist, 4), is_plane=bool(is_plane),
                        orientation=orient, angle_to_Z_deg=round(angle_to_vertical_axis(m["normal"]), 2),
                        iterations=m["iterations"], truth=true_cls))
    print(f"\nKlaster {c}: {len(cl)} pkt, wektor normalny = {np.round(m['normal'], 4)}, d = {m['d']:.4f}")
    print(f"   inliery: {m['n_inliers']} ({100 * inl_ratio:.1f}%), średnia odległość od płaszczyzny: {mean_dist:.4f} m")
    print(f"   -> {'PŁASZCZYZNA, ' + orient if is_plane else 'NIE jest płaszczyzną'}   (rzeczywista powierzchnia: {true_cls})")

# --- zadanie 5/6: weryfikacja pakietem pyransac3d --------------------------------
import pyransac3d as pyrsc
random.seed(SEED)            # pyransac3d korzysta z modułu random – ustawiamy ziarno dla powtarzalności
check = []
for c in range(3):
    cl = pts[labels == c]
    eq, inl = pyrsc.Plane().fit(cl, thresh=THRESHOLD, maxIteration=1000)
    n_ref = np.array(eq[:3]); n_own = np.array(results[c]["normal"])
    ang = float(np.degrees(np.arccos(np.clip(abs(n_ref @ n_own) / np.linalg.norm(n_ref), 0, 1))))
    check.append(dict(cluster=c, pyransac3d_normal=np.round(n_ref / np.linalg.norm(n_ref), 4).tolist(),
                      pyransac3d_inliers=len(inl), angle_between_normals_deg=round(ang, 3)))
cyl_cl = pts[labels == [r["truth"] for r in results].index("walec")]
cen, axis, rad, inl = pyrsc.Cylinder().fit(cyl_cl, thresh=0.05, maxIteration=10000)
cyl = dict(center=np.round(cen, 3).tolist(), axis=np.round(axis, 4).tolist(), radius=round(float(rad), 4), n_inliers=len(inl),
           n_points=len(cyl_cl))
print("\npyransac3d – porównanie normalnych:", json.dumps(check, ensure_ascii=False))
print("pyransac3d Cylinder:", cyl)

(RES / "lab2_synthetic_results.json").write_text(json.dumps(
    dict(threshold=THRESHOLD, plane_mean_dist_tol=PLANE_MEAN_DIST, kmeans_purity=purity,
         kmeans_centers=km.cluster_centers_.round(3).tolist(), clusters=results,
         pyransac3d_planes=check, pyransac3d_cylinder=cyl), indent=2, ensure_ascii=False))

# --- rysunki -------------------------------------------------------------------
cols = ["#1f77b4", "#2ca02c", "#d62728"]
fig = plt.figure(figsize=(7.2, 5.6))
ax = fig.add_subplot(111, projection="3d")
for c in range(3):
    p = pts[labels == c]
    ax.scatter(*p.T, s=1.0, c=cols[c], alpha=0.55, depthshade=False,
               label=f"klaster {c} ({results[c]['truth']})")
ax.scatter(*km.cluster_centers_.T, s=60, c="k", marker="X")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
span = pts.max(0) - pts.min(0); ax.set_box_aspect(span / span.max()); ax.view_init(22, -58)
handles = [plt.Line2D([], [], marker="o", ls="", color=cols[c], label=f"klaster {c} ({results[c]['truth']})") for c in range(3)]
handles.append(plt.Line2D([], [], marker="X", ls="", color="k", label="środki k-średnich"))
ax.legend(handles=handles, loc="upper left", fontsize=8)
ax.set_title("Segmentacja k-średnich (k = 3)", fontsize=10)
fig.savefig(FIG / "lab2_kmeans.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=False)
for c, a in enumerate(axs):
    cl = pts[labels == c]
    m = ransac_plane(cl, threshold=THRESHOLD, rng=SEED)
    dist = point_plane_distance(cl, m["normal"], m["d"])
    a.hist(dist, bins=60, color=cols[c], alpha=0.85)
    a.axvline(PLANE_MEAN_DIST, color="k", ls="--", lw=1)
    a.axvline(dist.mean(), color="orange", lw=1.5)
    a.set_title(f"klaster {c}: średnia = {dist.mean():.3f} m", fontsize=9)
    a.set_xlabel("odległość od płaszczyzny RANSAC [m]")
axs[0].set_ylabel("liczba punktów")
fig.tight_layout()
fig.savefig(FIG / "lab2_distance_hist.png", dpi=170)
plt.close(fig)
print("Zapisano wyniki i rysunki.")
