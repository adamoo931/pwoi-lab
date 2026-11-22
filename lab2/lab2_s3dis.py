import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from ransac import read_xyz_csv, ransac_plane, classify_orientation, angle_to_vertical_axis

HERE = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "conferenceRoom_1.txt"   # plik S3DIS (nie jest dołączony do repozytorium)
RES, FIG = HERE / "results", HERE / "figures"
OUT = RES / "s3dis"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)

K = 6                  # liczba iteracji (z instrukcji)
THRESHOLD = 0.02       # [m] próg inliera (2 cm; mediana odstępu najbliższych punktów ≈ 5 mm)
MAX_IT = 1000
SEED = 0

t0 = time.time()
data = read_xyz_csv(SRC)
xyz, rgb = data[:, :3], data[:, 3:6]
N = len(xyz)
print(f"Wczytano {N} punktów w {time.time() - t0:.1f} s; zakres X {xyz[:,0].min():.2f}..{xyz[:,0].max():.2f}, "
      f"Y {xyz[:,1].min():.2f}..{xyz[:,1].max():.2f}, Z {xyz[:,2].min():.2f}..{xyz[:,2].max():.2f}")


def iterative_planes(xyz, k, thr, seed):
    """Zwraca modele kolejno wykrytych płaszczyzn i indeksy punktów pozostałych po k iteracjach."""
    remaining = np.arange(len(xyz))
    models = []
    for i in range(k):
        t = time.time()
        m = ransac_plane(xyz[remaining], threshold=thr,
                         max_iterations=MAX_IT, rng=seed + i)
        mask = m.pop("inliers")          # maska inlierów względem "remaining"
        m["idx"] = remaining[mask]
        m["time_s"] = time.time() - t
        models.append(m)
        remaining = remaining[~mask]     # usunięcie punktów płaszczyznowych
    return models, remaining


# --- główna procedura ---------------------------------------------------------
models, remaining = iterative_planes(xyz, K, THRESHOLD, SEED)

planes = []
for i, m in enumerate(models, 1):
    pts = xyz[m["idx"]]
    res = pts @ m["normal"] + m["d"]
    n = m["normal"]
    orient = classify_orientation(n)
    dom = int(np.argmax(np.abs(n)))                      # oś, wzdłuż której leży normalna
    cen = pts.mean(axis=0)                               # środek ciężkości inlierów
    others = [j for j in range(3) if j != dom]
    # położenie płaszczyzny wzdłuż osi dom, liczone nad środkiem ciężkości inlierów
    # (przecięcie z osią układu byłoby zniekształcone przez niewielkie pochylenie płaszczyzny)
    position = float(-(m["d"] + sum(n[j] * cen[j] for j in others)) / n[dom])
    planes.append(dict(
        k=i, n_points=int(m["n_inliers"]), share_percent=round(100 * m["n_inliers"] / N, 2),
        normal=n.round(4).tolist(), d=round(m["d"], 4), orientation=orient, centroid=cen.round(4).tolist(),
        angle_to_Z_deg=round(angle_to_vertical_axis(n), 2), dominant_axis="XYZ"[dom], position=round(position, 3),
        rms_residual_m=round(float(np.sqrt(np.mean(res ** 2))), 4),
        bbox_min=pts.min(0).round(3).tolist(), bbox_max=pts.max(0).round(3).tolist(),
        ransac_iterations=m["iterations"], time_s=round(m["time_s"], 2)))

# nazwy: poziome – podłoga (najniżej) / sufit (najwyżej) / inna pozioma; pionowe – ściany
horiz = sorted([p for p in planes if p["orientation"] == "pozioma"], key=lambda p: p["position"])
for j, p in enumerate(horiz):
    p["label"] = "podłoga" if j == 0 else ("sufit" if j == len(horiz) - 1 else f"powierzchnia pozioma (z = {p['position']:.2f} m)")
walls = [p for p in planes if p["orientation"] == "pionowa"]
for p in walls:
    p["label"] = f"ściana {p['dominant_axis']} = {p['position']:.2f} m"
for p in planes:
    p.setdefault("label", "pochyła")

print(f"\n{'k':>2} {'punkty':>8} {'%':>6} {'normalna':>26} {'kąt do Z':>9} {'RMS':>7}  typ")
for p in planes:
    print(f"{p['k']:>2} {p['n_points']:>8} {p['share_percent']:>6.2f} {str(p['normal']):>26} {p['angle_to_Z_deg']:>8.2f}° "
          f"{p['rms_residual_m']:>7.4f}  {p['label']}  [{p['time_s']} s]")
extracted = sum(p["n_points"] for p in planes)
print(f"Łącznie punktów płaszczyznowych: {extracted} ({100 * extracted / N:.1f}%), pozostało {len(remaining)} ({100 * len(remaining) / N:.1f}%)")

# wymiary pomieszczenia wynikające z płaszczyzn
dims = {}
floor = next((p for p in planes if p["label"] == "podłoga"), None)
ceil = next((p for p in planes if p["label"] == "sufit"), None)
if floor and ceil:
    dims["wysokosc_m"] = round(ceil["position"] - floor["position"], 3)
for ax_name in "XY":
    ws = sorted([p for p in walls if p["dominant_axis"] == ax_name], key=lambda p: p["position"])
    if len(ws) >= 2:
        dims[f"odleglosc_scian_{ax_name}_m"] = round(ws[-1]["position"] - ws[0]["position"], 3)
dims["bbox_chmury_m"] = (xyz.max(0) - xyz.min(0)).round(3).tolist()
print("Wymiary z płaszczyzn:", dims)

# --- zapis chmur (CloudCompare) -----------------------------------------------
fmt = "%.3f %.3f %.3f %d %d %d"
for p, m in zip(planes, models):
    np.savetxt(OUT / f"plane_{p['k']}.xyz", np.hstack([xyz[m["idx"]], rgb[m["idx"]]]), fmt=fmt)
np.savetxt(OUT / "remaining.xyz", np.hstack([xyz[remaining], rgb[remaining]]), fmt=fmt)

# --- weryfikacja: pyransac3d (ta sama procedura iteracyjna) ---------------------
import pyransac3d as pyrsc
random.seed(SEED)
rem = np.arange(N)
check = []
for p in planes:
    t = time.time()
    eq, inl = pyrsc.Plane().fit(xyz[rem], thresh=THRESHOLD, maxIteration=MAX_IT)
    n_ref = np.array(eq[:3]) / np.linalg.norm(eq[:3])
    d_ref = eq[3] / np.linalg.norm(eq[:3])
    offset = float(abs(n_ref @ np.array(p["centroid"]) + d_ref))   # odl. środka ciężkości inlierów od płaszczyzny pyransac3d [m]
    ang = float(np.degrees(np.arccos(np.clip(abs(n_ref @ np.array(p["normal"])), 0, 1))))
    check.append(dict(k=p["k"], pyransac3d_normal=n_ref.round(4).tolist(), pyransac3d_inliers=len(inl),
                      angle_between_normals_deg=round(ang, 3), centroid_offset_mm=round(1000 * offset, 2),
                      time_s=round(time.time() - t, 2)))
    mask = np.ones(len(rem), bool); mask[inl] = False
    rem = rem[mask]
    print(f"pyransac3d k={p['k']}: inliery {len(inl)} (własny: {p['n_points']}), kąt między normalnymi {ang:.3f}°, odl. środka ciężkości {1000 * offset:.1f} mm")

# --- wpływ progu na wynik --------------------------------------------------------
sens = []
for thr in (0.01, 0.02, 0.03, 0.05):
    ms, rest = iterative_planes(xyz, K, thr, SEED)
    sens.append(dict(threshold=thr, counts=[int(m["n_inliers"]) for m in ms],
                     extracted_percent=round(100 * (N - len(rest)) / N, 1)))
    print(f"próg {thr} m -> {[int(m['n_inliers']) for m in ms]}, wyodrębniono {sens[-1]['extracted_percent']}%")

(RES / "lab2_s3dis_results.json").write_text(json.dumps(
    dict(source=SRC.name, n_points=N, K=K, threshold=THRESHOLD, max_iterations=MAX_IT, seed=SEED,
         planes=planes, extracted=extracted, remaining=int(len(remaining)), room_dims=dims,
         pyransac3d=check, threshold_sensitivity=sens), indent=2, ensure_ascii=False))

print(f"Gotowe w {time.time() - t0:.0f} s. Rysunki: python lab2_s3dis_figures.py")
