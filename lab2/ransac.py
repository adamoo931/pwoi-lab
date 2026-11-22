"""Ćwiczenie 2 – RANSAC (Random Sample Consensus) dla płaszczyzny w przestrzeni 3D.

Algorytm (wg https://en.wikipedia.org/wiki/Random_sample_consensus):
  1. Wylosuj minimalny podzbiór danych (dla płaszczyzny – 3 niewspółliniowe punkty).
  2. Wyznacz model (płaszczyznę n·x + d = 0, |n| = 1) z tego podzbioru.
  3. Policz punkty zgodne z modelem (inliery) – odległość punktu od płaszczyzny < próg.
  4. Zapamiętaj model z największą liczbą inlierów.
  5. Powtarzaj kroki 1-4 (liczba iteracji ograniczona z góry i adaptacyjnie zmniejszana).
  6. (usprawnienie) dopasuj płaszczyznę metodą najmniejszych kwadratów do inlierów
     najlepszego modelu i ponownie wyznacz inliery.
"""
import csv

import numpy as np


# --------------------------------------------------------------------------- I/O
def read_xyz_csv(path, delimiter=" "):
    """Wczytuje plik .xyz za pomocą pakietu csv. Zwraca tablicę (N, liczba_kolumn)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter=delimiter, skipinitialspace=True):
            if row:
                rows.append([float(v) for v in row if v != ""])
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------- geometria
def orient_normal(normal):
    """Ujednolica znak wektora normalnego (składowa Z >= 0; przy nz = 0 pierwsza niezerowa składowa > 0)."""
    n = np.asarray(normal, dtype=float)
    for comp in (n[2], n[0], n[1]):
        if abs(comp) > 1e-12:
            return n if comp > 0 else -n
    return n


def point_plane_distance(points, normal, d):
    """Odległości punktów od płaszczyzny n·x + d = 0 (|n| = 1)."""
    return np.abs(points @ normal + d)


def fit_plane_lsq(points):
    """Płaszczyzna najmniejszych kwadratów (ortogonalna): wektor własny macierzy kowariancji o najmniejszej wartości własnej."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    eigval, eigvec = np.linalg.eigh(centered.T @ centered)
    normal = eigvec[:, 0]
    return normal, -normal @ centroid


def ransac_plane(points, threshold=0.02, max_iterations=1000, min_iterations=100,
                 confidence=0.999, refine_steps=3, rng=None):
    """Dopasowanie płaszczyzny do chmury punktów algorytmem RANSAC.

    Parametry
    ---------
    points         : (N, 3) współrzędne punktów,
    threshold      : maksymalna odległość punktu od płaszczyzny, by uznać go za inlier,
    max_iterations : górna granica liczby iteracji,
    min_iterations : dolna granica liczby iteracji przy adaptacyjnym zatrzymaniu,
    confidence     : żądane prawdopodobieństwo wylosowania choć jednej próbki złożonej z samych inlierów
                     (steruje adaptacyjną liczbą iteracji  k = log(1-p) / log(1-w^3)),
    refine_steps   : liczba kroków doprecyzowania modelu metodą najmniejszych kwadratów,
    rng            : ziarno / generator liczb losowych.

    Zwraca słownik: normal, d, inliers (maska bool), n_inliers, iterations.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n_pts = len(points)
    if n_pts < 3:
        raise ValueError("Do dopasowania płaszczyzny potrzeba co najmniej 3 punktów")

    best_count, best_model = -1, None
    k_needed, it = max_iterations, 0
    while it < k_needed:
        it += 1
        p0, p1, p2 = points[rng.choice(n_pts, size=3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:                      # punkty współliniowe – próbka zdegenerowana
            continue
        normal /= norm
        d = -normal @ p0
        count = int(np.count_nonzero(np.abs(points @ normal + d) < threshold))
        if count > best_count:
            best_count, best_model = count, (normal, d)
            w = count / n_pts                 # bieżący udział inlierów
            if confidence < 1.0 and 0 < w < 1:
                k_needed = int(np.ceil(np.log(1 - confidence) / np.log(1 - w ** 3)))
                k_needed = min(max_iterations, max(min_iterations, k_needed))
            elif w >= 1:
                k_needed = min_iterations

    if best_model is None:
        raise RuntimeError("RANSAC nie znalazł żadnego niezdegenerowanego modelu")
    normal, d = best_model
    inliers = np.abs(points @ normal + d) < threshold

    for _ in range(refine_steps):             # doprecyzowanie metodą najmniejszych kwadratów
        if inliers.sum() < 3:
            break
        normal, d = fit_plane_lsq(points[inliers])
        inliers = np.abs(points @ normal + d) < threshold

    normal = np.asarray(normal)
    sign = 1.0 if np.allclose(orient_normal(normal), normal) else -1.0
    normal, d = sign * normal, sign * d
    return {"normal": normal, "d": float(d), "inliers": inliers,
            "n_inliers": int(inliers.sum()), "iterations": it}


# --------------------------------------------------------------------------- klasyfikacja
def angle_to_vertical_axis(normal):
    """Kąt [stopnie] między wektorem normalnym a osią Z (0° = płaszczyzna pozioma, 90° = pionowa)."""
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))


def classify_orientation(normal, tol_deg=10.0):
    """'pozioma' (normalna ~ Z), 'pionowa' (normalna ~ prostopadła do Z) lub 'pochyła'."""
    ang = angle_to_vertical_axis(normal)
    if ang <= tol_deg:
        return "pozioma"
    if ang >= 90.0 - tol_deg:
        return "pionowa"
    return "pochyła"
