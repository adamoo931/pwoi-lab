"""Ćwiczenie 1 – generatory chmur punktów w przestrzeni 3D.

Przyjęta orientacja układu odniesienia (zgodnie z instrukcją):
    X – w prawo, Y – w głąb sceny, Z – w górę (układ prawoskrętny).

Każda funkcja zwraca tablicę numpy o kształcie (n_points, 3) z kolumnami x, y, z.
Opcjonalny szum gaussowski (noise_std) symuluje błąd pomiarowy skanera.
"""
import numpy as np


def _rng(rng):
    """Zwraca generator liczb losowych (przyjmuje None, seed lub Generator)."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _add_noise(points, noise_std, rng):
    if noise_std and noise_std > 0:
        points = points + rng.normal(0.0, noise_std, size=points.shape)
    return points


def horizontal_plane(width, length, n_points, z=0.0, center=(0.0, 0.0),
                     noise_std=0.0, rng=None):
    """Płaska pozioma powierzchnia (równoległa do płaszczyzny XY, wektor normalny ~ Z).

    width  – rozmiar wzdłuż osi X, length – rozmiar wzdłuż osi Y,
    z      – wysokość powierzchni, center – środek prostokąta (x, y).
    """
    rng = _rng(rng)
    x = rng.uniform(-width / 2, width / 2, n_points) + center[0]
    y = rng.uniform(-length / 2, length / 2, n_points) + center[1]
    pts = np.column_stack([x, y, np.full(n_points, float(z))])
    return _add_noise(pts, noise_std, rng)


def vertical_plane(width, height, n_points, y=0.0, center_x=0.0, z_min=0.0,
                   noise_std=0.0, rng=None):
    """Płaska pionowa powierzchnia (równoległa do płaszczyzny XZ, wektor normalny ~ Y).

    width – rozmiar wzdłuż osi X, height – rozmiar wzdłuż osi Z (od z_min do z_min+height),
    y     – położenie płaszczyzny na osi Y, center_x – środek ściany na osi X.
    """
    rng = _rng(rng)
    x = rng.uniform(-width / 2, width / 2, n_points) + center_x
    z = rng.uniform(0.0, height, n_points) + z_min
    pts = np.column_stack([x, np.full(n_points, float(y)), z])
    return _add_noise(pts, noise_std, rng)


def cylinder(radius, height, n_points, center=(0.0, 0.0), z_min=0.0,
             noise_std=0.0, rng=None):
    """Powierzchnia boczna walca o pionowej osi (równoległej do Z).

    Kąt azymutalny i wysokość losowane są równomiernie, więc gęstość punktów
    na powierzchni walca jest stała.
    """
    rng = _rng(rng)
    theta = rng.uniform(0.0, 2 * np.pi, n_points)
    z = rng.uniform(0.0, height, n_points) + z_min
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    pts = np.column_stack([x, y, z])
    return _add_noise(pts, noise_std, rng)


def save_xyz(path, points, fmt="%.6f"):
    """Zapis chmury do pliku tekstowego .xyz (jedna linia = jeden punkt, kolumny rozdzielone spacją)."""
    np.savetxt(path, points, fmt=fmt, delimiter=" ")


def load_xyz(path):
    """Wczytanie pliku .xyz do tablicy numpy (x, y, z)."""
    return np.loadtxt(path, delimiter=" ")
