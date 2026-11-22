"""Ćwiczenie 3, krok 1 – obrazy tekstur do odrębnych katalogów (jedna kategoria = jeden katalog).

Uwaga: autor nie dysponował własnymi zdjęciami powierzchni, dlatego jako obrazy zastępcze użyto
trzech rzeczywistych zdjęć tekstur dostarczanych z pakietem scikit-image (licencja CC0):
    cegla – mur z cegły, trawa – trawnik, zwir – żwir (każde 512 x 512 px, skala szarości).
Aby użyć własnych zdjęć (gres, tynk, laminat...), wystarczy umieścić je w katalogach
textures/<nazwa_kategorii>/*.jpg i pominąć ten skrypt.
"""
from pathlib import Path

from skimage import data, io

HERE = Path(__file__).parent
SOURCES = {"cegla": data.brick, "trawa": data.grass, "zwir": data.gravel}

for name, loader in SOURCES.items():
    out = HERE / "textures" / name
    out.mkdir(parents=True, exist_ok=True)
    io.imsave(out / f"{name}_01.png", loader(), check_contrast=False)
    print("zapisano", out / f"{name}_01.png")
