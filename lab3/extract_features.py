"""Ćwiczenie 3, kroki 3-4 – cechy tekstury z macierzy zdarzeń (GLCM) i zapis do CSV.

Dla każdej próbki: skala szarości -> redukcja głębi jasności do 64 poziomów -> GLCM
(odległości 1, 3, 5 px; kierunki 0°, 45°, 90°, 135°; symetryczna, znormalizowana)
-> 6 cech x 3 odległości x 4 kierunki = 72 cechy + nazwa kategorii.

Wynik: results/texture_features.csv (72 kolumny cech + kolumna 'category')
       results/texture_features_meta.csv (nazwa pliku próbki w tej samej kolejności)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

HERE = Path(__file__).parent
PROPS = ["dissimilarity", "correlation", "contrast", "energy", "homogeneity", "ASM"]
DISTANCES = [1, 3, 5]
ANGLES_DEG = [0, 45, 90, 135]
ANGLES = np.deg2rad(ANGLES_DEG)
LEVELS = 64          # instrukcja: "5 bitów (64 poziomy)"; 64 poziomy = 6 bitów, przyjęto liczbę poziomów 64


def to_gray_uint8(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = img_as_ubyte(rgb2gray(a[..., :3]))
    return a.astype(np.uint8)


def feature_names():
    return [f"{p}_d{d}_a{a}" for p in PROPS for d in DISTANCES for a in ANGLES_DEG]


def glcm_features(gray_uint8, levels=LEVELS):
    """Wektor 72 cech GLCM dla obrazu w skali szarości (uint8)."""
    q = (gray_uint8 // (256 // levels)).astype(np.uint8)          # redukcja głębi jasności: 0..levels-1
    glcm = graycomatrix(q, distances=DISTANCES, angles=ANGLES, levels=levels, symmetric=True, normed=True)
    return np.concatenate([graycoprops(glcm, p).ravel() for p in PROPS])   # kolejność: cecha, odległość, kąt


def build_dataset(patches_dir, levels=LEVELS):
    rows, cats, files = [], [], []
    for cat_dir in sorted(p for p in Path(patches_dir).iterdir() if p.is_dir()):
        for f in sorted(cat_dir.glob("*.png")):
            rows.append(glcm_features(to_gray_uint8(Image.open(f)), levels))
            cats.append(cat_dir.name)
            files.append(f"{cat_dir.name}/{f.name}")
    df = pd.DataFrame(rows, columns=feature_names())
    df["category"] = cats
    return df, pd.DataFrame({"file": files, "category": cats})


if __name__ == "__main__":
    df, meta = build_dataset(HERE / "patches")
    assert not df.drop(columns="category").isna().any().any(), "NaN w cechach"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "texture_features.csv", index=False)
    meta.to_csv(out / "texture_features_meta.csv", index=False)
    print("kształt zbioru:", df.shape, "| cech:", df.shape[1] - 1)
    print(df["category"].value_counts().to_string())
    print("zapisano", out / "texture_features.csv")
