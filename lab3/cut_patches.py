"""Ćwiczenie 3, krok 2 – wycinanie próbek tekstury (fragmentów obrazu) o zadanym rozmiarze.

Funkcja cut_patches wczytuje obraz, skaluje go do zadanego rozmiaru (parametr image_size),
a następnie tnie siatką na fragmenty patch_size x patch_size i zapisuje je w katalogu kategorii.

Uruchomienie:  python cut_patches.py [--image-size 512] [--patch-size 64] [--stride 64]
"""
import argparse
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cut_patches(image_path, out_dir, image_size=(512, 512), patch_size=64, stride=None):
    """Wycina próbki z jednego obrazu.

    image_path : ścieżka do obrazu,
    out_dir    : katalog docelowy (kategoria tekstury),
    image_size : (szerokość, wysokość) do jakiej skalowany jest wczytany obraz,
    patch_size : bok kwadratowej próbki [px],
    stride     : krok siatki [px]; domyślnie = patch_size (próbki bez nakładania).
    Zwraca liczbę zapisanych próbek.
    """
    stride = stride or patch_size
    image_path, out_dir = Path(image_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    if img.size != tuple(image_size):
        img = img.resize(tuple(image_size), Image.BICUBIC)
    w, h = img.size
    n = 0
    for r in range(0, h - patch_size + 1, stride):
        for c in range(0, w - patch_size + 1, stride):
            patch = img.crop((c, r, c + patch_size, r + patch_size))
            patch.save(out_dir / f"{image_path.stem}_r{r:04d}_c{c:04d}.png")
            n += 1
    return n


def cut_all(textures_dir, patches_dir, image_size, patch_size, stride):
    """Przetwarza wszystkie kategorie (podkatalogi textures_dir)."""
    counts = {}
    for cat_dir in sorted(p for p in Path(textures_dir).iterdir() if p.is_dir()):
        counts[cat_dir.name] = sum(
            cut_patches(f, Path(patches_dir) / cat_dir.name, image_size, patch_size, stride)
            for f in sorted(cat_dir.iterdir()) if f.suffix.lower() in IMG_EXT)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=512, help="bok kwadratu, do którego skalowane są obrazy")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=None)
    a = ap.parse_args()
    counts = cut_all(HERE / "textures", HERE / "patches", (a.image_size, a.image_size), a.patch_size, a.stride)
    print("liczba próbek w kategoriach:", counts, "razem:", sum(counts.values()))
