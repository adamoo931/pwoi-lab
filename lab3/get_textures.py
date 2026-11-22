from pathlib import Path

from skimage import data, io

HERE = Path(__file__).parent
SOURCES = {"cegla": data.brick, "trawa": data.grass, "zwir": data.gravel}

for name, loader in SOURCES.items():
    out = HERE / "textures" / name
    out.mkdir(parents=True, exist_ok=True)
    io.imsave(out / f"{name}_01.png", loader(), check_contrast=False)
    print("zapisano", out / f"{name}_01.png")
