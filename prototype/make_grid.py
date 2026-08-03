# Spike: build the labeled comparison grid.
# Rows = creatures. Cols = {current, noobai} x {front, back} x {NEAREST, k-centroid},
# every cell a true 64x64 sprite upscaled x6 for viewing.
# Both downscalers start from the SAME P-mode 768 artifact (the pipeline's real
# integration point: saved quantized sprites feed stitch_spritesheet).
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from fakemon_forge.sprites import postprocess, _KEY_COLOR
from kcentroid import k_centroid
from spike_prompts import CREATURES

HERE = Path(__file__).parent
CUR = HERE / "out" / "current"
NB = HERE / "out" / "noobai"
CELL = 64
UP = 6


def to_64(p_mode_768: Image.Image, method: str) -> Image.Image:
    if method == "nearest":
        return p_mode_768.convert("RGB").resize((CELL, CELL), Image.NEAREST)
    small = k_centroid(p_mode_768.convert("RGB"), CELL, CELL)
    # Re-quantize at 64 so k-centroid's blended colours snap back to a 16-colour
    # Gen-3 palette (postprocess handles key/black/white reservation).
    return postprocess(small, size=CELL).convert("RGB")


def split_canvas(canvas: Image.Image) -> tuple[Image.Image, Image.Image]:
    # Landscape pair: front on the left, back on the right (verify per-canvas).
    w, h = canvas.size
    return canvas.crop((0, 0, w // 2, h)), canvas.crop((w // 2, 0, w, h))


def cell_or_blank(img: Image.Image | None, method: str) -> Image.Image:
    if img is None:
        return Image.new("RGB", (CELL, CELL), (90, 90, 90))
    return to_64(img, method)


COLS = ["cur front N", "cur front K", "nb front N", "nb front K",
        "cur back N", "cur back K", "nb back N", "nb back K"]

rows = []
for name, _types, _desc in CREATURES:
    cur_front = cur_back = nb_front = nb_back = None
    if (CUR / f"{name}_front.png").exists():
        cur_front = Image.open(CUR / f"{name}_front.png")
    if (CUR / f"{name}_back.png").exists():
        cur_back = Image.open(CUR / f"{name}_back.png")
    if (NB / f"{name}_canvas.png").exists():
        canvas = Image.open(NB / f"{name}_canvas.png")
        front_half, back_half = split_canvas(canvas)
        # Raw RGB halves -> the same production quantize the current sprites got.
        nb_front = postprocess(front_half, size=768)
        nb_back = postprocess(back_half, size=768)
    cells = [
        cell_or_blank(cur_front, "nearest"), cell_or_blank(cur_front, "kc"),
        cell_or_blank(nb_front, "nearest"), cell_or_blank(nb_front, "kc"),
        cell_or_blank(cur_back, "nearest"), cell_or_blank(cur_back, "kc"),
        cell_or_blank(nb_back, "nearest"), cell_or_blank(nb_back, "kc"),
    ]
    rows.append((name, cells))

pad, header = 8, 24
gw = len(COLS) * (CELL * UP + pad) + pad + 90
gh = header + len(rows) * (CELL * UP + pad) + pad
grid = Image.new("RGB", (gw, gh), (30, 30, 30))
draw = ImageDraw.Draw(grid)
for c, label in enumerate(COLS):
    draw.text((90 + pad + c * (CELL * UP + pad), 6), label, fill=(220, 220, 220))
for r, (name, cells) in enumerate(rows):
    y = header + pad + r * (CELL * UP + pad)
    draw.text((6, y + CELL * UP // 2), name, fill=(220, 220, 220))
    for c, cell in enumerate(cells):
        grid.paste(cell.resize((CELL * UP, CELL * UP), Image.NEAREST),
                   (90 + pad + c * (CELL * UP + pad), y))
grid.save(HERE / "out" / "comparison_grid.png")
print("grid saved:", HERE / "out" / "comparison_grid.png")
