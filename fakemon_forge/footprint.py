"""Render a Gen-3 Pokédex footprint from a stage's front sprite.

The footprint is the tiny monochrome "foot stamp" the Gen-3 Pokédex shows next
to a species: a single stylized foot (a tapered pad plus a few toe marks). The
output is a **16x16 RGBA PNG** where every pixel is either opaque black
``(0, 0, 0, 255)`` or fully transparent (``alpha == 0``) — no other colours, no
partial alpha.

This module is **Pillow-only**: it must not pull torch/diffusers into
``sys.modules``. It reuses two private helpers from ``sprites.py``
(``_background_index`` / ``_content_bbox``) for background/content detection —
those are safe to import because ``sprites.py`` keeps its torch/diffusers
imports function-local.
"""

from PIL import Image, ImageDraw

from fakemon_forge.sprites import _background_index, _content_bbox

_SIZE = 16  # the fixed footprint canvas edge, in pixels.
_SUPERSAMPLE = 4  # pad is drawn at 4x then downsampled for a crisp rounded edge.

# Fraction of the content bbox's rows (measured from the bottom) that form the
# "contact patch" — the silhouette of whatever is touching the ground.
_CONTACT_BAND_FRACTION = 0.15
# Connected components below this fraction of the band's pixel area are noise.
_NOISE_FRACTION = 0.002
# Downsample greys at or above this threshold become opaque black; below it,
# fully transparent — this is what keeps the two-value contract.
_BINARIZE_THRESHOLD = 128

# Primary-type -> toe count. The first (primary) type keys the lookup; any
# unlisted type (or an empty ``types`` list) yields 0 toes (a plain pad).
_TOE_COUNT = {
    "Dragon": 3, "Fighting": 3, "Fire": 3,   # claw wedges
    "Flying": 3,                              # bird prongs
    "Normal": 4, "Ground": 4,                # round toes
    "Water": 0, "Poison": 0, "Ghost": 0, "Psychic": 0,  # plain / webbed pad
}
# Primary-type -> toe mark style. "wedge" (claw), "prong" (bird), "dot" (round).
_TOE_STYLE = {
    "Dragon": "wedge", "Fighting": "wedge", "Fire": "wedge",
    "Flying": "prong",
    "Normal": "dot", "Ground": "dot",
}

# Contour override: an unusually toe-shaped silhouette can speak for itself.
# Expected to fire rarely — thresholds are deliberately conservative.
_CONTOUR_MIN_BUMPS = 2      # need at least this many upward bumps to override.
_CONTOUR_MIN_DEPTH = 2      # a bump must dip at least this many px to count.
_CONTOUR_MIN_COVERAGE = 0.6  # blob must fill this fraction of its width band.
_CONTOUR_MAX_TOES = 5


def generate_footprint(
    sprite_path: str,
    output_path: str,
    *,
    types: list[str],
    size_fraction: float = 0.9,
    blank: bool = False,
) -> None:
    """Render a 16x16 Pokédex footprint for a stage and write it to ``output_path``.

    ``sprite_path`` is the stage's ``sprite.png`` (P-mode, native 768, flat
    backdrop). ``types`` is the stage's type list — the first (primary) type
    keys the toe lookup. ``size_fraction`` is the fraction of the canvas the
    footprint's long axis spans. When ``blank`` is true, an all-transparent
    footprint is written and the sprite is never read.
    """
    if blank:
        _blank().save(output_path)
        return

    image = Image.open(sprite_path)
    if image.mode != "P":
        raise ValueError(f"Expected palette-mode sprite image, got {image.mode}")

    background = _background_index(image)
    bbox = _content_bbox(image, background)
    if bbox is None:
        _blank().save(output_path)
        return

    blob = _pick_foot_blob(image, background, bbox)
    if blob is None:
        _blank().save(output_path)
        return

    content_w = bbox[2] - bbox[0]
    width_ratio = (blob["width"] / content_w) if content_w else 0.0

    primary = types[0] if types else None
    toe_count = _TOE_COUNT.get(primary, 0)
    toe_style = _TOE_STYLE.get(primary, "dot")
    contour_toes = _contour_bump_count(blob)
    if contour_toes >= _CONTOUR_MIN_BUMPS:
        toe_count = contour_toes  # a strong bump signal overrides the table.

    footprint = _render(width_ratio, toe_count, toe_style, size_fraction)
    footprint.save(output_path)


def _blank() -> Image.Image:
    """An all-transparent 16x16 RGBA canvas."""
    return Image.new("RGBA", (_SIZE, _SIZE), (0, 0, 0, 0))


def _pick_foot_blob(image: Image.Image, background: int, bbox):
    """Return the widest non-noise contact-patch blob, or ``None``.

    The contact patch is the bottom ``_CONTACT_BAND_FRACTION`` of the content
    bbox's rows. Connected components (8-connectivity) below
    ``_NOISE_FRACTION`` of the band area are discarded; the widest survivor is
    "the foot". Returns a dict with its ``width`` and per-column top contour, or
    ``None`` when nothing survives.
    """
    left, top, right, bottom = bbox
    content_h = bottom - top
    band_h = max(1, round(content_h * _CONTACT_BAND_FRACTION))
    band_top = bottom - band_h
    band = image.crop((left, band_top, right, bottom))
    bw, bh = band.size
    data = band.get_flattened_data()

    foreground = {(i % bw, i // bw) for i, v in enumerate(data) if v != background}
    if not foreground:
        return None

    min_area = bw * bh * _NOISE_FRACTION
    seen: set = set()
    best = None
    for start in foreground:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    if neighbour in foreground and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        if len(component) < min_area:
            continue
        xs = [p[0] for p in component]
        width = max(xs) - min(xs) + 1
        if best is None or width > best["width"]:
            best = {"width": width, "component": component}

    if best is None:
        return None

    # Per-column top contour of the chosen blob (topmost y for each column).
    top_by_col: dict = {}
    for x, y in best["component"]:
        if x not in top_by_col or y < top_by_col[x]:
            top_by_col[x] = y
    best["top_contour"] = top_by_col
    return best


def _contour_bump_count(blob) -> int:
    """Count strong upward bumps in the chosen blob's top contour.

    A "bump" is a run of columns rising above and dipping back below a local
    baseline by at least ``_CONTOUR_MIN_DEPTH`` px. Only counts when the blob
    fills at least ``_CONTOUR_MIN_COVERAGE`` of its width span (a mostly-genuine
    contour). Deliberately conservative — expected to fire rarely.
    """
    contour = blob["top_contour"]
    if len(contour) < 3:
        return 0
    xmin, xmax = min(contour), max(contour)
    span = xmax - xmin + 1
    if span < 3 or len(contour) / span < _CONTOUR_MIN_COVERAGE:
        return 0

    heights = [contour[x] for x in range(xmin, xmax + 1) if x in contour]
    if len(heights) < 3:
        return 0
    baseline = max(heights)  # the lowest row reached (largest y) is "the ground".

    bumps = 0
    in_bump = False
    for h in heights:
        rising = (baseline - h) >= _CONTOUR_MIN_DEPTH
        if rising and not in_bump:
            bumps += 1
            in_bump = True
        elif not rising:
            in_bump = False
    return min(bumps, _CONTOUR_MAX_TOES)


def _render(width_ratio: float, toe_count: int, toe_style: str,
            size_fraction: float) -> Image.Image:
    """Compose the tapered pad (supersampled) plus toe marks stamped at 16 px."""
    long_axis = max(3, min(_SIZE, round(size_fraction * _SIZE)))
    foot_top = (_SIZE - long_axis) // 2
    foot_bottom = foot_top + long_axis
    cx = _SIZE // 2

    # Toe band on top, a >=1 px transparent gap, then the pad below. Drop the
    # toe band if there is no room to keep the pad legible.
    toe_h = 2 if toe_count > 0 else 0
    gap = 1 if toe_count > 0 else 0
    if long_axis - toe_h - gap < 3:
        toe_h = gap = 0
        toe_count = 0
    pad_top = foot_top + toe_h + gap
    pad_bottom = foot_bottom

    # Pad width scales with the blob's width relative to the content, clamped to
    # a minimum so a foot is always legible. A tiny width_ratio therefore yields
    # a narrow-but-tall pad — the intended hoof-print oval.
    max_pad_w = long_axis
    min_pad_w = 3
    pad_w = max(min_pad_w, min(max_pad_w, round(width_ratio * max_pad_w)))

    mask = _draw_pad(pad_w, pad_top, pad_bottom, cx)
    result = Image.new("RGBA", (_SIZE, _SIZE), (0, 0, 0, 0))
    rpx = result.load()
    mpx = mask.load()
    for y in range(_SIZE):
        for x in range(_SIZE):
            if mpx[x, y] >= _BINARIZE_THRESHOLD:
                rpx[x, y] = (0, 0, 0, 255)

    _stamp_toes(rpx, toe_count, toe_style, cx, foot_top, toe_h, pad_w)
    return result


def _draw_pad(pad_w: int, pad_top: int, pad_bottom: int, cx: int) -> Image.Image:
    """Draw a tapered pad supersampled, then downsample+binarize to a 16 px mask.

    Two overlapping ellipses — a wide upper "shoulder" and a narrower lower
    "base" — form the classic teardrop pad (wide near the toes, narrower at the
    base). Drawing at ``_SUPERSAMPLE``x and downsampling gives a rounded edge;
    the threshold in the caller keeps the result two-valued.
    """
    ss = _SUPERSAMPLE
    canvas = Image.new("L", (_SIZE * ss, _SIZE * ss), 0)
    draw = ImageDraw.Draw(canvas)

    pad_h = pad_bottom - pad_top
    half = pad_w * ss / 2.0
    cxs = cx * ss
    top = pad_top * ss
    bottom = pad_bottom * ss

    # Wide shoulders across the upper ~75% of the pad.
    shoulder_bottom = top + pad_h * ss * 0.75
    draw.ellipse([cxs - half, top, cxs + half, shoulder_bottom], fill=255)
    # Narrower base across the lower part, giving the taper.
    base_half = max(1.0, half * 0.6)
    base_top = top + pad_h * ss * 0.35
    draw.ellipse([cxs - base_half, base_top, cxs + base_half, bottom], fill=255)

    small = canvas.resize((_SIZE, _SIZE), Image.BILINEAR)
    return small.point(lambda p: 255 if p >= _BINARIZE_THRESHOLD else 0)


def _toe_columns(style: str, dy: int) -> list[int]:
    """Horizontal pixel offsets for a toe of ``style`` at toe-row ``dy`` (0 = top).

    Every toe is 1 px wide so distinct toes never fuse; only the vertical extent
    differs by style ("dot" is a single top pixel, wedge/prong run the full toe
    height).
    """
    if style == "dot" and dy > 0:
        return []
    return [0]


def _stamp_toes(rpx, toe_count: int, style: str, cx: int, foot_top: int,
                toe_h: int, pad_w: int) -> None:
    """Stamp ``toe_count`` toe marks directly on the 16 px grid above the pad.

    Toes are placed in rows ``[foot_top, foot_top + toe_h)`` — the pad starts a
    full gap row below, so the guaranteed transparent gap between toe tips and
    the pad edge is preserved. Spacing is kept >=2 px so toes read as separate
    marks; positions are clamped inside the canvas.
    """
    if toe_count <= 0 or toe_h <= 0:
        return
    spacing = max(2, round(pad_w / toe_count))
    for i in range(toe_count):
        x0 = round(cx + (i - (toe_count - 1) / 2) * spacing)
        for dy in range(toe_h):
            y = foot_top + dy
            for dx in _toe_columns(style, dy):
                x = x0 + dx
                if 0 <= x < _SIZE and 0 <= y < _SIZE:
                    rpx[x, y] = (0, 0, 0, 255)
