"""Pure-Pillow post-processing that builds the Gen-3 party-menu icon.

Turns an already-generated front sprite (a ``P``-mode Gen-3-contract sprite from
``sprites.py``) into ``sprite_small.png``: a **32x64** PNG laid out as two
vertically stacked 32x32 animation frames, with an **opaque** teal-green
background and a small (<= 16 colour) palette. There is no Stable Diffusion /
torch / diffusers involvement — the whole module is deterministic pure Pillow,
so it (and its tests) run in the slim keep sandbox.

Reference contract (measured from the official Venusaur icon, kept out of the
repo): 32x64, all pixels opaque, <= 16 distinct colours, teal-green background
``(96, 152, 128)`` dominating, and frame 2 == frame 1 shifted down 1px.
"""

from PIL import Image

from fakemon_forge.sprites import _KEY_COLOR, _KEY_TOLERANCE, _rgb_distance

# Party-menu icon background — an *opaque* teal-green that dominates the image
# and is forced onto palette index 0 (unlike the sprites' transparency key,
# which downstream tools alpha-key against, the icon background is a normal
# opaque colour). Value measured from the reference Venusaur icon.
_ICON_BG_COLOR = (96, 152, 128)
# Icon frame size — two of these stack vertically into the 32x64 output.
_ICON_SIZE = 32
# Up to this many quantized creature colours plus the teal background = <= 16.
# The sprite quantizer reserves black/white slots too; the icon reserves only
# teal, so it can spend two extra slots on the creature.
_MAX_ICON_CREATURE_COLORS = 15

# NOTE: this is a *per-mon* palette (up to 15 colours quantized from this render
# plus teal at index 0). Authentic Gen-3 instead shares 3 fixed palettes across
# every species and downstream tools remap on import — so do NOT "fix" this
# into shared fixed palettes; per-mon adaptive palettes are the intended output.


def generate_icon(source_path: str, output_path: str) -> None:
    """Build the 32x64 party-menu icon from a front sprite and save it as PNG.

    Opens ``source_path`` (must be a ``P``-mode sprite whose index 0 is the
    transparency key ``_KEY_COLOR``), downscales it once to a 32x32 opaque frame
    with teal at palette index 0, derives frame 2 as a 1px down-shift, stitches
    the two into a 32x64 image and writes it to ``output_path``.

    Raises ``ValueError`` if the source is not ``P``-mode (the caller is expected
    to wrap this in warn-and-continue, per the pipeline's per-view degradation).
    Returns ``None``.
    """
    source = Image.open(source_path)
    if source.mode != "P":
        raise ValueError(f"Expected palette-mode source image, got {source.mode}")

    frame1 = _build_frame1(source)
    frame2 = _shift_down_one(frame1)
    _stitch_and_save(frame1, frame2, output_path)


def _background_mask(source: Image.Image, rgb32: Image.Image) -> Image.Image:
    """Return a 32x32 ``L`` mask (255 = background) for the downscaled icon.

    A pixel is background when either the source's keyed region (palette index 0)
    downscales to a majority-background cell, or the downscaled RGB pixel still
    reads within ``_KEY_TOLERANCE`` of ``_KEY_COLOR`` after resampling. Assumes
    ``source`` is ``P``-mode; does not mutate its inputs.
    """
    # Native key mask: 255 where the source decodes to index 0 (the key), else 0.
    key_native = Image.new("L", source.size, 0)
    key_native.putdata([255 if idx == 0 else 0 for idx in source.get_flattened_data()])
    # LANCZOS-downscale the mask alongside the sprite; a cell is background when
    # the majority of its area was keyed (>= 128 after the resample).
    key_small = key_native.resize((_ICON_SIZE, _ICON_SIZE), Image.LANCZOS)

    key_data = list(key_small.get_flattened_data())
    rgb_data = list(rgb32.get_flattened_data())
    flags = [
        255
        if key_data[i] >= 128 or _rgb_distance(rgb_data[i], _KEY_COLOR) <= _KEY_TOLERANCE
        else 0
        for i in range(len(rgb_data))
    ]
    mask = Image.new("L", (_ICON_SIZE, _ICON_SIZE))
    mask.putdata(flags)
    return mask


def _creature_palette(rgb32: Image.Image, bg_mask: Image.Image) -> list[tuple[int, int, int]]:
    """Adaptively quantize the non-background region to up to 15 colours.

    Collects the creature pixels (where ``bg_mask`` is 0) and runs PIL adaptive
    (median-cut) quantization with dither off — deterministic, mirroring
    ``sprites.py``. Returns the up-to-15 creature RGB colours (fewer is fine; an
    empty creature region returns ``[]``). Inputs are not mutated.
    """
    rgb_data = list(rgb32.get_flattened_data())
    mask_data = list(bg_mask.get_flattened_data())
    creature_pixels = [rgb_data[i] for i in range(len(rgb_data)) if mask_data[i] == 0]
    if not creature_pixels:
        return []
    scratch = Image.new("RGB", (len(creature_pixels), 1))
    scratch.putdata(creature_pixels)
    quantized = scratch.quantize(colors=_MAX_ICON_CREATURE_COLORS, dither=Image.Dither.NONE)
    qpal = quantized.getpalette()
    used = sorted({idx for _, idx in quantized.getcolors(maxcolors=len(creature_pixels))})
    return [tuple(qpal[i * 3:i * 3 + 3]) for i in used]


def _build_frame1(source: Image.Image) -> Image.Image:
    """Return a 32x32 opaque ``P``-mode frame with teal forced onto index 0.

    Single high-quality downscale (RGB, one LANCZOS 768 -> 32 so a 1px outline is
    never chained through multiple resamples), then quantize the creature region
    to up to 15 colours against a palette whose index 0 is the teal background.
    Every background pixel is forced to index 0, so the result is opaque with no
    alpha holes. Assumes ``source`` is ``P``-mode; does not mutate it.
    """
    rgb32 = source.convert("RGB").resize((_ICON_SIZE, _ICON_SIZE), Image.LANCZOS)
    bg_mask = _background_mask(source, rgb32)
    creature_colors = _creature_palette(rgb32, bg_mask)

    # Teal at index 0, then the creature colours. Pad every remaining slot with
    # teal so a stray nearest-colour mapping can never introduce a new colour
    # (keeps the distinct-colour count trivially <= 16).
    palette_colors = [_ICON_BG_COLOR] + creature_colors
    palette_colors += [_ICON_BG_COLOR] * (256 - len(palette_colors))
    flat_palette = [channel for color in palette_colors for channel in color]
    reference = Image.new("P", (1, 1))
    reference.putpalette(flat_palette)

    frame1 = rgb32.quantize(palette=reference, dither=Image.Dither.NONE)
    # Force every background pixel onto index 0 (teal) regardless of which
    # creature colour happened to be nearest — no transparency, no alpha holes.
    frame1.paste(0, mask=bg_mask)
    return frame1


def _shift_down_one(frame1: Image.Image) -> Image.Image:
    """Return frame 2: ``frame1`` shifted down 1px, top row backfilled teal.

    Row 0 becomes the teal background (index 0), rows 1..31 equal frame 1's rows
    0..30, and frame 1's bottom row is cropped away. Pure copy/paste in frame 1's
    palette space — introduces no new colours. Reference-verified as faithful
    (real Gen-3 frame 2 differs by only ~64/1024 edge pixels).
    """
    frame2 = Image.new("P", frame1.size, 0)
    frame2.putpalette(frame1.getpalette())
    w, h = frame1.size
    frame2.paste(frame1.crop((0, 0, w, h - 1)), (0, 1))
    return frame2


def _stitch_and_save(frame1: Image.Image, frame2: Image.Image, output_path: str) -> None:
    """Stack frame 1 (top) over frame 2 (bottom) into a 32x64 PNG and save it.

    The output shares frame 1's palette (teal at index 0) and carries no
    transparency chunk, so it is fully opaque.
    """
    w, h = frame1.size
    icon = Image.new("P", (w, 2 * h), 0)
    icon.putpalette(frame1.getpalette())
    icon.paste(frame1, (0, 0))
    icon.paste(frame2, (0, h))
    icon.save(output_path, format="PNG")
