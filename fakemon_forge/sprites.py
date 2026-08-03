import colorsys
import hashlib
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

_BASE_MODEL_ID = "Lykon/dreamshaper-8"
_LORA_PATH = Path(__file__).parent.parent / "models" / "loras" / "pksp768_V2-1.safetensors"
_LORA_SCALE = 0.7
_GEN_SIZE = 768
_NUM_STEPS = 30
_CFG_SCALE = 7
_SPRITE_SIZE = 768  # native SD render size — individual sprites keep full detail
_PALETTE_COLORS = 16
_KEY_COLOR = (200, 200, 168)  # Gen-3 transparency key colour (RGB).
# Tunable eyeball placeholder (see the module spec, cf. procedural_squash's
# ``amount_px`` / build_frame2's ``low``/``high``): a pixel within this
# Euclidean RGB distance of the detected border background counts as
# background — both for the border flood fill and the enclosed-pocket scan.
_KEY_TOLERANCE = 30
# Tunable eyeball placeholder: the border ring is treated as near-uniform (a
# flat backdrop, not a gradient/vignette) when at least this fraction of its
# pixels are within ``_KEY_TOLERANCE`` of the mean border colour.
_BORDER_UNIFORM_FRACTION = 0.9
# Tunable eyeball placeholders: colour + connectivity alone can't tell a real
# background pocket (a leg gap, a ring hole) from a same-coloured creature
# detail (a highlight, a belly patch) — both are just an 8-connected island of
# near-background pixels sitting inside the silhouette. What separates them is
# what walls them in: you see through a pocket, so it is rimmed by the
# creature's dark *outline*, whereas painted detail sits on mid-tone body
# colour. "Dark" is measured relative to the creature's own mean luma, not as
# an absolute cutoff, so a black-bodied Fakemon's highlights are not mistaken
# for pockets: a boundary pixel counts as outline when its luma falls below
# this fraction of that mean, and the component is treated as a pocket once
# that much of its boundary reaches ``_POCKET_OUTLINE_FRACTION``. Scaling the
# cutoff rather than subtracting a fixed margin is what makes it hold at both
# ends — on a near-black creature every fixed margin worth using on a bright
# one drops the cutoff below zero, and no pocket would ever key again.
_OUTLINE_LUMA_RATIO = 0.65
_POCKET_OUTLINE_FRACTION = 0.55
# Gen-3 palette contract: 3 reserved slots (key, black, white) plus at most this
# many creature colours, so the whole palette stays <= 16.
_MAX_CREATURE_COLORS = 13
# Tunable eyeball placeholder (kept smaller than the detection ``_KEY_TOLERANCE``
# above): a creature colour within this Euclidean RGB distance of ``_KEY_COLOR``
# is nudged away so it can never be mistaken for the transparency background.
_KEY_COLLISION_DISTANCE = 12
# Tunable eyeball placeholder: the middle 20% of a side-by-side front/back
# canvas's columns (as fractions of width) searched for the background gap
# between the two sprites.
_SPLIT_SEARCH_LOW = 0.4
_SPLIT_SEARCH_HIGH = 0.6

_TYPE_TAGS = {
    "Normal": "normaltype", "Fire": "firetype", "Water": "watertype",
    "Electric": "electrictype", "Grass": "grasstype", "Ice": "icetype",
    "Fighting": "fightingtype", "Poison": "poisontype", "Ground": "groundtype",
    "Flying": "flyingtype", "Psychic": "psychictype", "Bug": "bugtype",
    "Rock": "rocktype", "Ghost": "ghosttype", "Dragon": "dragontype",
    "Dark": "darktype", "Steel": "steeltype", "Fairy": "fairytype",
}

_GEN_STYLE = "gen3"


def build_prompt(sprite_prompt: str, types: list[str], extra_tags: list[str] | None = None) -> str:
    type_tags = [_TYPE_TAGS[t] for t in types if t in _TYPE_TAGS]
    all_tags = type_tags + [_GEN_STYLE] + (extra_tags or [])
    tags = " ".join(all_tags)
    return f"{tags} {sprite_prompt}".strip() if tags else sprite_prompt


def _encode_prompt(prompt: str, pipeline):
    from compel import Compel
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    return compel(prompt)


def k_centroid(image: Image.Image, width: int, height: int, centroids: int = 2) -> Image.Image:
    """Downscale ``image`` to ``width`` x ``height`` by per-tile dominant colour.

    For each output pixel, clusters the corresponding source tile into
    ``centroids`` colours (k-means quantize) and keeps the largest cluster's
    colour. Unlike ``NEAREST`` (picks one arbitrary source pixel per tile) or
    blended filters like ``LANCZOS`` (can synthesize colours absent from the
    tile via ringing at hard edges), this always emits a colour that actually
    occurs in the tile, aligned with the tile's dominant content — the
    property pixel-art downscaling needs. MIT-licensed algorithm ported from
    Astropulse's "pixeldetector". Does not mutate ``image``.
    """
    image = image.convert("RGB")
    out = Image.new("RGB", (width, height))
    wf = image.width / width
    hf = image.height / height
    for x in range(width):
        for y in range(height):
            tile = image.crop((int(x * wf), int(y * hf), int((x + 1) * wf), int((y + 1) * hf)))
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids)
            counts = tile.getcolors()
            idx = max(counts, key=lambda c: c[0])[1]
            pal = tile.getpalette()
            out.putpixel((x, y), tuple(pal[idx * 3:idx * 3 + 3]))
    return out


def postprocess(image: Image.Image, size: int | None = None) -> Image.Image:
    """Turn a raw SD RGB image into a ``P``-mode Gen-3-contract sprite.

    Delegates to ``_quantize_gen3`` — the single source of truth for the Gen-3
    palette contract (``_KEY_COLOR`` at index 0, reserved black/white, at most
    ``_MAX_CREATURE_COLORS`` creature colours, background pixels on index 0).
    ``size`` defaults to ``_SPRITE_SIZE`` (the native SD render size, so the
    default path performs no downscale at all). ``_quantize_gen3`` already
    performs the resize + colour/contrast enhance pre-steps internally (before
    flattening the background to the key), so they are not repeated here.
    """
    return _quantize_gen3(image, size)


def quantize_to_reference(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Quantize ``image`` against a fixed ``reference`` palette instead of an adaptive one.

    Unlike ``postprocess``, which builds a fresh 16-colour palette every call,
    this reuses ``reference``'s exact palette so a whole sprite set (front frames
    plus back sprite) can share one 16-colour palette. The pre-steps (resize +
    colour/contrast enhance, then flatten background to the key) match
    ``_quantize_gen3`` so either path yields the same input to quantization.
    Inputs are not mutated.

    Flattening the background to ``_KEY_COLOR`` before quantizing is what keeps a
    noisy near-white candidate background on **index 0** (the key) instead of
    nearest-mapping into the reserved white slot: after the flatten every
    background pixel is byte-exactly the reference's index-0 key entry.
    """
    if reference.mode != "P":
        raise ValueError(f"Expected palette-mode reference image, got {reference.mode}")
    # Enhance BEFORE flattening (mirroring ``_quantize_gen3``) so the enhance
    # can't shift the key off its byte-exact value. The reference defines the
    # target size, so locked views always match the view they lock to.
    image = image.resize(reference.size, Image.NEAREST)
    image = ImageEnhance.Color(image).enhance(1.1)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    if image.mode != "RGB":
        image = image.convert("RGB")  # _flatten_background_to_key assumes RGB
    image = _flatten_background_to_key(image)
    return image.quantize(palette=reference)


def _rgb_distance(a, b) -> float:
    """Euclidean distance between two RGB tuples."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _relative_luma(color) -> float:
    """Perceived brightness of an RGB colour (Rec. 601 luma), 0-255.

    Used to tell the creature's dark outline from its mid-tone body, which is
    a brightness judgement rather than a hue one — a red outline on a red body
    still reads as the edge.
    """
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def _creature_luma_mean(image: Image.Image, bg) -> float:
    """Mean ``_relative_luma`` of the non-background pixels of a keyed image.

    The reference brightness "outline" is judged against, so the judgement
    adapts to the creature instead of assuming a light one: on a black-bodied
    Fakemon an absolute dark-cutoff would call the whole body outline and key
    every highlight as a pocket. Returns ``0.0`` when there is no creature at
    all (an all-background image), which leaves nothing for it to gate.

    Area-weighted, so the body the creature is mostly made of is what sets the
    mean — "outline" means "darker than the body", so a one-pixel speck must
    not count for as much as the torso. Tallied via ``Counter`` over a flat
    ``get_flattened_data`` pass, which costs one C-level count plus a loop over
    the distinct colours instead of 590k ``px[x, y]`` lookups at 768px.
    """
    total = 0.0
    count = 0
    for p, n in Counter(image.get_flattened_data()).items():
        if p == _KEY_COLOR or _rgb_distance(p, bg) <= _KEY_TOLERANCE:
            continue
        total += _relative_luma(p) * n
        count += n
    return total / count if count else 0.0


def _is_background_pocket(touches_border, abuts_keyed, boundary, outline) -> bool:
    """Whether a near-background component is background, not creature detail.

    See ``_flatten_background_to_key``'s stage 2 for the three ways a component
    qualifies. ``boundary`` and ``outline`` count contacts with creature pixels
    and with outline-dark ones respectively, so a long run of outline weighs
    more than a single touching corner. A component with no boundary at all
    never touched a creature pixel, so there is no creature for it to be detail
    *of* — it is background.
    """
    if touches_border or abuts_keyed or not boundary:
        return True
    return outline / boundary >= _POCKET_OUTLINE_FRACTION


def _display_key(color) -> tuple[int, int, int]:
    """The colour as Gen 3 actually displays it: 5 bits per channel.

    Anything below that depth is invisible, so ``(4, 0, 0)`` and ``(0, 0, 0)``
    are one colour on screen even though they differ as 8-bit RGB.
    """
    return tuple(channel >> 3 for channel in color)


def _dedupe_by_display_depth(colors, reserved) -> list:
    """``colors`` minus any entry already shown by ``reserved`` or an earlier one.

    The palette holds 16 slots and the creature's share is small, so two slots
    that render as the same colour cost real detail for nothing. Compared at
    display depth rather than as 8-bit RGB, which is what the eye — and the
    hardware — actually sees.
    """
    seen = {_display_key(color) for color in reserved}
    kept = []
    for color in colors:
        key = _display_key(color)
        if key in seen:
            continue
        seen.add(key)
        kept.append(color)
    return kept


def _border_ring(image: Image.Image) -> list[tuple[int, int, int]]:
    """The pixels of the 1-px outer ring (top/bottom rows, left/right columns)."""
    w, h = image.size
    px = image.load()
    ring = [px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)]
    ring += [px[0, y] for y in range(h)] + [px[w - 1, y] for y in range(h)]
    return ring


def _flatten_background_to_key(image: Image.Image) -> Image.Image:
    """Return a new RGB image with the background flattened to ``_KEY_COLOR``.

    The SD background is near-white noise smeared across several tones, so this
    keys it to a single dedicated Gen-3 transparency colour ``(200, 200, 168)``
    while leaving the creature untouched. Two stages:

    1. **Border flood fill.** The background colour ``bg`` is detected as the
       per-channel mean of the 1-px border ring (robust to near-white noise,
       and not assuming pure white — SD sometimes paints tints/vignettes). The
       connected outer background is flood-filled from the corners with a
       ``_KEY_TOLERANCE`` threshold (an exact match won't do on noisy pixels).
    2. **Background-pocket scan.** The remaining pixels within
       ``_KEY_TOLERANCE`` of ``bg`` are grouped into 8-connected components, and
       each component is keyed only if it is background rather than creature
       detail. Three ways to qualify:

       * it touches the image border — outer background the flood walled itself
         out of (a leg gap opening onto the bottom edge);
       * it abuts pixels stage 1 already keyed — background the flood tried and
         failed to cross, because ``ImageDraw.floodfill`` thresholds on
         *Manhattan* distance and walks 4-connected while this scan uses
         Euclidean ``_rgb_distance`` and 8-connectivity;
       * it is rimmed by the creature's dark outline — you see *through* a real
         pocket (gaps between legs, the hole of a ring-shaped creature), so its
         boundary is silhouette edge, not body.

       Anything left is a same-coloured creature detail (a shield highlight, a
       white belly patch) that just happens to be near ``bg`` in colour, and is
       left untouched — keying it would punch a hole through the creature.

    Robustness fallback: if the border ring is *not* near-uniform (a gradient /
    vignette background), keying a single ``bg`` could eat the creature, so
    instead only the dominant border colour is keyed, a warning is emitted to
    stderr, and a best-effort result is returned — generation never fails on it.

    Assumes RGB input (does not convert or validate mode) and does not mutate
    the input; returns a fresh RGB image of the same size.
    """
    out = image.copy()
    w, h = out.size
    ring = _border_ring(out)
    n = len(ring)
    bg = tuple(round(sum(c[i] for c in ring) / n) for i in range(3))

    near_fraction = sum(1 for c in ring if _rgb_distance(c, bg) <= _KEY_TOLERANCE) / n
    px = out.load()

    if near_fraction < _BORDER_UNIFORM_FRACTION:
        # Gradient/vignette border: don't flood a single bg (it could eat the
        # creature). Key only the dominant border colour and warn — never raise.
        dominant = Counter(ring).most_common(1)[0][0]
        print(
            "warning: _flatten_background_to_key detected a non-uniform "
            "(gradient/vignette) border; keying only the dominant border colour",
            file=sys.stderr,
        )
        for y in range(h):
            for x in range(w):
                if _rgb_distance(px[x, y], dominant) <= _KEY_TOLERANCE:
                    px[x, y] = _KEY_COLOR
        return out

    # Stage 1: flood the connected outer background from the corners.
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if _rgb_distance(px[seed[0], seed[1]], bg) <= _KEY_TOLERANCE:
            ImageDraw.floodfill(out, seed, _KEY_COLOR, thresh=_KEY_TOLERANCE)

    # Stage 2: key the background pockets stage 1 could not reach, via
    # connected-component analysis.
    px = out.load()
    outline_max_luma = _creature_luma_mean(out, bg) * _OUTLINE_LUMA_RATIO
    visited = [[False] * w for _ in range(h)]
    for y0 in range(h):
        for x0 in range(w):
            if visited[y0][x0]:
                continue
            visited[y0][x0] = True
            p = px[x0, y0]
            if p == _KEY_COLOR or _rgb_distance(p, bg) > _KEY_TOLERANCE:
                continue

            # Flood this near-background component (8-connectivity), tallying
            # what walls it in: the image border, pixels stage 1 already keyed,
            # and how much of its contact with the creature is outline-dark.
            # Contacts are counted, not pixels, so a long run of outline weighs
            # more than a single touching corner.
            component = [(x0, y0)]
            touches_border = x0 in (0, w - 1) or y0 in (0, h - 1)
            abuts_keyed = False
            boundary = 0
            outline = 0
            stack = [(x0, y0)]
            while stack:
                x, y = stack.pop()
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if (nx, ny) == (x, y) or not (0 <= nx < w and 0 <= ny < h):
                            continue
                        q = px[nx, ny]
                        if q == _KEY_COLOR:
                            abuts_keyed = True
                            continue
                        if _rgb_distance(q, bg) > _KEY_TOLERANCE:
                            boundary += 1
                            if _relative_luma(q) <= outline_max_luma:
                                outline += 1
                            continue
                        if visited[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        component.append((nx, ny))
                        if nx in (0, w - 1) or ny in (0, h - 1):
                            touches_border = True
                        stack.append((nx, ny))

            if _is_background_pocket(touches_border, abuts_keyed, boundary, outline):
                for (x, y) in component:
                    px[x, y] = _KEY_COLOR
    return out


def _nudge_off_key(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Push a creature colour beyond ``_KEY_COLLISION_DISTANCE`` of ``_KEY_COLOR``.

    A creature colour within the collision radius of the transparency key could
    be confused with the flattened background, so it is moved along the
    key->colour direction just past the radius (a fixed fallback direction is
    used on the degenerate ``colour == key`` case). Channels are clamped to
    ``[0, 255]``. Colours already outside the radius are returned unchanged.
    """
    if _rgb_distance(color, _KEY_COLOR) > _KEY_COLLISION_DISTANCE:
        return color
    direction = [color[i] - _KEY_COLOR[i] for i in range(3)]
    norm = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
    if norm == 0:
        direction, norm = [0, 0, 1], 1.0  # colour == key: fixed fallback direction
    # +2 leaves margin so integer rounding can't pull the result back inside.
    target = _KEY_COLLISION_DISTANCE + 2
    return tuple(
        min(255, max(0, round(_KEY_COLOR[i] + direction[i] / norm * target)))
        for i in range(3)
    )


def _quantize_gen3(image: Image.Image, size: int | None = None) -> Image.Image:
    """Return a ``size`` x ``size`` ``P``-mode image obeying the Gen-3 palette contract.

    Palette layout: ``_KEY_COLOR`` at index 0, reserved black and white at
    indices 1 and 2 (always present, even if the creature uses neither), then up
    to ``_MAX_CREATURE_COLORS`` creature colours. Every background (key) pixel
    decodes to index 0 and no creature colour lands within
    ``_KEY_COLLISION_DISTANCE`` of the key.

    Deterministic pure PIL (median-cut + fixed arithmetic). Assumes RGB input
    (mirroring ``_flatten_background_to_key``) and does not mutate it.
    """
    # Resize + enhance for parity with ``postprocess``, done BEFORE flattening so
    # the enhance can't shift the key off its byte-exact value.
    target = size or _SPRITE_SIZE
    enhanced = image.resize((target, target), Image.NEAREST)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.1)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.1)

    flat = _flatten_background_to_key(enhanced)

    # Quantize the creature region alone (key pixels excluded) so the whole
    # 13-colour budget is spent on the creature, then nudge any colour off the key.
    creature_pixels = [p for p in flat.get_flattened_data() if p != _KEY_COLOR]
    creature_colors: list[tuple[int, int, int]] = []
    if creature_pixels:
        scratch = Image.new("RGB", (len(creature_pixels), 1))
        scratch.putdata(creature_pixels)
        quantized = scratch.quantize(colors=_MAX_CREATURE_COLORS)
        qpal = quantized.getpalette()
        used = sorted({idx for _, idx in quantized.getcolors(maxcolors=len(creature_pixels))})
        creature_colors = [_nudge_off_key(tuple(qpal[i * 3:i * 3 + 3])) for i in used]

    # Assemble the deterministic palette and map every pixel against it (dither
    # off). The key is index 0 with no duplicate, so background pixels resolve to
    # index 0; creature pixels near pure black/white legitimately snap to the
    # reserved slots without spending creature budget.
    reserved = [_KEY_COLOR, (0, 0, 0), (255, 255, 255)]
    creature_colors = _dedupe_by_display_depth(creature_colors, reserved)
    palette_colors = reserved + creature_colors
    flat_palette = [channel for color in palette_colors for channel in color]
    reference = Image.new("P", (1, 1))
    reference.putpalette(flat_palette)
    return flat.quantize(palette=reference, dither=Image.Dither.NONE)


def _is_achromatic(r: int, g: int, b: int) -> bool:
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return lum < 40 or lum > 215


def generate_shiny(sprite_path: str, name: str, output_path: str) -> None:
    """Derive a shiny palette from an existing sprite by hue-rotating mid-tone colors.

    The Gen-3 contract's reserved entries are pinned unconditionally, never left
    to the achromatic threshold: the transparency key at **index 0** (the key
    ``(200, 200, 168)`` is chromatic — lum ~196 — so the threshold would rotate
    it), plus any ``(255, 255, 255)`` (white) or ``(0, 0, 0)`` (black) entry.
    All other (creature) entries are hue-rotated by a shift seeded from the
    Pokémon's name so each one is unique. ``_is_achromatic`` still preserves any
    remaining very-bright/very-dark creature entries.
    """
    img = Image.open(sprite_path)
    if img.mode != "P":
        raise ValueError(f"Expected palette-mode image, got {img.mode}")

    hue_shift = (int(hashlib.md5(name.encode()).hexdigest(), 16) % 300 + 30) / 360.0

    flat = img.getpalette()  # [R, G, B, R, G, B, ...] × 256
    new_palette = []
    for i in range(0, len(flat), 3):
        r, g, b = flat[i], flat[i + 1], flat[i + 2]
        # Pin key (index 0), white, and black unconditionally — by explicit
        # match, independent of _is_achromatic — so the chromatic key is
        # provably never rotated.
        if i == 0 or (r, g, b) == (255, 255, 255) or (r, g, b) == (0, 0, 0):
            new_palette.extend([r, g, b])
        elif _is_achromatic(r, g, b):
            new_palette.extend([r, g, b])
        else:
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h = (h + hue_shift) % 1.0
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            new_palette.extend([round(nr * 255), round(ng * 255), round(nb * 255)])

    shiny = img.copy()
    shiny.putpalette(new_palette)
    shiny.save(output_path)


def _background_index(frame1: Image.Image) -> int:
    """Return frame 1's background palette index.

    Default choice: the *most common* palette index (the flat backdrop behind
    the creature dominates a sprite). Used both to fill the squash/recenter
    canvas and to define "non-background" for bbox detection.
    """
    # getcolors() yields (count, index) pairs for a P-mode image.
    count_index = max(frame1.getcolors(maxcolors=frame1.size[0] * frame1.size[1]))
    return count_index[1]


def _content_bbox(image: Image.Image, background: int):
    """Bounding box of the non-background region, or ``None`` if all background."""
    mask = image.point(lambda p: 255 if p != background else 0)
    return mask.getbbox()


def split_front_back_canvas(canvas: Image.Image) -> tuple[Image.Image, Image.Image] | None:
    """Cut a side-by-side front/back canvas into a front half and a back half.

    ``canvas`` is assumed to hold a front sprite on the left and a back sprite
    on the right, sharing one flat background (mirroring
    ``_flatten_background_to_key``'s RGB assumption — not converted or
    validated here). The background colour ``bg`` is the per-channel mean of
    ``_border_ring(canvas)``, exactly as ``_flatten_background_to_key``
    computes it.

    Only the middle ``_SPLIT_SEARCH_LOW``-``_SPLIT_SEARCH_HIGH`` fraction of
    columns is searched for the widest run of columns that are background for
    their *full height* (every pixel within ``_KEY_TOLERANCE`` of ``bg``) —
    that's where the gap between the two sprites is expected to fall. The cut
    lands at the widest run's centre column; ties go to the first (leftmost)
    run encountered. Returns ``(front_half, back_half)`` crops of ``canvas``,
    or ``None`` if no full-height background run exists in that window (e.g.
    the two sprites' silhouettes span the whole search window). ``canvas`` is
    not mutated.
    """
    w, h = canvas.size
    ring = _border_ring(canvas)
    bg = tuple(round(sum(c[i] for c in ring) / len(ring)) for i in range(3))

    px = canvas.load()
    x_start = int(_SPLIT_SEARCH_LOW * w)
    x_end = int(_SPLIT_SEARCH_HIGH * w)

    best_run = None
    run_start = None
    for x in range(x_start, x_end):
        is_full_height_bg = all(_rgb_distance(px[x, y], bg) <= _KEY_TOLERANCE for y in range(h))
        if is_full_height_bg:
            if run_start is None:
                run_start = x
            continue
        if run_start is not None:
            if best_run is None or (x - run_start) > (best_run[1] - best_run[0]):
                best_run = (run_start, x)
            run_start = None
    if run_start is not None and (best_run is None or (x_end - run_start) > (best_run[1] - best_run[0])):
        best_run = (run_start, x_end)

    if best_run is None:
        return None

    cut = (best_run[0] + best_run[1]) // 2
    return canvas.crop((0, 0, cut, h)), canvas.crop((cut, 0, w, h))


def procedural_squash(frame1: Image.Image, amount_px: int | None = None) -> Image.Image:
    """Bottom-anchored vertical squash of ``frame1`` — the Gen-3 breathing frame.

    Compresses frame 1's content by ``amount_px`` rows and pastes it at
    the bottom, so the creature's feet stay planted while the top compresses.
    This is the fallback frame 2 whenever an img2img candidate is rejected, so
    it is built to be genuinely good and to land inside the acceptance band.

    ``amount_px`` defaults to ``height // 48`` (2px on a 96px sprite, 16px at
    the native 768) so the squash reads the same at any sprite size; it is a
    tunable eyeball placeholder (see the module spec). Operates entirely in
    P-space (nearest-neighbour reuses existing indices) so the result shares
    frame 1's exact palette without a re-quantize. Input is not mutated.
    """
    if frame1.mode != "P":
        raise ValueError(f"Expected palette-mode frame1 image, got {frame1.mode}")
    w, h = frame1.size
    if amount_px is None:
        amount_px = max(1, h // 48)
    background = _background_index(frame1)
    # NEAREST in P-space introduces no new colours (indices are reused verbatim).
    squashed = frame1.resize((w, h - amount_px), Image.NEAREST)
    canvas = Image.new("P", (w, h), background)
    canvas.putpalette(frame1.getpalette())
    # Paste at (0, amount_px): bottom row aligns with the canvas bottom and the
    # top ``amount_px`` rows become background.
    canvas.paste(squashed, (0, amount_px))
    return canvas


def recenter_to_anchor(candidate: Image.Image, frame1: Image.Image) -> Image.Image:
    """Translate ``candidate`` so its content bbox anchors to frame 1's.

    The anchor is the bottom-centre of the non-background bounding box, so the
    two frames share a planted-feet position and the second frame does not
    appear to teleport/jitter. ``candidate`` is expected to already share
    frame 1's palette (in the ``build_frame2`` flow it has been palette-locked).
    Returns a fresh P-mode image of frame 1's size sharing its exact palette;
    inputs are not mutated.
    """
    if frame1.mode != "P":
        raise ValueError(f"Expected palette-mode frame1 image, got {frame1.mode}")
    background = _background_index(frame1)
    canvas = Image.new("P", frame1.size, background)
    canvas.putpalette(frame1.getpalette())

    frame1_bbox = _content_bbox(frame1, background)
    candidate_bbox = _content_bbox(candidate, background)
    if frame1_bbox is None or candidate_bbox is None:
        # No bbox on one side -> no translation possible; return the candidate
        # re-locked to frame 1's palette, untranslated.
        out = candidate.copy()
        out.putpalette(frame1.getpalette())
        return out

    # bbox is (left, top, right, bottom) with right/bottom exclusive.
    anchor_x = (frame1_bbox[0] + frame1_bbox[2]) / 2
    anchor_y = frame1_bbox[3]
    content = candidate.crop(candidate_bbox)
    content_w = candidate_bbox[2] - candidate_bbox[0]
    content_h = candidate_bbox[3] - candidate_bbox[1]
    paste_left = round(anchor_x - content_w / 2)
    paste_top = round(anchor_y - content_h)
    canvas.paste(content, (paste_left, paste_top))
    return canvas


def difference_ratio(a: Image.Image, b: Image.Image) -> float:
    """Fraction (0.0-1.0) of pixels that differ between two same-size images.

    For same-palette P-mode inputs this compares palette indices directly, so
    ``difference_ratio(x, x) == 0.0`` and images differing everywhere give 1.0.
    """
    if a.size != b.size:
        raise ValueError(f"Cannot compare different-size images: {a.size} vs {b.size}")
    data_a = a.get_flattened_data()
    data_b = b.get_flattened_data()
    differing = sum(1 for x, y in zip(data_a, data_b) if x != y)
    return differing / len(data_a)


def build_frame2(
    frame1: Image.Image,
    candidate: Image.Image | None = None,
    low: float = 0.02,
    high: float = 0.30,
) -> Image.Image:
    """Turn frame 1 plus an optional candidate into a guaranteed-valid frame 2.

    Accepts the candidate only when its palette-locked, recentred difference
    from frame 1 lands inside ``[low, high]`` — below ``low`` is texture
    shimmer (not motion), above ``high`` is identity drift / teleporting.
    Otherwise (or with no candidate) falls back to ``procedural_squash``.
    Always returns a valid P-mode frame of frame 1's size sharing its palette.

    ``low`` / ``high`` are tunable eyeball placeholders (see the module spec);
    a later ML slice or a human is expected to tune them.
    """
    if frame1.mode != "P":
        raise ValueError(f"Expected palette-mode frame1 image, got {frame1.mode}")
    if candidate is None:
        return procedural_squash(frame1)
    locked = quantize_to_reference(candidate, frame1)  # also resizes to frame1's size
    recentred = recenter_to_anchor(locked, frame1)
    ratio = difference_ratio(recentred, frame1)
    if low <= ratio <= high:
        return recentred
    return procedural_squash(frame1)


def _make_generator(seed: int | None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g


def generate_sprite(
    prompt: str, types: list[str], output_path: str, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None,
) -> None:
    conditioning = _encode_prompt(build_prompt(prompt, types, extra_tags), pipeline)
    result = pipeline(
        prompt_embeds=conditioning,
        width=_GEN_SIZE,
        height=_GEN_SIZE,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
    )
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def generate_back_sprite(
    prompt: str, types: list[str], output_path: str, *, pipeline, seed: int | None = None
) -> None:
    generate_sprite(prompt, types, output_path, pipeline=pipeline, extra_tags=["backside"], seed=seed)



def _run_img2img(
    prompt: str, types: list[str], image_path: str, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None, strength: float = 0.8,
) -> Image.Image:
    """Run the img2img pipeline and return the raw RGB pipeline image.

    Shared by ``generate_sprite_img2img`` (which then adaptively quantizes +
    saves) and ``generate_frame2`` (which hands the raw candidate to
    ``build_frame2`` so it isn't double-quantized off frame 1's palette).
    """
    init = Image.open(image_path).convert("RGB").resize((_GEN_SIZE, _GEN_SIZE), Image.LANCZOS)
    conditioning = _encode_prompt(build_prompt(prompt, types, extra_tags), pipeline)
    result = pipeline(
        prompt_embeds=conditioning,
        image=init,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
        strength=strength,
    )
    return result.images[0]


def generate_sprite_img2img(
    prompt: str, types: list[str], image_path: str, output_path: str, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None, strength: float = 0.8,
    reference_path: str | None = None,
) -> None:
    """Generate a sprite via img2img and save it as a P-mode PNG.

    When ``reference_path`` is given, the raw candidate is locked to that
    reference image's exact 16-colour palette via ``quantize_to_reference`` — so
    the back sprite can share the front frames' palette instead of building its
    own adaptive one. When ``reference_path`` is ``None`` (every other caller),
    the candidate is quantized adaptively via ``postprocess`` exactly as before.
    """
    candidate = _run_img2img(
        prompt, types, image_path, pipeline=pipeline,
        extra_tags=extra_tags, seed=seed, strength=strength,
    )
    if reference_path is None:
        sprite = postprocess(candidate)
    else:
        reference = Image.open(reference_path)
        sprite = quantize_to_reference(candidate, reference)
    sprite.save(output_path)


def generate_frame2(
    prompt: str, types: list[str], front_sprite_path: str, output_path: str, *, pipeline,
    seed: int | None = None, strength: float = 0.35, extra_tags: list[str] | None = None,
) -> None:
    """Generate the second front-animation frame and save it to ``output_path``.

    Runs img2img from the finished front sprite at low ``strength`` with an
    animation tag (defaults to ``["open mouth"]``) using frame 1's seed, then
    hands the raw RGB candidate to ``build_frame2`` — which palette-locks +
    recenters it, accepts it iff its difference from frame 1 is in-band, and
    otherwise falls back to a procedural squash. The result always shares
    frame 1's exact 16-colour palette.
    """
    candidate = _run_img2img(
        prompt, types, front_sprite_path, pipeline=pipeline,
        extra_tags=extra_tags or ["open mouth"], seed=seed, strength=strength,
    )
    frame1 = Image.open(front_sprite_path)
    frame2 = build_frame2(frame1, candidate)
    frame2.save(output_path)


# Cell layout of the 4x2 stitched sheet, matching the hand-made reference
# sheets kept outside the repo (Blitin, Bluchis, and one official sheet):
# row 0 is normal/shiny column pairs of front then back, row 1 is frame 2.
# The remaining two cells stay on the transparency key.
_SHEET_LAYOUT = [
    ("sprite.png", 0, 0),
    ("sprite_shiny.png", 1, 0),
    ("sprite_back.png", 2, 0),
    ("sprite_back_shiny.png", 3, 0),
    ("sprite_frame2.png", 0, 1),
    ("sprite_frame2_shiny.png", 1, 1),
]


def stitch_spritesheet(stage_dir: str, output_path: str, *, cell_size: int = 64) -> None:
    """Stitch a stage's six sprite views into one 4x2 sheet of GBA-sized cells.

    The canvas (including the two unused cells) is filled with ``_KEY_COLOR``
    so a downstream tool can key transparency off the whole sheet. A missing view
    leaves its cell on the key rather than failing — mirroring how sprite
    generation degrades per-view. The individual views are kept at the native
    SD render size (``_SPRITE_SIZE``), so the default 64px cell is a **single**
    ``k_centroid`` downscale (768/64 = an exact /12) — one resample from full
    detail, never a chain — which picks each cell's dominant tile colour
    without blending new colours into the palette.
    """
    sheet = Image.new("RGB", (4 * cell_size, 2 * cell_size), _KEY_COLOR)
    for name, col, row in _SHEET_LAYOUT:
        path = Path(stage_dir) / name
        if not path.exists():
            continue
        cell = Image.open(path).convert("RGB")
        if cell.size != (cell_size, cell_size):
            cell = k_centroid(cell, cell_size, cell_size)
        sheet.paste(cell, (col * cell_size, row * cell_size))
    sheet.save(output_path)


def make_img2img_pipeline(txt2img_pipe):
    from diffusers import StableDiffusionImg2ImgPipeline
    return StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)


def _device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _apply_lora(pipe) -> None:
    from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin

    path = str(_LORA_PATH)
    state_dict, network_alphas, metadata = StableDiffusionLoraLoaderMixin.lora_state_dict(
        path, return_lora_metadata=True
    )

    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet,
        metadata=metadata,
        _pipeline=pipe,
    )

    # diffusers converts kohya TE keys to "text_encoder.text_model.encoder.*" but
    # the actual text encoder modules are named "encoder.*" (no text_model. wrapper),
    # so rank detection fails.  Strip the extra level before handing off.
    def _drop_text_model(d):
        if not d:
            return d
        old = "text_encoder.text_model."
        new = "text_encoder."
        return {new + k[len(old):] if k.startswith(old) else k: v for k, v in d.items()}

    pipe.load_lora_into_text_encoder(
        _drop_text_model(state_dict),
        network_alphas=_drop_text_model(network_alphas),
        text_encoder=pipe.text_encoder,
        lora_scale=pipe.lora_scale,
        metadata=metadata,
        _pipeline=pipe,
    )
    pipe.fuse_lora(lora_scale=_LORA_SCALE)


def _set_dpmpp_karras(pipe) -> None:
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )


def _load_base_pipeline(pipe_cls):
    device, dtype = _device_and_dtype()
    pipe = pipe_cls.from_pretrained(_BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None)
    _apply_lora(pipe)
    _set_dpmpp_karras(pipe)
    return pipe.to(device)


def load_txt2img_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        return _load_base_pipeline(StableDiffusionPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)


def load_img2img_pipeline():
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        return _load_base_pipeline(StableDiffusionImg2ImgPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)
