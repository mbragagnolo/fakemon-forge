import colorsys
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

_BASE_MODEL_ID = "Laxhar/noobai-XL-1.1"
# Manual download required (never committed; models/ is gitignored): Civitai
# model 378602, "Pokemon Sprite XL PixelArt back&front" (login required).
_LORA_PATH = Path(__file__).parent.parent / "models" / "loras" / "pkspbf_nb_v1.safetensors"
# Fuse the sprite LoRA at full strength, and keep CFG low. Both match the
# research spike that produced the output this backend was adopted for
# (prototype/gen_noobai.py at fb4dd19: fuse_lora(lora_scale=1.0),
# guidance_scale=5.5). Shipping at 0.7 / 7 instead was a silent quality
# regression: the LoRA is the only thing making the render read as a Gen-3
# sprite rather than generic anime art, so weakening it to 70% costs exactly
# the property it was adopted for, and NoobAI/Illustrious-family bases are
# tuned for low CFG — 7 oversaturates and hardens edges the quantiser then
# bakes into the 16-colour palette.
_LORA_SCALE = 1.0
_GEN_SIZE = 768
_NUM_STEPS = 28
_CFG_SCALE = 5.5
# CLIP's context window. Anything past it is silently dropped by the text
# encoder — no error, no truncation marker in the output, just a prompt that
# quietly stopped saying "white background".
_CLIP_TOKEN_LIMIT = 77
# Conservative CLIP tokens-per-word for this prompt vocabulary. Measured with
# CLIPTokenizer("openai/clip-vit-large-patch14") over the spike prompts and
# observed live prompts: 1.48-1.71, the high end being short tag-heavy strings
# where the two special tokens weigh most. Rounded up, so the estimate errs
# toward trimming a tag that would have fitted rather than losing the suffix.
_TOKENS_PER_WORD = 1.8
# Composition anchors prepended to every prompt. The sprite_prompt spec asks
# the LLM for shape/colour/feature tags, and nothing in that vocabulary ever
# states the *framing* — so the model was free to read a size word as a
# instruction to fill the canvas. Observed live: a stage-2 prompt leading
# "large ceramic mug" and ending "imposing stance" rendered a full-bleed
# photographic close-up with no background left, which then left no uniform
# border for the pair split to find. Stage 1 of the same line survived only
# because "tiny" and "cute expression" implied the framing by accident.
# Stating it outright removes the accident. These sit at the front, right
# after the LoRA trigger, where they carry weight and cannot be truncated.
_FRAMING_TAGS = "single creature, full body, centered"
# The sprite_prompt spec forbids these, but the LLM does not reliably obey: one
# call after the ban was added it still opened stage 2 with "large ceramic
# cauldron" (stage 1 of the same call complied). They are stripped here too, so
# the guarantee does not depend on a model following an instruction. Scale
# belongs in height_dm/weight_hg; in a prompt it only tells the renderer how
# much of the frame to fill.
_FRAMING_WORDS = (
    "large", "huge", "giant", "gigantic", "enormous", "colossal", "massive",
    "towering", "imposing", "close-up", "closeup", "dramatic", "epic",
)
_FRAMING_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _FRAMING_WORDS) + r")\b", re.IGNORECASE
)
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
# Front+back pair canvas width (literal per the issue): the fused LoRA renders
# front on the left half, back on the right half, at twice the single-sprite
# width. Height stays ``_GEN_SIZE``.
_PAIR_WIDTH = 1536

_NEGATIVE_PROMPT = "worst quality, low quality, blurry, watermark, signature, text, jpeg artifacts"


# The SD1.5 LoRA this backend replaced had a trained "firetype"/"watertype"
# trigger vocabulary, so type conditioning used to be mechanical: look the type
# up in a table, prepend the tag. The SDXL LoRA knows no such vocabulary, so the
# type signal has to arrive as ordinary description ("wreathed in embers"), and
# only the LLM that picked the types can write it. That obligation is spelled
# out in the ``sprite_prompt`` spec in ``generator.py`` (and pinned by a test
# there) — it is the sole reason the ``types`` argument threaded through the
# generate_* functions below is accepted but never read. Anything mechanical
# here would just fight the prompt the LLM already wrote.
def _estimate_clip_tokens(text: str) -> int:
    """Conservative CLIP token count for ``text``, without a real tokenizer.

    ``build_prompt`` is torch-free and unit-tested with no ML stack installed,
    so it cannot call ``CLIPTokenizer``. Words times ``_TOKENS_PER_WORD``, plus
    the two special tokens CLIP wraps every sequence in.
    """
    return round(len(text.split()) * _TOKENS_PER_WORD) + 2


def _strip_framing_words(sprite_prompt: str) -> str:
    """Remove ``_FRAMING_WORDS`` from ``sprite_prompt``, tag by tag.

    Only the offending word goes; the tag it sat in survives, so "large
    ceramic cauldron" becomes "ceramic cauldron" rather than being dropped
    outright — the creature is still described, it just no longer instructs the
    renderer to fill the frame. A tag consisting of nothing else is dropped.
    """
    cleaned, changed = [], False
    for tag in (t.strip() for t in sprite_prompt.split(",")):
        if not tag:
            continue
        stripped = re.sub(r"\s{2,}", " ", _FRAMING_WORD_RE.sub("", tag)).strip()
        if stripped != tag:
            changed = True
        if stripped:
            cleaned.append(stripped)
    if changed:
        print(
            "warning: build_prompt stripped framing/scale words from sprite_prompt "
            "(they tell the renderer how much of the frame to fill, not how big "
            "the creature is)",
            file=sys.stderr,
        )
    return ", ".join(cleaned)


def _trim_tags_to_fit(sprite_prompt: str, prefix: str, suffix: str) -> str:
    """Drop trailing ``sprite_prompt`` tags until the whole prompt fits CLIP.

    The style anchors live in ``suffix`` (``white background``, plus any
    ``extra_tags``), so they are what an overflow would silently delete —
    the creature description would survive intact and the *instructions* would
    vanish. Trimming the description instead inverts that: the prompt loses its
    least-important trailing detail and keeps every styling tag.

    Ordering is deliberately left alone. Moving the anchors to the front would
    also solve the overflow, but ``gen3, <description>, white background`` is
    the exact shape the research spike validated, and this is a safety net for
    a contract the LLM is already asked to satisfy — not a licence to redesign
    a prompt that works. At least one tag is always kept.
    """
    tags = [t.strip() for t in sprite_prompt.split(",") if t.strip()]
    kept: list[str] = []
    for tag in tags:
        candidate = ", ".join(kept + [tag])
        if kept and _estimate_clip_tokens(prefix + candidate + suffix) > _CLIP_TOKEN_LIMIT:
            break
        kept.append(tag)
    if len(kept) < len(tags):
        print(
            f"warning: build_prompt dropped {len(tags) - len(kept)} of {len(tags)} "
            f"sprite_prompt tags to keep the style tags inside CLIP's "
            f"{_CLIP_TOKEN_LIMIT}-token window",
            file=sys.stderr,
        )
    return ", ".join(kept)


def build_prompt(sprite_prompt: str, extra_tags: list[str] | None = None) -> str:
    """Plain SDXL prompt string: type wording is baked into ``sprite_prompt`` upstream.

    Trims ``sprite_prompt`` if the assembled prompt would overrun CLIP's
    context window, so the trailing style tags always survive — see
    ``_trim_tags_to_fit``. The ``sprite_prompt`` spec in ``generator.py``
    already caps length, so this should not normally fire; it exists because
    the overflow it guards is silent, and a model that ignores the cap would
    otherwise degrade sprite quality with nothing in the logs to say why.
    """
    prefix = f"gen3, {_FRAMING_TAGS}, "
    if extra_tags:
        suffix = f", {', '.join(extra_tags)}, white background"
    else:
        suffix = ", white background"
    sprite_prompt = _strip_framing_words(sprite_prompt)
    return prefix + _trim_tags_to_fit(sprite_prompt, prefix, suffix) + suffix


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

    Only downscales meaningfully: if either target dimension exceeds the
    source's, the per-output-pixel tile would be empty and there is no
    dominant colour to pick, so the whole resize falls back to ``NEAREST``
    (which likewise only replicates existing colours).
    """
    image = image.convert("RGB")
    wf = image.width / width
    hf = image.height / height
    if wf < 1 or hf < 1:
        return image.resize((width, height), Image.NEAREST)
    out = Image.new("RGB", (width, height))
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


def quantize_to_reference(
    image: Image.Image, reference: Image.Image, *, enhance: bool = True
) -> Image.Image:
    """Quantize ``image`` against a fixed ``reference`` palette instead of an adaptive one.

    Unlike ``postprocess``, which builds a fresh 16-colour palette every call,
    this reuses ``reference``'s exact palette so a whole sprite set (front frames
    plus back sprite) can share one 16-colour palette. The pre-steps (resize +
    colour/contrast enhance, then flatten background to the key) match
    ``_quantize_gen3`` so either path yields the same input to quantization.
    Inputs are not mutated.

    ``enhance=False`` skips the colour/contrast enhance for candidates whose
    colours already derive from the enhanced reference — the frame-2 img2img
    candidate's init is built *from* frame 1's palette, so enhancing again
    shifts whole regions onto neighbouring slots and manufactures colour churn
    (#90). A raw render with no such lineage (the back sprite) keeps the
    default, for parity with the single enhance its front view got.

    Flattening the background to ``_KEY_COLOR`` before quantizing is what puts a
    noisy near-white candidate background on **index 0** (the key): the flatten
    makes every background pixel byte-exactly the key, and ``_map_flat_to_palette``
    assigns index 0 from exactly those pixels — the key never competes in the
    nearest-colour mapping, so a creature midtone near the key cannot be
    swallowed as transparency (see the mapping helper for why that matters).
    """
    if reference.mode != "P":
        raise ValueError(f"Expected palette-mode reference image, got {reference.mode}")
    # Enhance BEFORE flattening (mirroring ``_quantize_gen3``) so the enhance
    # can't shift the key off its byte-exact value. The reference defines the
    # target size, so locked views always match the view they lock to.
    image = image.resize(reference.size, Image.NEAREST)
    if enhance:
        image = ImageEnhance.Color(image).enhance(1.1)
        image = ImageEnhance.Contrast(image).enhance(1.1)
    if image.mode != "RGB":
        image = image.convert("RGB")  # _flatten_background_to_key assumes RGB
    image = _flatten_background_to_key(image)
    return _map_flat_to_palette(image, reference.getpalette())


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


def _detect_background(ring) -> tuple[int, int, int]:
    """The background colour of a ``_border_ring``: its per-channel mean.

    The module's one convention for "what colour is the backdrop" — robust to
    near-white noise and not assuming pure white, since SD sometimes paints
    tints. Every site that needs it goes through here so the flatten and the
    front/back split can't drift apart on the definition.
    """
    n = len(ring)
    return tuple(round(sum(c[i] for c in ring) / n) for i in range(3))


def _border_is_uniform(ring, bg) -> bool:
    """Whether ``ring`` is a flat backdrop rather than a gradient/vignette.

    True when at least ``_BORDER_UNIFORM_FRACTION`` of the ring sits within
    ``_KEY_TOLERANCE`` of ``bg``. When it is false a single ``bg`` does not
    describe the backdrop, and anything keyed off one — the flatten, the
    front/back split — has to say so rather than quietly act on a colour that
    matches nothing.
    """
    near = sum(1 for c in ring if _rgb_distance(c, bg) <= _KEY_TOLERANCE)
    return near / len(ring) >= _BORDER_UNIFORM_FRACTION


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
    bg = _detect_background(ring)
    px = out.load()

    if not _border_is_uniform(ring, bg):
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


def _map_flat_to_palette(flat: Image.Image, flat_palette: list[int]) -> Image.Image:
    """Map a background-flattened RGB image onto a Gen-3 palette, keyed by mask.

    ``flat_palette`` is a flat ``[R, G, B, ...]`` palette with the key at entry
    0. The key must NOT compete in the nearest-colour search: it sits mid-range
    in the light-warm-gray/beige region of RGB space, so a creature midtone (or
    an anti-aliased seam between the 768-res faux pixels) closer to it than to
    any creature centroid would nearest-map onto transparency — the enclosed
    key speckle every sprite of the 2026-08-04 round carried (#90).
    ``_nudge_off_key`` cannot prevent that: it moves *centroids* off the key,
    but the per-pixel mapping was still free to pick the key entry.

    So index 0 is assigned purely from the flatten: pixels the flatten keyed
    are byte-exactly ``_KEY_COLOR`` and become index 0; every other pixel is
    nearest-mapped against the palette *minus* its key entry (dither off, so a
    diffused error can't smear creature pixels toward the key either) and
    shifted back by one to land on the full palette's indices. The clamp
    covers the pathological case of the mapping picking a zero-padded trailing
    entry — colour-identical to reserved black, so it is sent to slot 1.
    """
    nokey = Image.new("P", (1, 1))
    nokey.putpalette(flat_palette[3:])
    mapped = flat.quantize(palette=nokey, dither=Image.Dither.NONE)
    out = Image.new("P", flat.size)
    out.putpalette(flat_palette)
    out.putdata([
        0 if p == _KEY_COLOR else (i + 1 if i < 255 else 1)
        for p, i in zip(flat.get_flattened_data(), mapped.get_flattened_data())
    ])
    return out


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

    # Assemble the deterministic palette and map every pixel against it via
    # ``_map_flat_to_palette``: index 0 comes from the flatten's byte-exact key
    # pixels, creature pixels are mapped against the palette minus the key (so
    # a midtone near the key can never fall onto transparency), and creature
    # pixels near pure black/white legitimately snap to the reserved slots
    # without spending creature budget.
    reserved = [_KEY_COLOR, (0, 0, 0), (255, 255, 255)]
    creature_colors = _dedupe_by_display_depth(creature_colors, reserved)
    palette_colors = reserved + creature_colors
    flat_palette = [channel for color in palette_colors for channel in color]
    return _map_flat_to_palette(flat, flat_palette)


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
    validated here). The background colour comes from ``_detect_background``,
    the same convention the flatten uses.

    Only the middle ``_SPLIT_SEARCH_LOW``-``_SPLIT_SEARCH_HIGH`` fraction of
    columns is searched for the widest run of columns that are background for
    their *full height* (every pixel within ``_KEY_TOLERANCE`` of ``bg``) —
    that's where the gap between the two sprites is expected to fall. The cut
    lands at the widest run's centre column; ties go to the first (leftmost)
    run encountered. ``canvas`` is not mutated.

    Returns ``(front_half, back_half)`` crops, or ``None``. Three things about
    that contract the caller has to handle, none of which this slice decides:

    * **The halves are not equal width.** The cut tracks the actual gap, so on
      an off-centre generation one half is wider than the other — on a
      1536-wide canvas each can run ~614-920px. A caller that resizes both to
      one square distorts the two sprites by *different* aspect ratios; it
      wants ``_content_bbox`` and padding, not a straight stretch.
    * **``None`` means "no band", not "no two sprites".** A blank canvas, or
      one holding a single sprite off to one side, has plenty of full-height
      background and splits "successfully" into halves one of which is empty.
      Reroll logic keyed only on ``None`` will happily accept that.
    * **A gradient/vignette backdrop returns ``None``** even when the gap is
      plainly there, because one mean ``bg`` doesn't describe such a backdrop
      (the ring mean lands far from the gap's actual pixels). Rejected up front
      rather than scanned against a colour that matches nothing — same reason
      ``_flatten_background_to_key`` branches on ``_border_is_uniform``, though
      it warns and degrades where this reports no band and lets the caller
      reroll.
    """
    w, h = canvas.size
    ring = _border_ring(canvas)
    bg = _detect_background(ring)
    if not _border_is_uniform(ring, bg):
        return None

    px = canvas.load()
    x_start = int(_SPLIT_SEARCH_LOW * w)
    x_end = int(_SPLIT_SEARCH_HIGH * w)

    # Maximal ``[start, end)`` column ranges that are background for their full
    # height. The window is only 20% of the canvas, so this list stays short.
    runs = []
    run_start = None
    for x in range(x_start, x_end):
        if all(_rgb_distance(px[x, y], bg) <= _KEY_TOLERANCE for y in range(h)):
            if run_start is None:
                run_start = x
        elif run_start is not None:
            runs.append((run_start, x))
            run_start = None
    if run_start is not None:
        runs.append((run_start, x_end))

    if not runs:
        return None

    # Widest run wins; ``max`` returns the first of equal-width runs, so a tie
    # goes to the leftmost one.
    start, end = max(runs, key=lambda run: run[1] - run[0])
    cut = (start + end) // 2
    return canvas.crop((0, 0, cut, h)), canvas.crop((cut, 0, w, h))


def _content_columns(image: Image.Image, background) -> tuple[int, int] | None:
    """First and last columns of ``image`` holding a non-background pixel.

    Column-wise rather than a full bbox because the only caller squares a
    split half up, and that moves content horizontally only — the half is
    already exactly as tall as the square it lands on. ``None`` when every
    pixel is within ``_KEY_TOLERANCE`` of ``background`` (an empty half).
    """
    px = image.load()
    w, h = image.size
    columns = [
        x for x in range(w)
        if any(_rgb_distance(px[x, y], background) > _KEY_TOLERANCE for y in range(h))
    ]
    if not columns:
        return None
    return columns[0], columns[-1]


def _fit_half_to_square(half: Image.Image) -> Image.Image:
    """Sit a split half on a square background canvas without distorting it.

    The content-aware cut lands anywhere in the middle 20% of the canvas, so
    the two halves come out unequal widths: a gap centred at column 650 of a
    1536-wide canvas yields a 650px front and an 886px back. Handing those
    straight to ``postprocess`` / ``quantize_to_reference`` — both of which
    *resize* to a square — would stretch that front +18% horizontally and
    squeeze the back -13%, so front and back of one creature come out with
    visibly different proportions. That is worse geometry than the
    naive-midline fallback these halves exist to improve on, which cuts
    768/768 and distorts nothing.

    Pasting onto a square canvas of the half's own height keeps every pixel at
    1:1 in both axes instead: a narrow half gains background padding, a wide
    one is cropped to a square window, and a half that needs both gets both —
    PIL clips a negative paste offset and leaves the canvas showing through a
    positive one, so a single paste covers every case.

    The window is centred on the half's *content*, not on the half itself, so
    the creature lands mid-square however far off-centre the cut fell — which
    is also what makes the crop safe. Cropping only ever removes columns the
    content does not reach, so the sole way to lose content is a creature
    wider than the square, where a centred window is the best available answer
    anyway.
    """
    w, h = half.size
    background = _detect_background(_border_ring(half))
    columns = _content_columns(half, background)
    if columns is None:
        # Empty half: nothing to centre on, so centre the half itself.
        left = round((h - w) / 2)
    else:
        left = round(h / 2 - (columns[0] + columns[1] + 1) / 2)
    square = Image.new("RGB", (h, h), background)
    square.paste(half, (left, 0))
    return square


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


def silhouette_difference_ratio(a: Image.Image, b: Image.Image) -> float:
    """Fraction (0.0-1.0) of pixels whose background/creature status differs.

    A pixel counts only when it is the key (index 0) in exactly one of the two
    same-palette P-mode images — the outline moved there. This is the part of
    ``difference_ratio`` that measures *motion*: it is blind to a body pixel
    changing colour in place, so palette churn cannot inflate it.
    """
    if a.size != b.size:
        raise ValueError(f"Cannot compare different-size images: {a.size} vs {b.size}")
    data_a = a.get_flattened_data()
    data_b = b.get_flattened_data()
    differing = sum(1 for x, y in zip(data_a, data_b) if (x == 0) != (y == 0))
    return differing / len(data_a)


def color_churn_ratio(a: Image.Image, b: Image.Image) -> float:
    """Fraction (0.0-1.0) of pixels that are creature in both images but differ.

    The complement of ``silhouette_difference_ratio`` within the diff: index
    changes where both images agree the creature is — reshaded fur,
    re-quantized boundaries — which read as colour flicker, not motion, when
    the frames alternate.
    """
    if a.size != b.size:
        raise ValueError(f"Cannot compare different-size images: {a.size} vs {b.size}")
    data_a = a.get_flattened_data()
    data_b = b.get_flattened_data()
    churn = sum(1 for x, y in zip(data_a, data_b) if x != y and x != 0 and y != 0)
    return churn / len(data_a)


# Tunable eyeball placeholders for ``build_frame2``'s structural gate,
# calibrated against the 2026-08-04 round (#90). The silhouette floor is under
# one procedural squash's worth of outline motion (a 96px squash measures
# ~0.013, the round's 768px squash frames 0.032-0.083); the ceiling marks
# identity drift / teleporting. The churn cap is measured against the *squash*
# (whose interior carries the intended motion), so genuine interior movement is
# free and only unfaithful reshading pays: the round's 25 flicker false
# positives all carried well over 0.05 of it.
_FRAME2_MIN_SILHOUETTE = 0.01
_FRAME2_MAX_SILHOUETTE = 0.10
_FRAME2_MAX_COLOR_CHURN = 0.05


def build_frame2(
    frame1: Image.Image,
    candidate: Image.Image | None = None,
    min_silhouette: float = _FRAME2_MIN_SILHOUETTE,
    max_silhouette: float = _FRAME2_MAX_SILHOUETTE,
    max_churn: float = _FRAME2_MAX_COLOR_CHURN,
) -> Image.Image:
    """Turn frame 1 plus an optional candidate into a guaranteed-valid frame 2.

    The procedural squash is the default frame 2; a palette-locked, recentred
    candidate replaces it only when it passes the structural gate, whose three
    conditions each kill one observed failure mode (#90):

    * silhouette motion vs frame 1 inside ``[min_silhouette, max_silhouette]``
      — below is a static pose, above is identity drift / teleporting;
    * silhouette *closer to the squash* than to frame 1 — the motion is the
      intended pose change, not outline noise (the round's img2img candidates
      mostly denoised the squash init back to frame 1's pose);
    * colour churn vs the squash at most ``max_churn`` — the squash's interior
      carries the intended movement, so honest interior motion costs nothing
      here and only reshading (flicker) trips the cap.

    Gating on raw ``difference_ratio`` vs frame 1 was the #90 false positive:
    img2img + re-quantization churn alone flipped 5-25% of indices with no
    pose change, so flicker always read as motion. Always returns a valid
    P-mode frame of frame 1's size sharing its palette.

    The thresholds are tunable eyeball placeholders (see the module spec); a
    later ML slice or a human is expected to tune them.
    """
    if frame1.mode != "P":
        raise ValueError(f"Expected palette-mode frame1 image, got {frame1.mode}")
    squash = procedural_squash(frame1)
    if candidate is None:
        return squash
    # enhance=False: the candidate's colours derive from frame 1's already-
    # enhanced palette (via the squash init), so the lock must not enhance
    # them a second time — see quantize_to_reference. Also resizes to frame1's
    # size.
    locked = quantize_to_reference(candidate, frame1, enhance=False)
    recentred = recenter_to_anchor(locked, frame1)
    silhouette = silhouette_difference_ratio(recentred, frame1)
    if (
        min_silhouette <= silhouette <= max_silhouette
        and silhouette_difference_ratio(recentred, squash) < silhouette
        and color_churn_ratio(recentred, squash) <= max_churn
    ):
        return recentred
    return squash


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
    result = pipeline(
        prompt=build_prompt(prompt, extra_tags),
        negative_prompt=_NEGATIVE_PROMPT,
        width=_GEN_SIZE,
        height=_GEN_SIZE,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
    )
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def _split_front_back_with_retry(canvas: Image.Image, regenerate):
    """Split a front/back canvas, rerolling once and falling back to a naive cut.

    Torch-free: ``regenerate`` is a zero-arg callable the caller supplies to
    produce a fresh canvas (``generate_sprite_pair`` wires it to a second
    ``pipeline`` call), kept external so this retry/fallback decision stays a
    plain, unit-testable function. Tries ``split_front_back_canvas`` on
    ``canvas``; on failure calls ``regenerate()`` once and tries again on the
    fresh canvas; on a second failure falls back to a naive midline split of
    that fresh canvas and warns. Mirrors ``_flatten_background_to_key``'s
    gradient-border fallback: a best-effort result plus a ``stderr`` warning,
    never a raise — including when ``regenerate`` itself raises.
    """
    result = split_front_back_canvas(canvas)
    if result is not None:
        return result

    try:
        fresh = regenerate()
    except Exception as exc:
        # A failed reroll (a transient OOM on the second 1536-wide render, say)
        # must not cost the caller the front sprite the first canvas can still
        # give up — fall through to a naive split of the canvas already in hand.
        print(
            f"warning: _split_front_back_with_retry reroll render failed ({exc}); "
            "falling back to a naive midline split of the first canvas",
            file=sys.stderr,
        )
    else:
        result = split_front_back_canvas(fresh)
        if result is not None:
            return result
        canvas = fresh
        print(
            "warning: _split_front_back_with_retry found no clean split column even "
            "after a reroll; falling back to a naive midline split",
            file=sys.stderr,
        )

    w, h = canvas.size
    cut = w // 2
    return canvas.crop((0, 0, cut, h)), canvas.crop((cut, 0, w, h))


def generate_sprite_pair(
    prompt: str, types: list[str], front_output_path: str, back_output_path: str,
    *, pipeline, seed: int | None = None,
) -> None:
    """Generate a front+back sprite pair from one side-by-side SDXL canvas.

    One ``pipeline`` txt2img call renders a ``_PAIR_WIDTH`` x ``_GEN_SIZE``
    canvas — front on the left half, back on the right half, per the fused
    back&front LoRA. ``_split_front_back_with_retry`` cuts the two apart
    (rerolling with ``seed + 1`` once, then falling back to a naive midline
    split, if the content-aware split fails), and ``_fit_half_to_square``
    squares each half up so the off-centre cut costs neither view its
    proportions. The front is quantized adaptively via ``postprocess`` and
    always saved. The back is locked to the front's exact palette via
    ``quantize_to_reference``; if it comes back empty/background-only (every
    pixel at the Gen-3 contract's key index 0), it is skipped with a ``stderr``
    warning instead of being saved — this function never raises for a split or
    empty-back degradation, only for a genuine ``pipeline`` failure.
    """
    def _render(render_seed):
        result = pipeline(
            prompt=build_prompt(prompt),
            negative_prompt=_NEGATIVE_PROMPT,
            width=_PAIR_WIDTH,
            height=_GEN_SIZE,
            num_inference_steps=_NUM_STEPS,
            guidance_scale=_CFG_SCALE,
            generator=_make_generator(render_seed),
        )
        return result.images[0]

    canvas = _render(seed)
    reroll_seed = seed + 1 if seed is not None else None
    front_raw, back_raw = _split_front_back_with_retry(canvas, lambda: _render(reroll_seed))

    front = postprocess(_fit_half_to_square(front_raw))
    front.save(front_output_path)

    back = quantize_to_reference(_fit_half_to_square(back_raw), front)
    if _content_bbox(back, background=0) is None:
        print(
            f"warning: generate_sprite_pair back half for {back_output_path} is "
            "empty/background-only; skipping back sprite",
            file=sys.stderr,
        )
        return
    back.save(back_output_path)


def _run_img2img_on_image(
    prompt: str, types: list[str], init: Image.Image, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None, strength: float = 0.8,
) -> Image.Image:
    """Run the img2img pipeline against an already-prepared RGB init image.

    Shared by ``_run_img2img`` (which loads the init image from disk) and
    ``generate_frame2`` (which builds its init image in-memory from
    ``procedural_squash`` rather than loading one from a path).
    """
    result = pipeline(
        prompt=build_prompt(prompt, extra_tags),
        negative_prompt=_NEGATIVE_PROMPT,
        image=init,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
        strength=strength,
    )
    return result.images[0]


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
    return _run_img2img_on_image(
        prompt, types, init, pipeline=pipeline,
        extra_tags=extra_tags, seed=seed, strength=strength,
    )


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
    seed: int | None = None, strength: float = 0.30, extra_tags: list[str] | None = None,
) -> None:
    """Generate the second front-animation frame and save it to ``output_path``.

    Runs img2img from ``procedural_squash(frame1)`` (not the raw front
    sprite) at low ``strength`` with an animation tag (defaults to
    ``["open mouth"]``) using frame 1's seed. Seeding the init image with the
    squash guarantees a real structural pose change to clean up, rather than
    the near-identical recolour img2img produces from an unmodified init
    image. The raw RGB candidate is then handed to ``build_frame2`` — which
    palette-locks + recenters it, accepts it iff it passes the structural gate
    (real silhouette motion, low colour churn), and otherwise falls back to
    the procedural squash itself. The result always shares frame 1's exact
    16-colour palette.
    """
    frame1 = Image.open(front_sprite_path)
    squash_init = procedural_squash(frame1).convert("RGB").resize(
        (_GEN_SIZE, _GEN_SIZE), Image.LANCZOS
    )
    candidate = _run_img2img_on_image(
        prompt, types, squash_init, pipeline=pipeline,
        extra_tags=extra_tags or ["open mouth"], seed=seed, strength=strength,
    )
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
    """Derive an img2img pipeline that reuses ``txt2img_pipe``'s loaded components.

    The derived pipe gets its **own** ``enable_model_cpu_offload()`` rather than
    riding on the parent's. The offload hooks live on the shared modules, but
    the ``_all_hooks`` bookkeeping they are driven through lives on the
    *pipeline* — so without this call the derived pipe's
    ``maybe_free_model_hooks()`` (which every diffusers ``__call__`` runs on the
    way out) silently does nothing, and whatever component ran last — the VAE,
    upcast to fp32 to decode — stays GPU-resident until the next run evicts it.
    That is the residency the 8GB budget cannot afford.

    Enabling it on a pipe that shares another's components is safe here because
    the two are the same pipeline in all the ways offload cares about: identical
    ``model_cpu_offload_seq`` over identical modules. ``enable_model_cpu_offload``
    strips every hook and reinstalls from scratch, so whichever pipe ran last
    leaves the modules in exactly the state the other would have built anyway.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe = StableDiffusionXLImg2ImgPipeline(**txt2img_pipe.components)
    return _enable_vram_measures(pipe)


def _device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _apply_lora(pipe) -> None:
    from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

    state_dict, network_alphas, metadata = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
        str(_LORA_PATH), return_lora_metadata=True, unet_config=pipe.unet.config
    )
    pipe.load_lora_into_unet(
        state_dict, network_alphas=network_alphas, unet=pipe.unet,
        metadata=metadata, _pipeline=pipe,
    )

    def _drop_text_model(d, prefix):
        if not d:
            return d
        old = f"{prefix}.text_model."
        new = f"{prefix}."
        return {new + k[len(old):] if k.startswith(old) else k: v for k, v in d.items()}

    # te1 (CLIPTextModel) names its modules WITHOUT the "text_model." wrapper level
    # (needs the strip); te2 (CLIPTextModelWithProjection) names them WITH it (keys
    # must stay untouched) -- verified against named_modules() of each encoder.
    for encoder, prefix, fix in ((pipe.text_encoder, "text_encoder", True),
                                 (pipe.text_encoder_2, "text_encoder_2", False)):
        sd = _drop_text_model(state_dict, prefix) if fix else state_dict
        al = _drop_text_model(network_alphas, prefix) if fix else network_alphas
        pipe.load_lora_into_text_encoder(
            sd, network_alphas=al, text_encoder=encoder, prefix=prefix,
            lora_scale=pipe.lora_scale, metadata=metadata, _pipeline=pipe,
        )
    pipe.fuse_lora(lora_scale=_LORA_SCALE)


def _enable_vram_measures(pipe):
    """Apply the CUDA-only VRAM measures to ``pipe`` and return it.

    Mandatory on CUDA (not opt-in) to stay inside the 8GB budget, and applied to
    every pipeline that is ever called — including one derived from another's
    components (see ``make_img2img_pipeline``). Off CUDA this is a no-op:
    ``enable_model_cpu_offload`` needs an accelerator and raises without one.

    ``enable_model_cpu_offload`` installs hooks that move each component to the
    GPU only while it runs, so from here on *it* owns device placement: a
    ``pipe.to("cuda")`` afterwards would make every component resident at once
    and hand the offload's savings straight back (diffusers warns on exactly
    this combination), so the move is deliberately skipped on this path.
    """
    import torch
    if not torch.cuda.is_available():
        return pipe
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    else:
        pipe.vae.enable_tiling()
    return pipe


def _load_base_pipeline(pipe_cls):
    from diffusers import EulerAncestralDiscreteScheduler

    device, dtype = _device_and_dtype()
    pipe = pipe_cls.from_pretrained(_BASE_MODEL_ID, torch_dtype=dtype)
    _apply_lora(pipe)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if device != "cuda":
        return pipe.to(device)
    return _enable_vram_measures(pipe)


def load_txt2img_pipeline():
    try:
        from diffusers import StableDiffusionXLPipeline
        return _load_base_pipeline(StableDiffusionXLPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)


def load_img2img_pipeline():
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        return _load_base_pipeline(StableDiffusionXLImg2ImgPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)
