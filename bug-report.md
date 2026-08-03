# Bug: global background sweep eats legitimate near-background creature detail

## Summary

`_flatten_background_to_key`'s stage 2 ("global near-background sweep") keys
*any* pixel in the whole image that is within `_KEY_TOLERANCE` of the
detected background colour, with no check for whether that pixel is actually
part of a background region. On sprites with near-white/near-background
creature detail (a shield highlight, a white belly patch, etc.) this recolors
legitimate creature pixels to `_KEY_COLOR`, corrupting the sprite.

## Repro steps

Pure-PIL repro, no SD/torch needed — run from repo root:

```python
from PIL import Image, ImageDraw
from fakemon_forge.sprites import _flatten_background_to_key, _KEY_COLOR

img = Image.new("RGB", (96, 96), (250, 250, 250))          # background
d = ImageDraw.Draw(img)
d.ellipse((20, 20, 76, 76), fill=(60, 120, 200))            # creature body
d.ellipse((44, 44, 52, 52), fill=(245, 245, 245))            # "shield highlight" detail, near-bg colour

out = _flatten_background_to_key(img)
px = out.load()
print(px[48, 48])          # -> (200, 200, 168), i.e. _KEY_COLOR
print(px[48, 48] == _KEY_COLOR)   # True
```

The highlight patch sits entirely inside the creature body (surrounded on
all sides by opaque creature-coloured pixels, nowhere near the border) yet
is recoloured to the transparency key.

## Expected vs. actual

- **Expected:** only actual background — the connected outer background
  (stage 1) plus genuinely enclosed background pockets (gaps between legs,
  the hole of a ring-shaped creature) — is keyed. Creature detail that
  merely happens to be a similar colour to the background, without forming
  an enclosed background region, must be left untouched.
- **Actual:** stage 2 (`fakemon_forge/sprites.py:225-231`) does an
  unconditional per-pixel colour-distance scan over the entire image and
  keys every matching pixel regardless of position, adjacency, or whether it
  belongs to a background-connected component. Any creature pixel that
  happens to fall within `_KEY_TOLERANCE` of the detected background colour
  is keyed, whether or not it is part of a background pocket.

## Root cause

**Confirmed.** `fakemon_forge/sprites.py:225-231`:

```python
# Stage 2: key any remaining near-background pixels (enclosed pockets).
px = out.load()
for y in range(h):
    for x in range(w):
        p = px[x, y]
        if p != _KEY_COLOR and _rgb_distance(p, bg) <= _KEY_TOLERANCE:
            px[x, y] = _KEY_COLOR
```

This loop has no notion of connectivity or enclosure at all — it is a flat
scan of every pixel in the image, keying anything colour-close to `bg`. The
docstring's own framing ("enclosed pockets ... the flood cannot reach") is
aspirational; nothing in the implementation restricts the sweep to pixels
that are actually part of an enclosed background region. It coincidentally
does the right thing for the enclosed-pocket case (the existing
`test_flatten_keys_enclosed_pocket_via_global_sweep` test), because pocket
pixels do happen to be colour-close to `bg` — but the same unconditional
rule fires on any other pixel in the image with a similar colour, including
legitimate creature detail. Stage 1 (border flood fill, lines 220-223) is
correctly connectivity-based (`ImageDraw.floodfill` from the four corners)
and is not implicated.

I verified this is the actual mechanism (not, e.g., stage 1's flood fill
leaking into the creature) by disabling stage 2 in a scratch interpreter
session and confirming the highlight patch survives untouched when only
stage 1 runs, then re-enabling stage 2 and confirming it gets keyed — the
defect is isolated to the stage-2 loop's lack of any connectivity/enclosure
check.

## Affected files

- `fakemon_forge/sprites.py` — `_flatten_background_to_key` (lines 171-232,
  the bug is specifically lines 225-231)
- `fakemon_forge/sprites.py` — `_quantize_gen3` and `quantize_to_reference`
  both call `_flatten_background_to_key` and inherit the corruption
  downstream into the final palette-quantized sprite.

## Regression info

Not a regression. `git blame HEAD -L 171,232 -- fakemon_forge/sprites.py`
shows the entire function, including the stage-2 loop, was introduced whole
in a single commit:

```
8ff6e7f8 keep: implement (keep/dc83a6d7)   2026-07-31 16:30:47 +0200   Marcos Bragagnolo
```

There is no earlier, correct version of this function to regress from — the
global-sweep design has been present since the function's introduction.
(Note: `git status`/`git diff HEAD` currently shows `fakemon_forge/sprites.py`
as modified in the working tree, but that diff is CRLF line-ending noise
only — confirmed via `git diff` line-count parity and a whitespace-stripped
comparison of old vs. new hunks; the actual code logic in the working tree
is identical to `HEAD`.)

## Proposed fix approach

Replace stage 2's flat pixel scan with connected-component analysis, as
scoped in issue #63:

1. Build a mask of pixels within `_KEY_TOLERANCE` of `bg` that are not
   already keyed by stage 1.
2. Find its 8-connected components.
3. Key only components that do not touch the image border (border-touching
   near-bg components would mean stage 1's flood fill failed to reach them,
   e.g. through a `<30`-tolerance chokepoint — those still belong to the
   outer background, not an interior pocket).
4. Leave all other near-bg-coloured pixels untouched.

This keeps stage 1 (correct, connectivity-based) and the gradient/vignette
fallback path untouched, and only rewrites the stage-2 loop plus its
docstring paragraph.

One open design question for the fix, worth flagging rather than silently
resolving: a small isolated near-bg-coloured patch fully surrounded by
creature pixels (this bug's repro case) and a genuine enclosed background
pocket (existing `_ring_sprite` test) are both, in the general case,
topologically identical — an 8-connected island of near-bg pixels, touching
neither the border nor stage-1's keyed region. Pure colour+connectivity
analysis cannot always distinguish "a real hole in the silhouette" from "a
same-coloured detail patch" from geometry alone. Component *size* relative
to the sprite (real leg-gaps/ring-holes are proportionally small but not
single-highlight-sized) may be the practical disambiguator worth
considering during implementation, but that's a judgement call for the fix
phase, not asserted as fact here.
