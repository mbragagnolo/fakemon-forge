# Bug: orphan-pixel artifacts from single-sample RGB downscaling in the sprite/icon pipeline

## Summary

Two RGB downscale sites in the sprite/icon post-processing pipeline produce
"orphan pixel" artifacts — output pixels whose colour doesn't reflect the
dominant colour of the source region they were downsampled from, breaking
alignment with the intended pixel-art grid:

- `fakemon_forge/sprites.py::stitch_spritesheet` (line 585): downscales each
  768x768 view to a `cell_size` (default 64) cell using `Image.NEAREST`.
- `fakemon_forge/icon.py::_build_frame1` (line 116): downscales the 768x768
  source sprite to 32x32 using `Image.LANCZOS`.

## Repro steps

Reproduced the two distinct failure mechanisms with throwaway scripts (not
committed) against the actual PIL 12.3.0 installed in this environment:

1. **NEAREST (stitch_spritesheet)**: built a 12x12 synthetic source tile (the
   768/64 = 12x downscale ratio used in production) filled with the
   background key colour `(200, 200, 168)` plus 3 sparse "orphan" dark
   pixels `(30, 30, 30)` scattered at random positions (simulating
   anti-aliasing/outline noise from the SD render). `tile.resize((1, 1),
   Image.NEAREST)` samples exactly one fixed-offset source pixel per output
   pixel. Over 200 randomized trials, that fixed sample point landed on an
   orphan pixel instead of the dominant background in ~0.5–1% of cases —
   i.e. NEAREST has no mechanism to prefer the tile's dominant colour, so any
   single-pixel noise in the source tile can silently flip an output pixel to
   the wrong colour.

2. **LANCZOS (icon._build_frame1)**: built a 12x12 image with a hard vertical
   edge, red `(255, 0, 0)` on the left half and blue `(0, 0, 255)` on the
   right. `img.resize((2, 1), Image.LANCZOS)` produced `(232, 0, 23)` and
   `(23, 0, 232)` — colours that exist **nowhere** in the source image.
   LANCZOS's negative-lobe ringing at hard edges (which pixel art is full of)
   invents blended colours outright, worse than NEAREST's "wrong but at
   least real" sample.

Both mechanisms are confirmed causes of "orphan pixel" / grid-misaligned
artifacts in the downscaled sprite and icon output.

## Expected vs. actual

- **Expected**: each output pixel's colour should reflect the dominant
  colour actually present in the corresponding source tile, since the
  source is pixel art (or near-pixel-art after SD generation) being
  downsampled to a smaller pixel grid — not photographic content where
  blending/arbitrary sampling is acceptable.
- **Actual**: `Image.NEAREST` picks one arbitrary fixed-offset sample per
  tile (ignoring the tile's actual colour distribution), and `Image.LANCZOS`
  blends across the tile and can synthesize colours absent from the source
  entirely. Both produce visibly wrong, grid-misaligned pixels in the final
  sprite/icon.

## Root cause

**Confirmed finding**: `stitch_spritesheet` (sprites.py:585) and
`_build_frame1` (icon.py:116) downscale raw RGB image data using resampling
filters (`NEAREST`, `LANCZOS`) that each select or synthesize a value from
only a narrow/weighted slice of the source tile, rather than a value
representative of the tile's dominant colour. This is architecturally
correct given PIL's built-in resize filters — none of them do "majority
colour of tile" reduction, which is what pixel-art downscaling needs. A
k-centroid (cluster source tile → take the largest cluster's colour)
downscale, as verified in the issue's prior spike research (issue #61), is
not present anywhere in this codebase — there is no existing k-centroid
implementation to route these call sites through.

This is **not a regression**: no such k-centroid function exists in the
codebase's history, and the existing NEAREST/LANCZOS calls are the original,
intentional design in both files (see the docstrings at sprites.py:566-577
and icon.py:107-115, which explicitly justify NEAREST/LANCZOS as
"single downscale, no chaining" — a real property, just not the one that
prevents orphan pixels). It's a known limitation being addressed as a
planned enhancement, not a broken behavior that used to work correctly.

## Affected files

- `fakemon_forge/sprites.py` — `stitch_spritesheet`, line 585 (`Image.NEAREST`
  cell resize). No `k_centroid` helper currently exists near `_quantize_gen3`/
  `postprocess` (sprites.py:59, 231).
- `fakemon_forge/icon.py` — `_build_frame1`, line 116 (`Image.LANCZOS` resize).
  Already imports private helpers from `sprites.py` (`_KEY_COLOR`,
  `_KEY_TOLERANCE`, `_rgb_distance` at icon.py:17), so a new `k_centroid`
  import would follow the same pattern.

Confirmed NOT affected (must stay untouched, per issue scope):
- `sprites.py::procedural_squash` (NEAREST on P-mode/palette-indexed data,
  line 360 — comment at line 359 explicitly notes this is safe because
  NEAREST on P-mode indices can't introduce new colours).
- `sprites.py::quantize_to_reference` (NEAREST immediately followed by
  quantize-to-reference-palette, line 93).
- `sprites.py::_quantize_gen3` (NEAREST immediately followed by
  quantize-to-fixed-palette, line 246).
- `fakemon_forge/footprint.py` — reads the sprite at native size; no RGB
  downscale call site exists here today, confirmed by grep (only `resize`
  hits in this codebase are the 5 listed above, in sprites.py/icon.py).

## Regression info

Not applicable — not a regression. No k-centroid implementation has ever
existed in this codebase (confirmed via `git log`/`grep`); this is a net-new
enhancement requested in issue #62, informed by spike research done for
issue #61 (commits `0ad3255`, `e0a2512`, `d974827`, `f2e869c` on this branch).

## Proposed fix approach

1. Add `k_centroid(image, width, height, centroids=2)` to `sprites.py`, next
   to `_quantize_gen3`/`postprocess` (both pure-PIL, no torch/diffusers
   import), using the algorithm given in issue #62 (MIT-licensed, ported
   from Astropulse's "pixeldetector"): for each output pixel, crop the
   corresponding source tile, `quantize(colors=centroids, method=1,
   kmeans=centroids)` to cluster it, and take the centroid with the most
   pixels as the output colour.
2. Replace `stitch_spritesheet`'s `cell.resize((cell_size, cell_size),
   Image.NEAREST)` (sprites.py:585) with `k_centroid(cell, cell_size,
   cell_size)`.
3. Replace `_build_frame1`'s `source.convert("RGB").resize((_ICON_SIZE,
   _ICON_SIZE), Image.LANCZOS)` (icon.py:116) with
   `k_centroid(source.convert("RGB"), _ICON_SIZE, _ICON_SIZE)`, importing
   `k_centroid` from `fakemon_forge.sprites` alongside the existing
   `_KEY_COLOR`/`_KEY_TOLERANCE`/`_rgb_distance` import (icon.py:17).
4. Leave `procedural_squash`, `quantize_to_reference`, and `_quantize_gen3`'s
   NEAREST calls untouched (palette-locked or already palette-indexed —
   out of scope, per issue #62).
5. Add unit tests to `tests/test_sprites.py` for `k_centroid` (output size,
   RGB mode, and a hard-edge two-colour-half synthetic image asserting the
   output only contains the two source colours — no invented colours,
   unlike the LANCZOS repro above). Update
   `test_spritesheet_downscale_introduces_no_new_colors`
   (tests/test_sprites.py:986) since it currently only documents the NEAREST
   guarantee in its comment (line 987) — the assertion itself
   (`sheet_colors <= allowed`) is filter-agnostic and should still pass
   under k_centroid, but the comment needs updating. Add/confirm a
   `tests/test_icon.py` test that `_build_frame1` still yields a <=16-colour
   opaque 32x32 `P`-mode frame after the swap.
