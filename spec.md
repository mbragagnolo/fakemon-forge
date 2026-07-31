# Spec: Pure-PIL frame-2 assembly (procedural squash, anchoring, diff-ratio, accept-or-fallback)

## Summary

`fakemon-forge` is gaining an Emerald-style two-frame front animation: every
front sprite becomes two frames played on battle entry. This slice adds the
**pure-PIL decision core** that turns "frame 1 (the finished 96×96 `P`-mode
front sprite) plus an optional candidate second frame" into a guaranteed-valid
frame 2 — with **no ML involved**, so it is fully unit-testable in the
torch-free keep sandbox (only pytest + Pillow present).

Five new pure-PIL helpers are added to `fakemon_forge/sprites.py`:

1. `procedural_squash(frame1, amount_px=2)` — bottom-anchored vertical squash
   (the classic Gen-3 breathing/bounce), used as the fallback frame 2.
2. A private background-index helper — identifies frame 1's background palette
   index (default: most common index).
3. `recenter_to_anchor(candidate, frame1)` — translates `candidate` so its
   non-background bounding box aligns to frame 1's anchor (bottom-center),
   killing position jitter between frames.
4. `difference_ratio(a, b)` — fraction (0.0–1.0) of differing pixels.
5. `build_frame2(frame1, candidate=None, low=0.02, high=0.30)` — the
   accept-or-fallback orchestrator that always returns a valid 96×96 `P`-mode
   frame sharing frame 1's exact palette.

Nothing is wired into `main`, the CLI, or any ML path in this slice — this
lands the tested library layer only. A later slice generates the img2img
candidate and wires `build_frame2` into the pipeline.

This slice depends on `quantize_to_reference(image, reference)` (already in
`sprites.py`), which quantizes an image against a reference `P`-mode image's
exact 16-colour palette. Every frame 2 must share frame 1's palette; reuse this
helper (or operate in P-space and re-lock) to guarantee that.

## Inputs

Shared conventions (from existing `sprites.py`):

- `_SPRITE_SIZE = 96`, `_PALETTE_COLORS = 16`. Sprites are 96×96 `P`-mode
  images with a ≤16-colour palette produced by `postprocess` /
  `quantize_to_reference`.
- A `P`-mode image's palette is retrieved with `.getpalette()` (a flat
  `[R,G,B, R,G,B, …]` list, 768 entries) and its pixel indices with
  `.get_flattened_data()` (Pillow ≥ 12; the codebase pins Pillow 12.3).
  "Shares frame 1's palette" is defined operationally as
  `out.getpalette() == frame1.getpalette()`.

Per-helper:

- **`procedural_squash(frame1, amount_px=2)`**
  - `frame1: PIL.Image.Image` — a 96×96 `P`-mode front sprite.
  - `amount_px: int = 2` — vertical pixels to compress off the top; tunable.
- **`_background_index(frame1)`** (private helper; exact name at
  implementer's discretion, e.g. `_background_index` / `_bg_index`)
  - `frame1: PIL.Image.Image` — a `P`-mode image.
- **`recenter_to_anchor(candidate, frame1)`**
  - `candidate: PIL.Image.Image` — 96×96 frame-2 candidate. In the
    `build_frame2` flow it has already been palette-locked to `frame1`, so it
    is `P`-mode sharing frame 1's palette; standalone callers are expected to
    pass a `P`-mode candidate that already shares frame 1's palette.
  - `frame1: PIL.Image.Image` — the reference 96×96 `P`-mode front sprite.
- **`difference_ratio(a, b)`**
  - `a, b: PIL.Image.Image` — two **same-size** images. In practice both are
    96×96 `P`-mode sharing one palette, so index-wise comparison is meaningful.
- **`build_frame2(frame1, candidate=None, low=0.02, high=0.30)`**
  - `frame1: PIL.Image.Image` — 96×96 `P`-mode finished front sprite.
  - `candidate: PIL.Image.Image | None = None` — optional img2img second frame
    (any mode/size; it is passed through `quantize_to_reference`, which resizes
    to 96×96 and locks the palette).
  - `low: float = 0.02`, `high: float = 0.30` — acceptance band on
    `difference_ratio`; tunable.

## Outputs

- **`procedural_squash`** → 96×96 `P`-mode image, `getpalette()` equal to
  `frame1`'s, whose content is a bottom-anchored vertical squash of frame 1.
- **`_background_index`** → `int`, the chosen background palette index.
- **`recenter_to_anchor`** → 96×96 `P`-mode image sharing frame 1's palette,
  with `candidate`'s content translated so its non-background bbox is anchored
  bottom-centre to frame 1's.
- **`difference_ratio`** → `float` in `[0.0, 1.0]`.
- **`build_frame2`** → 96×96 `P`-mode image sharing frame 1's exact palette:
  either the accepted (palette-locked, recentred) candidate, or
  `procedural_squash(frame1)`.

All helpers are non-mutating: inputs are not modified (matches
`quantize_to_reference` / `postprocess` behaviour and their tests).

## Behavior

### `procedural_squash(frame1, amount_px=2)`

1. Determine the background index via `_background_index(frame1)`.
2. Vertically squash frame 1's content to `96 × (96 - amount_px)`. To preserve
   the palette (introduce no new colours), resize **in P-space** with
   `Image.NEAREST` (nearest-neighbour reuses existing indices; palette carries
   over on a `P`-mode resize).
3. Create a fresh 96×96 `P` canvas filled with the background index and give it
   frame 1's palette (`putpalette(frame1.getpalette())`).
4. Paste the squashed content anchored to the **bottom**: paste at `(0,
   amount_px)` so the bottom row aligns with the canvas bottom and the top
   `amount_px` rows become background. This keeps the creature's feet planted
   while the top compresses (the breathing look).
5. Re-lock the palette to guarantee `getpalette() == frame1.getpalette()`
   (either the operate-in-P-space construction above already guarantees this,
   or round-trip through `quantize_to_reference(result, frame1)`). The
   P-space construction is preferred because it is exact and avoids a re-quantize
   that could shift indices; `quantize_to_reference` is the documented fallback.
6. Result differs from frame 1 (the top few rows changed to background /
   shifted content) but by a **small** amount — by construction it should land
   inside the default acceptance band, since this is what ships when a
   candidate is rejected.

`amount_px` is a tunable eyeball placeholder (see Assumptions); a code comment
must say so.

### `_background_index(frame1)`

Return the **most common palette index** in `frame1` (default choice, stated in
a code comment). Implementation: from `frame1.getcolors()` (or a histogram /
`Counter` over `get_flattened_data()`), pick the index with the highest count.
This index defines "background" for squash canvas fill and for bbox detection.

### `recenter_to_anchor(candidate, frame1)`

1. Determine the background index (from `frame1`; `candidate` shares its palette
   so the same index applies).
2. Compute each image's **non-background bounding box**: build a mask that is
   nonzero where the pixel index ≠ background index, and take its `getbbox()`.
3. Compute each bbox's **anchor** = bottom-centre: `anchor_x = (left + right) /
   2`, `anchor_y = bottom` (bbox coords are `(left, top, right, bottom)` with
   `right`/`bottom` exclusive).
4. Crop `candidate` to its own bbox to get the content, then paste that content
   onto a fresh 96×96 `P` canvas (filled with background index, frame 1's
   palette) so the content's bottom-centre lands on frame 1's anchor:
   `paste_left = round(frame1_anchor_x - content_width / 2)`,
   `paste_top = round(frame1_anchor_y - content_height)`.
5. Return the recentred `P` canvas; `getpalette() == frame1.getpalette()`.

Rationale: aligning bboxes bottom-centre prevents the second frame from
appearing to teleport/jitter relative to the first (feet-planted breathing).

### `difference_ratio(a, b)`

Fraction of pixels that differ: `differing_pixel_count / total_pixels`. For
same-palette `P` images, compare pixel indices directly (e.g. count positions
where `a.get_flattened_data() != b.get_flattened_data()`, or use
`ImageChops.difference` and count nonzero pixels). `difference_ratio(x, x)` is
exactly `0.0`; two images differing in every pixel give `1.0`.

### `build_frame2(frame1, candidate=None, low=0.02, high=0.30)`

1. If `candidate is None`: return `procedural_squash(frame1)`.
2. Otherwise:
   a. Palette-lock: `locked = quantize_to_reference(candidate, frame1)` (also
      resizes to 96×96).
   b. Anchor: `recentred = recenter_to_anchor(locked, frame1)`.
   c. Measure: `ratio = difference_ratio(recentred, frame1)`.
   d. **Accept** the candidate (return `recentred`) iff `low <= ratio <= high`.
      - `ratio < low` → too identical → texture shimmer, not motion → reject.
      - `ratio > high` → identity drift / teleporting → reject.
   e. On rejection, return `procedural_squash(frame1)`.
3. Always returns a valid 96×96 `P`-mode frame with
   `getpalette() == frame1.getpalette()`.

`low` / `high` are tunable eyeball placeholders (see Assumptions); a code
comment must say so.

## Edge cases

- **`amount_px` at bounds**: `amount_px=0` yields a frame equal to frame 1
  (difference ratio 0) — the default is 2 and callers are not expected to pass
  0, but the function should still produce a valid 96×96 `P` image.
  `amount_px >= 96` would leave zero content height; treat `amount_px` as
  expected in `1..95` (the sensible squash range) and document that.
- **All-background frame / empty bbox**: if a frame is entirely the background
  index, `getbbox()` on the mask returns `None`. `recenter_to_anchor` must not
  crash — fall back to returning the candidate re-locked to frame 1's palette
  unchanged (no translation possible without a bbox).
- **Candidate exactly equal to frame 1** after locking+recentring →
  `ratio == 0.0 < low` → falls back to squash.
- **Candidate wildly different** (e.g. inverted / noise) → `ratio > high` →
  falls back to squash.
- **In-band candidate** → returned as the palette-locked, recentred image,
  sharing frame 1's palette.
- **Non-square / non-96 candidate**: handled by `quantize_to_reference`, which
  resizes to 96×96 before `recenter_to_anchor` sees it.
- **Palette exactness**: nearest-neighbour P-space resize and paste-onto-
  bg-canvas keep indices valid; the returned palette must be byte-for-byte
  equal to `frame1.getpalette()`.

## Errors

- `procedural_squash`, `recenter_to_anchor`, `build_frame2` require a `P`-mode
  `frame1`; if `frame1.mode != "P"`, raise `ValueError` mentioning
  "palette-mode" — consistent with `quantize_to_reference` and `generate_shiny`.
- `difference_ratio` requires `a.size == b.size`; on mismatch raise
  `ValueError` (comparing different-size images is undefined here). In the
  `build_frame2` flow both are always 96×96, so this only guards misuse.
- No new external dependencies, network calls, file I/O, or `import torch` /
  `from diffusers import …` anywhere in these helpers — they must import only
  from `PIL` (and stdlib), so the light suite stays green without torch.

## Constraints & dependencies

- Pure PIL + stdlib only. Allowed PIL surface: `Image` (new/copy/resize/paste/
  crop/point/putpalette/getpalette/getcolors/get_flattened_data/getbbox) and
  optionally `ImageChops`. No torch, diffusers, transformers, compel, or numpy.
- Must live in `fakemon_forge/sprites.py` alongside the existing pure-PIL
  helpers (`postprocess`, `quantize_to_reference`, `generate_shiny`).
- Reuse `quantize_to_reference` for palette locking; reuse `_SPRITE_SIZE` /
  `_PALETTE_COLORS` module constants rather than hard-coding 96 / 16.
- Non-mutating: inputs unchanged after any call.
- Torch-free testability: tests go in `tests/test_sprites.py` (the regular,
  torch-free file — **not** `test_sprites_ml.py` and **no** `@pytest.mark.ml`),
  since none of these helpers touch ML code. Build `P`-mode fixtures via
  `postprocess` / `quantize_to_reference` exactly as the existing tests do.

## Tests (to add in `tests/test_sprites.py`)

Light, pure-PIL. Fixtures built with existing `postprocess` /
`quantize_to_reference` helpers (and the existing `_rgb_image` / `_noisy_image`
factories). Cover:

- `procedural_squash`: output is 96×96, mode `P`, `getpalette()` equals frame
  1's, differs from frame 1 but by a small amount inside the acceptance band
  (`low <= difference_ratio(out, frame1) <= high`).
- `difference_ratio(x, x) == 0.0`; two clearly different images give a ratio
  near the expected bound (e.g. a solid vs an inverted/other-colour image → high).
- `build_frame2(frame1)` with no candidate returns the squash (differs from
  frame 1, shares palette).
- `build_frame2` with a near-identical candidate (ratio `< low`) falls back to
  squash.
- `build_frame2` with a wildly-different candidate (ratio `> high`) falls back
  to squash.
- `build_frame2` with an in-band candidate returns the palette-locked,
  recentred candidate, sharing frame 1's palette.
- `recenter_to_anchor` aligns a shifted candidate's non-background bbox to
  frame 1's (compare bbox anchors before/after, or assert the recentred bbox
  matches frame 1's bottom-centre).
- (Recommended) non-mutation of inputs, mirroring existing
  `*_does_not_mutate_*` tests.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** Acceptance band defaults `low=0.02`, `high=0.30` and squash
  `amount_px=2` are eyeball placeholders (the issue notes "the img2img strength
  dial has no principled value"). They are exposed as parameters so the later
  ML slice or a human can tune them, and code comments must state they are
  tunable.
- **[picked]** Background = the **most common palette index** of frame 1.
- **[picked]** Anchor = **bottom-centre** of the non-background bbox
  (feet-planted breathing look).
- **[picked]** Palette locking for `procedural_squash` is done by operating in
  P-space + `putpalette(frame1.getpalette())` (exact, no re-quantize), with
  `quantize_to_reference(result, frame1)` as the documented fallback. Both
  satisfy the `getpalette()`-equality contract; the issue permits either.
- **[picked]** `procedural_squash` pastes the squashed content at `(0,
  amount_px)` (bottom-anchored), leaving the top `amount_px` rows as
  background.
- **[picked]** `recenter_to_anchor` uses frame 1's background index for both
  images (the candidate has been palette-locked to frame 1, so they share the
  index space). Standalone callers are expected to pass a candidate already
  sharing frame 1's palette.
- **[picked]** `recenter_to_anchor` on an all-background image (empty bbox)
  returns the candidate palette-locked but untranslated rather than raising.
- **[picked]** `difference_ratio` raises `ValueError` on size mismatch (rather
  than resizing or returning `1.0`).
- **[picked]** The background-index helper is a private function
  (`_`-prefixed); exact name left to the implementer.
- **[picked]** `difference_ratio` compares palette **indices** for `P`-mode
  inputs (valid because comparisons in this feature are always between images
  sharing one palette). An implementation may instead diff via RGB/`ImageChops`
  as long as `difference_ratio(x, x) == 0.0` holds.
- **[confirmed]** 96×96 size, 16-colour palette, `P`-mode, and the
  `.getpalette()`-equality definition of "shares palette" come from
  `_SPRITE_SIZE` / `_PALETTE_COLORS`, `quantize_to_reference`, and existing
  tests.
- **[confirmed]** `ValueError` with a "palette-mode" message for non-`P` inputs
  matches `quantize_to_reference` / `generate_shiny`.
- **[confirmed]** Tests belong in the torch-free `tests/test_sprites.py`; these
  helpers touch no ML code, so no `@pytest.mark.ml` and no torch import.
- **[confirmed]** `get_flattened_data()` exists on Pillow 12.3 (the pinned
  version) and returns palette indices; used by existing tests.
