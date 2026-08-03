# Spec: Content-aware front/back split for side-by-side canvases

## Summary

Issue #61 will retool sprite generation onto a backend that renders the front
and back sprite **side by side in one wide canvas** (front on the left half,
back on the right half) in a single generation call. This slice (3/9) adds
only the pure-PIL logic that cuts such a canvas into a front half and a back
half — no pipeline wiring, no torch/diffusers, unit-testable against synthetic
images.

A new function is added to `fakemon_forge/sprites.py`, alongside the module's
other small pure-PIL helpers (`_border_ring`, `_content_bbox`,
`_background_index`): given a side-by-side RGB canvas, it searches the middle
20% of columns for the widest run of columns that are background for their
full height, and either returns the canvas cut at that run's centre column, or
a clearly distinguishable "no band found" result if no such run exists in that
window.

Explicitly out of scope (left for the later slice that wires this into the new
backend, per the issue):
- Rerolling generation with a new seed when no band is found.
- The naive-midline-cut-plus-stderr-warning fallback.
- Any change to `main.py` or `generate_sprite*`/pipeline code.

**Done when:** the split function is implemented in `sprites.py`, covered by
the three test scenarios below in `tests/test_sprites.py`, introduces no
torch/diffusers import, and is not called from `main.py` or any
`generate_sprite*` function.

## Inputs

### `split_front_back_canvas(canvas: Image.Image) -> tuple[Image.Image, Image.Image] | None`

- `canvas` — an RGB `Image.Image` containing a front sprite on the left half
  and a back sprite on the right half, on a flat background. Assumed to already
  be RGB (mirroring `_flatten_background_to_key` / `_quantize_gen3`'s
  convention elsewhere in this module) — not converted or validated.
- No other parameters. Tuning knobs (search window fraction, background
  tolerance) are module constants, not arguments, matching how
  `_KEY_TOLERANCE` / `_BORDER_UNIFORM_FRACTION` are used elsewhere.

### New module constants (near the existing tunables, e.g. `_KEY_TOLERANCE`)

- `_SPLIT_SEARCH_LOW = 0.4`, `_SPLIT_SEARCH_HIGH = 0.6` — the middle-20%
  column window (as fractions of canvas width) searched for a full-height
  background run, per the issue's spec. Documented as a "tunable eyeball
  placeholder" in the module's existing style.

No new dependency on `_KEY_TOLERANCE`'s value, `_rgb_distance`, or
`_border_ring` — all three are reused as-is.

## Outputs

- **Band found:** `(front_half, back_half)` — a 2-tuple of fresh RGB
  `Image.Image` crops of `canvas`. `front_half` is the region left of the cut
  column, `back_half` is the region at/right of it; both share `canvas`'s
  height. `canvas` itself is not mutated (crops return new images).
- **No band found:** `None`. This is a return-value branch, not an exception —
  callers must be able to distinguish the two cases with a plain `is None`
  check, never by parsing strings or catching exceptions for control flow.

## Behavior

1. Compute the background colour `bg` as the per-channel mean of
   `_border_ring(canvas)` — byte-for-byte the same calculation
   `_flatten_background_to_key` already performs (lines computing `bg` there),
   reusing `_border_ring` directly rather than re-implementing it. This is the
   *only* piece of `_flatten_background_to_key`'s logic reused here — the flood
   fill / enclosed-pocket scan is not needed, since only the scalar `bg` colour
   is wanted.
2. Compute the search window's column bounds from `canvas.width`:
   `x_start = int(_SPLIT_SEARCH_LOW * width)`,
   `x_end = int(_SPLIT_SEARCH_HIGH * width)` (exclusive upper bound — see
   Assumptions for the exact rounding convention picked).
3. For each column `x` in `[x_start, x_end)`, classify it as a **full-height
   background column** iff every pixel `(x, y)` for `y` in `[0, height)`
   satisfies `_rgb_distance(canvas.getpixel((x, y)), bg) <= _KEY_TOLERANCE`.
4. Scan the window left to right for maximal contiguous runs of full-height
   background columns. Track the widest (most columns) run seen; on a tie
   between two runs of equal width, the first (leftmost) one encountered wins
   (see Assumptions).
5. If at least one such run exists:
   - `cut = (run_start + run_end) // 2` — the run's centre column (integer
     division; see Assumptions for the exact tie convention on even-width
     runs).
   - Return `(canvas.crop((0, 0, cut, height)), canvas.crop((cut, 0, width, height)))`.
6. If no full-height background run exists anywhere in `[x_start, x_end)`,
   return `None`.

## Edge cases

- **Clean centred gap:** a single full-height background run sits near the
  window's centre → its own centre column is the cut; each half's content
  matches the corresponding original square.
- **Off-centre gap plus a second, narrower gap:** two disjoint full-height
  background runs exist in the window → the wider one is chosen regardless of
  position, per step 4; a narrower run elsewhere in the window is ignored even
  if it is closer to the literal midline.
- **Subjects span the whole window:** no column in `[x_start, x_end)` is
  full-height background (e.g. the two silhouettes' bounding shapes overlap
  the entire middle 20%) → `None`, per step 6.
- **Tiny canvas / degenerate window:** if `int(_SPLIT_SEARCH_HIGH * width) <=
  int(_SPLIT_SEARCH_LOW * width)` the window contains zero columns → no run
  can exist → `None`.
- **Single-column run:** a run of width 1 is a valid run; its "centre column"
  is itself.
- **Entire window is background:** the widest (and only) run is the whole
  window; the cut lands at its centre, same as any other run — no special
  casing needed.
- **Non-background-tolerant noise:** a column that is background for every row
  but one (e.g. one anti-aliased pixel just outside `_KEY_TOLERANCE`) is *not*
  a full-height background column — the "every pixel" requirement in step 3 is
  strict, matching the issue's "full height" wording.

## Errors

- No new exceptions are introduced. The function assumes `canvas` is a valid
  RGB image (same convention as `_flatten_background_to_key` /
  `_quantize_gen3`, which likewise don't validate mode); passing a non-RGB
  image is out of scope, mirroring the rest of the module.
- No I/O, no calls into `pipeline`/torch/diffusers — this keeps the function
  usable from `tests/test_sprites.py` (the non-`ml` file) without triggering
  the `import torch` skip machinery described in `CLAUDE.md`.

## Constraints & dependencies

- Lives in `fakemon_forge/sprites.py`, placed near `_border_ring` /
  `_content_bbox` / `_background_index` per the task's placement guidance.
- Reuses existing module primitives (`_border_ring`, `_rgb_distance`,
  `_KEY_TOLERANCE`) rather than duplicating background-detection logic; only
  the two new search-window fraction constants are added.
- Must not import `torch` or `diffusers`, directly or transitively — pure PIL
  + stdlib only, so it belongs in `tests/test_sprites.py`, not
  `tests/test_sprites_ml.py`.
- Must **not** be called from `main.py` or any `generate_sprite*` /
  `_run_img2img` function in this slice — that wiring, plus the
  reroll-on-no-band and naive-midline-fallback behaviors, is explicitly
  deferred to the later slice that adopts the new backend.
- No change to any existing function's signature or behavior.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the current codebase.

- **[picked]** Function name `split_front_back_canvas`, returning
  `tuple[Image.Image, Image.Image] | None`. The `None`-on-not-found shape
  mirrors `_content_bbox`'s existing "bbox or `None` if all background" idiom
  in this exact module, so it's the most stylistically consistent choice for
  a caller to branch on without exceptions or string-parsing.
- **[picked]** Window bounds use `int(fraction * width)` truncation
  (`x_start` inclusive, `x_end` exclusive), matching the truncating-index style
  already used in `k_centroid` (`int(x * wf)`). No rounding to nearest, no
  clamping beyond what truncation naturally gives.
- **[picked]** Tie-break when two+ runs share the maximum width: the first
  (leftmost) one encountered wins. The issue does not specify a tie-break; the
  test suite's second scenario (widest run wins over an earlier/narrower one)
  doesn't exercise an exact-width tie, so this is an arbitrary but
  deterministic default.
- **[picked]** Cut column is the run's integer-midpoint,
  `(run_start + run_end) // 2` (floor on even-width runs). Not specified by
  the issue; floor division is the simplest deterministic choice and keeps the
  cut column always inside the run.
- **[picked/interpreted]** `front_half` and `back_half` are crops on either
  side of the single `cut` column (widths `cut` and `width - cut`), which are
  *not* necessarily each exactly `width // 2`. The issue's phrase "each half
  the canvas's width" is read as "each image holds one half of the canvas's
  content" rather than a literal equal-width mandate — forcing exact
  half-width regardless of where the detected band sits would collapse this
  content-aware cut into the naive midline cut the issue explicitly contrasts
  it with (and explicitly excludes from this slice). Flagged here because the
  issue text is genuinely ambiguous on this point.
- **[picked]** `bg` is computed once from the *whole* canvas's border ring
  (not separately per half), since both sprites come from one generation call
  sharing one flat backdrop — consistent with "using the existing
  background-colour detection convention... exactly as
  `_flatten_background_to_key` already computes it."
- **[confirmed]** `_border_ring`, `_rgb_distance`, and `_KEY_TOLERANCE` exist
  in `fakemon_forge/sprites.py` today and are safe to reuse verbatim (no
  torch/diffusers import in any of the three).
- **[confirmed]** `tests/test_sprites.py` is the correct test location: it is
  the non-`ml` file per `CLAUDE.md`, and this function never triggers
  `import torch`.
