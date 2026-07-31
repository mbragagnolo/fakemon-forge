# Spec: `_quantize_gen3` — creature-region quantize (≤13) + deterministic Gen-3 palette assembly

## Summary

`fakemon-forge` renders 96×96 Gen-3-style sprites; sprite post-processing lives
in `fakemon_forge/sprites.py`. We are moving generated sprites onto an authentic
Gen-3 **palette contract**:

```
index 0 : transparency key (200, 200, 168)   <- always, exactly
black   (0, 0, 0)                             <- always reserved
white   (255, 255, 255)                       <- always reserved
creature: <= 13 colours
```

White and black are **always reserved** (a policy decision matching real Gen-3
outline/highlight practice) whether or not the creature uses them, so the total
palette is **≤ 16** (3 reserved + ≤ 13 creature) with the key at **index 0**.

A prior, already-merged slice added the constant `_KEY_COLOR = (200, 200, 168)`
and the pure helper `_flatten_background_to_key(rgb_image) -> rgb_image` (flood
fill from borders + global near-background sweep) to `sprites.py`. This slice
(2/4 of #11) adds the **next standalone building block**: a pure helper
`_quantize_gen3(image) -> P-mode image` that takes an **RGB** image and returns a
96×96 **P-mode** image obeying the contract above.

This slice **only adds the helper (and two constants)**. It does **not** wire it
into `postprocess` / `quantize_to_reference` / `generate_shiny` / `main` — a
later slice does that. Those functions are left byte-for-byte untouched, so all
existing tests keep passing unchanged.

### Explicitly out of scope

- **Wiring into the pipeline.** `postprocess`, `quantize_to_reference`,
  `generate_shiny`, `generate_sprite*`, and `main` are **not** modified. A later
  slice (3–4/4 of #11) rebuilds the contract on top of `_quantize_gen3`.
- **Changing `_flatten_background_to_key`.** It is reused as-is; its
  border-detection / tolerance / gradient-fallback behaviour is not touched.
- **`.ini` / writer / CLI changes.** None; this is a pure-PIL helper only.
- **Frame-2 / back-sprite / shiny palette sharing.** Unrelated to this slice.

## Inputs

### New: `_quantize_gen3(image: Image.Image) -> Image.Image`

- `image` — an **RGB** `PIL.Image.Image` of any size. Its background **may or may
  not** already be flattened to `_KEY_COLOR` (the helper flattens idempotently).
  Typically the raw SD output (e.g. 768×768) or an already-96×96 sprite.
  - **[picked]** The helper **assumes RGB** and does **not** convert or validate
    the mode, mirroring `_flatten_background_to_key`'s documented "Assumes RGB
    input" contract. Non-RGB input is undefined behaviour (may raise inside the
    per-pixel/flatten logic). See Assumptions.
- The input is **not mutated** (all steps return fresh images / operate on
  copies).

### New module constants (in `sprites.py`)

- `_MAX_CREATURE_COLORS = 13` — the creature colour budget (excludes the 3
  reserved slots). **[confirmed by issue]**
- `_KEY_COLLISION_DISTANCE = 12` — **[picked]** a small tunable Euclidean-RGB
  distance. Any creature palette colour within this distance of `_KEY_COLOR` is
  nudged away so it can never be mistaken for the transparency background. Kept
  distinct from (and smaller than) the existing `_KEY_TOLERANCE = 30`, which
  governs background *detection*, not creature/key *collision*. Treated as a
  tunable eyeball placeholder in the style of the existing
  `_KEY_TOLERANCE` / `amount_px` / `low`/`high` constants.

## Outputs

- A fresh **`P`-mode**, **96×96** `PIL.Image.Image` such that:
  - `getpalette()[0:3] == [200, 200, 168]` — the key at **index 0**, exactly.
  - `(0, 0, 0)` (black) and `(255, 255, 255)` (white) are present at fixed
    reserved slots — **[picked]** black at index 1, white at index 2 (the exact
    reserved order is an internal detail as long as key is index 0 and both black
    and white are always present, per the issue).
  - The creature colours occupy the slots after the reserved three (indices 3…),
    numbering **≤ 13 distinct** colours actually used by non-background pixels.
  - **Total distinct colours actually used ≤ 16.**
  - **Every** pixel that is the key colour (the flattened background) decodes to
    palette **index 0**.
  - **No** non-index-0 pixel decodes to an RGB within `_KEY_COLLISION_DISTANCE`
    of `_KEY_COLOR` (the collision-nudge guarantee).

## Behavior

`_quantize_gen3` is deterministic pure PIL (no randomness, no time, no I/O). The
steps, in order:

1. **Resize + enhance (parity with `postprocess`).** On a copy / fresh result:
   `image.resize((96, 96), Image.NEAREST)` →
   `ImageEnhance.Color(...).enhance(1.1)` →
   `ImageEnhance.Contrast(...).enhance(1.1)`. This mirrors the exact pre-steps in
   `postprocess` and `quantize_to_reference`, so `_quantize_gen3`'s creature
   region is comparable to what the current pipeline quantizes.
   - **Ordering note (important):** enhance runs **before** flattening. Applying
     `ImageEnhance.Color`/`Contrast` *after* flattening would shift the key
     `(200,200,168)` off its exact value (Color 1.1 alone drives it to
     ≈`(200,200,165)`), breaking the "index 0 is exactly the key / every bg pixel
     maps to index 0" contract. So: resize → enhance → **flatten** → quantize.
2. **Flatten the background to the key.** Call
   `_flatten_background_to_key(enhanced_rgb)`. Idempotent if already flattened.
   The result is an RGB 96×96 image whose background pixels equal `_KEY_COLOR`
   **exactly**, creature pixels untouched.
3. **Isolate the creature region.** Treat every pixel **not equal to**
   `_KEY_COLOR` as creature; every pixel equal to `_KEY_COLOR` as background. The
   background is **excluded** from the colour count so the entire 13-colour budget
   is spent on the creature (mask out key pixels, quantize the remainder).
4. **Quantize the creature region alone to ≤ 13 colours.** Adaptively quantize
   only the creature pixels (e.g. gather the non-key pixel values into a scratch
   image and `.quantize(colors=_MAX_CREATURE_COLORS)`), yielding up to 13 creature
   RGB colours. Deterministic (median-cut on identical input is stable).
   - **All-background input** (no creature pixels) → **0** creature colours; not
     an error (see Edge cases).
5. **Nudge creature colours off the key.** For each of the ≤ 13 creature colours,
   if `_rgb_distance(colour, _KEY_COLOR) <= _KEY_COLLISION_DISTANCE`, push it
   **away from** the key until it is strictly beyond `_KEY_COLLISION_DISTANCE`
   (move along the key→colour direction; on the degenerate `colour == key` case
   use a fixed fallback direction), clamping channels to `[0, 255]`. Reuses the
   existing `_rgb_distance` helper. This guarantees no creature colour can be
   confused with the transparency background.
6. **Assemble the final palette deterministically.** Build the ordered RGB list
   `[_KEY_COLOR, (0,0,0), (255,255,255), *creature_colours]` — key at index 0,
   then the always-present reserved black and white, then the ≤ 13 (nudged)
   creature colours. Reserved black/white occupy their fixed slots **even if the
   creature uses neither**. Pad the remaining slots up to 256 as needed to form a
   valid palette.
7. **Map every pixel to the assembled palette.** Produce the final `P`-mode
   96×96 image so that background (key) pixels → index 0 and creature pixels →
   their nearest creature/reserved palette index (e.g. quantize the flattened RGB
   against a `P`-mode reference carrying the assembled palette, dither off — the
   same nearest-colour mapping `quantize_to_reference` already relies on). Because
   the background is exactly `_KEY_COLOR` and `_KEY_COLOR` is index 0 with no
   duplicate, every background pixel resolves to index 0. Creature pixels that sit
   near pure black / white legitimately snap to the reserved slots (authentic
   Gen-3 outline/highlight behaviour) and do **not** count against the 13-colour
   creature budget.

The result is a new image; the input `image` object is never modified.

## Edge cases

- **Already-flattened input** → `_flatten_background_to_key` is idempotent;
  background pixels are re-detected as the key and left as the key. Same result.
- **All-background image** (every pixel keys out) → 0 creature colours; output is
  all index 0, palette still `[key, black, white, …]` with black/white reserved.
  No crash, no empty-quantize error (the ≤13 path must tolerate an empty creature
  pixel set).
- **Creature with < 13 distinct colours** → uses fewer than 13 creature slots;
  contract (≤13, ≤16 total) still holds.
- **Creature with ≫ 13 colours** → quantized down to exactly ≤ 13 creature
  colours.
- **Creature that uses neither pure black nor white** → black and white are
  **still present** in the palette at their reserved slots (reserved-slot
  guarantee); they simply go unused by pixels.
- **Creature colour coincidentally near the key** (e.g. a khaki/tan creature that
  survived flattening because it was far from the *detected border* background) →
  nudged in step 5 so no creature pixel decodes within `_KEY_COLLISION_DISTANCE`
  of the key.
- **Non-96×96 input** → resized to 96×96 (NEAREST) in step 1 before anything
  else, exactly as `postprocess` does.
- **Gradient/vignette border** → handled entirely inside
  `_flatten_background_to_key` (dominant-colour keying + stderr warning); this
  helper adds no new behaviour for it.

## Errors

- `_quantize_gen3` raises no new exceptions of its own for valid RGB input.
- It does **not** validate the input mode (per the RGB assumption); a non-RGB
  input is undefined behaviour and may raise from within
  `_flatten_background_to_key` / the per-pixel logic. This matches
  `_flatten_background_to_key`, which likewise assumes RGB without validating.
- No `sys.exit`, no stderr output beyond whatever `_flatten_background_to_key`
  may emit (its existing gradient-border warning).

## Constraints & dependencies

- Change is confined to `fakemon_forge/sprites.py`: add `_MAX_CREATURE_COLORS`,
  `_KEY_COLLISION_DISTANCE`, and `_quantize_gen3`. Reuse the existing
  `_KEY_COLOR`, `_flatten_background_to_key`, `_rgb_distance`, and `_SPRITE_SIZE`;
  do not duplicate the resize/enhance recipe conceptually beyond matching it.
- **Determinism:** no `random`, no time, no network/disk — pure PIL median-cut
  and fixed arithmetic, so identical input yields identical output.
- **Torch-free:** `_quantize_gen3` triggers **no** `import torch` /
  `from diffusers import …`. Per `CLAUDE.md`'s test-slicing rule its tests belong
  in `tests/test_sprites.py` (the torch-free file), **not** `test_sprites_ml.py`.
  Expect the usual ~21 `ml` skips in the keep sandbox — expected, do not install
  torch.
- **Untouched surfaces:** `postprocess`, `quantize_to_reference`,
  `generate_shiny`, `generate_sprite*`, `procedural_squash`,
  `recenter_to_anchor`, `build_frame2`, and `main` are **not** modified, so their
  existing tests remain green without change.

## Tests (light — `tests/test_sprites.py`, torch-free)

Add a `_quantize_gen3` test group following the existing pure-PIL patterns
(`_rgb_image`, `_noisy_image`, `_sprite_rgb`, `_noisy_border_sprite`,
`_ring_sprite`, `_KEY_COLOR`). Import `_quantize_gen3` (and, where handy,
`_MAX_CREATURE_COLORS` / `_KEY_COLLISION_DISTANCE`). Suggested cases:

- **Mode & size:** output is `P`-mode and `(96, 96)`.
- **Key at index 0:** `getpalette()[0:3] == [200, 200, 168]`.
- **Reserved black & white present:** the palette contains `(0, 0, 0)` and
  `(255, 255, 255)` at their reserved positions (`getpalette()[3:6]` /
  `[6:9]`).
- **Creature-colour budget:** the number of **distinct creature colours** —
  palette entries actually used by non-background pixels, excluding
  key/black/white — is **≤ 13**, and **total distinct colours used ≤ 16**.
  (Use a noisy / multi-colour creature so the cap is exercised.)
- **Background → index 0:** every pixel that was the key colour resolves to
  palette index 0 (e.g. feed a `_noisy_border_sprite()` / `_ring_sprite()` whose
  background flattens to the key, then assert border/pocket pixels decode to
  index 0).
- **Reserved-slot guarantee:** for a creature that uses neither black nor white
  (e.g. a mid-tone body only), black and white are **still** in the palette.
- **Collision nudge:** for a creature whose body colour is deliberately near the
  key (within `_KEY_COLLISION_DISTANCE` of `(200,200,168)`), assert **no**
  non-index-0 pixel decodes to within `_KEY_COLLISION_DISTANCE` of `_KEY_COLOR`.
- **No mutation:** the input image's size and flattened pixel data are unchanged
  after the call (mirror `test_flatten_does_not_mutate_input`).

Run `pytest` from the repo root; the light suite runs in the keep sandbox
without torch.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase or the issue.

- **[picked]** Helper name `_quantize_gen3`, signature `(image: Image.Image) ->
  Image.Image`, RGB → 96×96 `P`. Chosen so a later slice can build the same
  contract from it inside both `postprocess` (adaptive) and
  `quantize_to_reference` (fixed) paths.
- **[picked]** Reserved order is key = index 0, black = index 1, white = index 2,
  creature = indices 3+. The issue states the exact reserved order is an internal
  detail provided the key is index 0 and black + white are always present.
- **[picked]** `_KEY_COLLISION_DISTANCE = 12`, a small tunable Euclidean-RGB
  distance, kept separate from and smaller than the existing detection
  `_KEY_TOLERANCE = 30`. The nudge pushes an offending creature colour **away
  from** the key just past this distance (fixed fallback direction if the colour
  equals the key exactly), clamped to `[0, 255]`.
- **[picked]** Enhance/resize factors mirror `postprocess`/`quantize_to_reference`
  exactly: resize to 96×96 NEAREST, `ImageEnhance.Color(1.1)`,
  `ImageEnhance.Contrast(1.1)`, applied **before** flattening (so the key stays
  byte-exact) and before creature quantization.
- **[picked]** The helper **assumes RGB** input and does not convert/validate the
  mode, matching `_flatten_background_to_key`'s documented contract. A later
  wiring slice is responsible for feeding it RGB.
- **[picked]** Creature-region quantization excludes key-coloured pixels from the
  colour count by quantizing the non-key pixels alone (rather than quantizing the
  whole image and hoping the background collapses to one slot), guaranteeing the
  full 13-colour budget goes to the creature.
- **[picked]** Final pixel mapping uses PIL nearest-colour mapping against a
  `P`-mode reference carrying the assembled palette (the same mechanism
  `quantize_to_reference` uses), dither disabled for determinism; unused palette
  slots are padded to form a valid 256-entry palette. Creature pixels near pure
  black/white legitimately snap to the reserved slots and do not count against
  the creature budget.
- **[confirmed]** `_MAX_CREATURE_COLORS = 13`, total palette ≤ 16 (3 reserved +
  ≤ 13 creature), key at index 0 — directly from the issue's palette contract.
- **[confirmed]** `_KEY_COLOR = (200, 200, 168)`, `_flatten_background_to_key`,
  and `_rgb_distance` already exist in `sprites.py` and are reused unchanged.
- **[confirmed]** The helper is torch-free, so its tests live in
  `tests/test_sprites.py`; the keep sandbox reports ~21 `ml` skips, which is
  expected and correct (do not install torch).
- **[confirmed]** This slice only adds `_quantize_gen3` (+ constants);
  `postprocess`, `quantize_to_reference`, `generate_shiny`, and `main` remain
  untouched, so their existing tests stay green.
