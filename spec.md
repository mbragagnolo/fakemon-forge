# Spec: Pin key/white/black in shiny + align reference-lock background to index 0

## Summary

`fakemon-forge` renders each stage as **six views**: front frame 1, front
frame 2, back, and a shiny of each. Prior slices gave the front sprite an
authentic Gen-3 palette contract, produced by `_quantize_gen3` /
`postprocess` in `fakemon_forge/sprites.py`:

```
index 0 : transparency key (200, 200, 168)   ← _KEY_COLOR
index 1 : black (0, 0, 0)                     ← always reserved
index 2 : white (255, 255, 255)               ← always reserved
indices 3.. : <= 13 creature colours          ← _MAX_CREATURE_COLORS
```

Two functions still violate that contract, so the non-front views drift off it:

1. **`quantize_to_reference(image, reference)`** locks frame 2 and the back
   sprite to frame 1's palette by nearest-colour mapping (`quantize(palette=
   reference)`). But frame 2 / back come off SD with **noisy near-white
   backgrounds**; nearest-colour maps those near-white pixels to the reserved
   **white** slot `(255,255,255)` instead of the **key at index 0**. That
   punches the background into a creature-white slot and breaks transparency
   keying, because keying is done by palette **index 0**, not by colour.
2. **`generate_shiny(sprite_path, name, output_path)`** builds a shiny palette
   by hue-rotating **chromatic** entries and preserving **achromatic** ones
   (`_is_achromatic`: luminance `< 40` or `> 215`). The key `(200,200,168)` has
   luminance ≈ 196.4 → **chromatic** (a desaturated green), so it is currently
   hue-rotated — changing the transparency key on shinies. White is achromatic
   (preserved today) and black is achromatic too, but per the contract the key,
   white, and black must be pinned **explicitly**, never left to a luminance
   threshold that could reclassify them.

This slice makes all six views honour the contract, with no change required
outside `sprites.py` (and its torch-free tests). `main.py` already threads
`reference_path=sprite.png` for the back sprite and calls `generate_shiny` per
view — that wiring is confirmed correct and stays untouched.

### Explicitly out of scope

- No change to `postprocess` / `_quantize_gen3` (they already produce the
  contract correctly) or to any constant (`_KEY_COLOR`, `_MAX_CREATURE_COLORS`,
  tolerances).
- No change to `main.py`, `build_frame2`, `procedural_squash`,
  `recenter_to_anchor`, `generate_sprite_img2img`, or any ML/pipeline code.
- No new public functions, parameters, or signatures. Both edited functions
  keep their exact current signatures.
- Not touching the achromatic hue-rotation heuristic itself for *creature*
  entries — only ensuring key/white/black are pinned regardless of it.

## Inputs

### `quantize_to_reference(image, reference)`
- `image`: a PIL image (the raw img2img candidate for frame 2 / back). In
  practice RGB, but any mode PIL can `resize`/enhance/`quantize`; near-white
  noisy background is the case that matters.
- `reference`: a PIL **P-mode** image whose palette is the shared Gen-3-contract
  palette (frame 1's), i.e. `_KEY_COLOR` at index 0, black at index 1, white at
  index 2, then creature colours.

### `generate_shiny(sprite_path, name, output_path)`
- `sprite_path`: filesystem path to an existing **P-mode** sprite PNG whose
  palette obeys the contract (index 0 = `_KEY_COLOR`; a `(0,0,0)` and a
  `(255,255,255)` entry reserved).
- `name`: the Pokémon name; seeds a deterministic per-name hue shift.
- `output_path`: filesystem path to write the shiny PNG.

## Outputs

### `quantize_to_reference`
Returns a **new** P-mode, 96×96 image sharing `reference`'s palette **exactly**
(`out.getpalette() == reference.getpalette()`). Every background pixel of the
input resolves to **index 0** (the key), not to the white slot. Reserved
black/white slots remain in place (they are already in the reference palette and
nearest-colour mapping keeps them). Neither `image` nor `reference` is mutated.

### `generate_shiny`
Writes a P-mode shiny PNG to `output_path` whose palette is the input palette
with **creature** entries hue-rotated by the name-seeded shift and the
**key (index 0), white `(255,255,255)`, and black `(0,0,0)` entries copied
through unchanged**. Pixel index data is unchanged (only the palette is
rewritten), so the shiny is the same shape as the normal. Returns `None`.

## Behavior

### `quantize_to_reference` (background alignment)
1. Reject a non-P `reference` with the existing `ValueError` (unchanged; see
   Errors).
2. Resize `image` to 96×96 with `Image.NEAREST` and apply the same
   `ImageEnhance.Color(1.1)` then `ImageEnhance.Contrast(1.1)` pre-steps as
   today, matching `postprocess` / `_quantize_gen3` so either path feeds the
   same input to quantization. (Enhance runs **before** flattening so it can't
   shift the key off its byte-exact value — mirroring `_quantize_gen3`.)
3. **New step:** flatten the (enhanced, RGB) background to the key with the
   existing `_flatten_background_to_key`, so every background pixel becomes
   exactly `_KEY_COLOR` `(200,200,168)`. `_flatten_background_to_key` assumes
   RGB input, so ensure the enhanced image is RGB before the call (it already
   is on the real img2img path; convert defensively if needed).
4. `quantize(palette=reference, ...)` against the reference. Now every
   background pixel is at distance 0 from the reference's index-0 key entry, so
   it nearest-maps to **index 0**, not white. Creature pixels near pure
   white/black still snap to the reserved slots as before.
5. Return the resulting P-mode 96×96 image; do not mutate inputs (operate on
   copies, as the current code does via `resize` returning a new image).

Reuse the front-sprite helper `_flatten_background_to_key` rather than a
bespoke border-map, so front, frame 2, and back all key the background
identically (same detection, same tolerance, same gradient-border fallback and
stderr warning).

### `generate_shiny` (pin key/white/black)
1. Open `sprite_path`; reject non-P with the existing `ValueError` (unchanged).
2. Compute the name-seeded `hue_shift` exactly as today
   (`md5(name) % 300 + 30 → /360`), so cross-view consistency and determinism
   are preserved.
3. Walk the palette in `(r,g,b)` triples. For each entry, **pin** it (copy
   `[r,g,b]` through unchanged) when **any** of:
   - it is **index 0** (the transparency key `_KEY_COLOR`), OR
   - `(r,g,b) == (255,255,255)` (white), OR
   - `(r,g,b) == (0,0,0)` (black).
   Otherwise, hue-rotate it by `hue_shift` in HSV (as today). This pinning is
   by explicit match, **independent of `_is_achromatic`**, so the chromatic key
   is provably never rotated. The `_is_achromatic` check may remain to also
   preserve other very-bright/very-dark creature entries, or be superseded —
   either is acceptable as long as key/white/black are provably pinned.
4. `putpalette` the new palette onto a copy and save to `output_path`.

Note the key is pinned by **index 0** specifically (not by colour match against
`_KEY_COLOR`), matching the contract that index 0 is the key slot; white/black
are pinned by exact colour match since their reserved indices (1, 2) are also
their colours.

## Edge cases

- **Noisy near-white background** (the motivating case): after
  `_flatten_background_to_key` every background pixel is exactly `_KEY_COLOR`
  and maps to index 0. This is the behaviour the new test asserts.
- **Gradient/vignette border in the candidate:** `_flatten_background_to_key`
  already handles this by keying only the dominant border colour and emitting a
  stderr warning without raising; `quantize_to_reference` inherits that
  behaviour for free.
- **All-background candidate:** flattens fully to the key → every pixel maps to
  index 0; output palette is still the reference's.
- **Creature colour legitimately near white/black:** still snaps to the
  reserved white/black slots via nearest-colour, unchanged.
- **Shiny where a creature entry happens to equal `_KEY_COLOR` at an index
  other than 0:** only index 0 is pinned as the key; a creature entry that
  coincidentally equals the key colour at another index is treated as a
  creature entry (rotated). Under the contract `_nudge_off_key` keeps creature
  colours off the key, so this is not expected to arise; the spec pins by
  index 0 for the key deliberately.
- **Shiny palette shorter/longer than 256 or with unused tail entries:** iterate
  over whatever `getpalette()` returns (as today, `[R,G,B]×N`); pinning/rotation
  applies per triple. Unused entries are harmless.
- **Cross-view consistency:** three views sharing one contract palette, run
  through `generate_shiny` with the same `name`, must yield three **identical**
  rotated palettes whose index 0 / white / black equal the normals'. Preserved
  because the shift is name-seeded and the pinned slots are identical across
  views.

## Errors

- `quantize_to_reference`: `ValueError(f"Expected palette-mode reference image,
  got {reference.mode}")` when `reference.mode != "P"` — unchanged, still
  matches `"palette-mode"`.
- `generate_shiny`: `ValueError(f"Expected palette-mode image, got {img.mode}")`
  when the opened image is not P — unchanged.
- No new error conditions introduced. `_flatten_background_to_key` never raises
  (it warns and returns best-effort on non-uniform borders).

## Constraints & dependencies

- Pure **PIL** (`Image`, `ImageEnhance`, `ImageDraw` via the helper) and
  stdlib (`colorsys`, `hashlib`). No torch/diffusers touched, so tests live in
  `tests/test_sprites.py` and run in the keep sandbox (pytest + Pillow, no
  torch). Per `CLAUDE.md`, only real-`import torch` code goes in
  `test_sprites_ml.py`; these functions qualify for the torch-free file.
- Keep the existing invariants both functions already carry: P-mode 96×96
  output sharing the reference palette exactly (`quantize_to_reference`);
  P-mode requirement and no-mutation for both; deterministic name-seeded shift
  (`generate_shiny`).
- Output must remain deterministic (median-cut/nearest arithmetic; no RNG).
- Existing passing tests must stay green, notably:
  `test_quantize_to_reference_*` (palette reuse, 96×96, ≤16 colours, no
  mutation, non-P rejection), the `build_frame2` acceptance-band tests (which
  depend on `quantize_to_reference` behaviour — see risk note), and
  `test_cross_view_shinies_share_one_rotated_palette`.

### Risk note — `build_frame2` acceptance-band tests

`build_frame2` calls `quantize_to_reference(candidate, frame1)` and accepts the
candidate only when its diff from frame 1 is in `[0.02, 0.30]`. Several existing
tests craft candidates around the *old* behaviour (background **not** flattened,
so a dark backdrop maps to reserved black and a key backdrop maps to index 0):

- `test_build_frame2_near_identical_candidate_falls_back` — relies on
  `_sprite_rgb`'s dark `(40,40,60)` backdrop mapping to reserved black so the
  whole backdrop differs from frame 1's key background → ratio above `high` →
  squash fallback. After this change the background is flattened to the key
  first; the dark backdrop is the **border** and will be detected/keyed by
  `_flatten_background_to_key`, mapping it to index 0 like frame 1 — the
  backdrop would then **match** frame 1, lowering the ratio.
- `test_build_frame2_in_band_candidate_is_accepted` and the
  `_key_background_sprite` helper deliberately swap the dark backdrop for the
  key precisely because "`quantize_to_reference` does not flatten a candidate's
  background to the key" — an assumption this slice **inverts**.

The implementation must re-verify these `build_frame2` tests after the change
and, if the flatten step shifts their diff ratios out of the expected band,
update those tests (and/or the `_key_background_sprite` helper docstring) to
reflect the new, correct behaviour — flattening the candidate background to the
key is the intended outcome. Adjusting these tests is in scope for this slice;
changing `build_frame2`'s logic or thresholds is **not**. This is called out so
the implementer treats a red `build_frame2` test as an expected consequence to
reconcile, not a regression to avoid.

## Tests (torch-free, `tests/test_sprites.py`)

Add:
1. **Reference-lock background→index 0.** Build a reference via
   `postprocess(...)` (key at index 0). Feed `quantize_to_reference` an RGB
   candidate with a **noisy near-white background** and a distinct creature
   (e.g. `_noisy_border_sprite()` upscaled, or a new helper). Assert every
   background/border pixel of the output is index **0** (not the white slot's
   index), and `out.getpalette() == reference.getpalette()`.
2. **Reserved slots preserved through the lock.** The locked output's palette
   still has `_KEY_COLOR` at index 0, `(0,0,0)` and `(255,255,255)` at their
   reserved positions.
3. **Shiny pins the key.** Take a P-mode sprite whose index 0 is `_KEY_COLOR`,
   run `generate_shiny`; assert the output palette's index 0 is still exactly
   `[200,200,168]` (the chromatic key is NOT rotated).
4. **Shiny pins white and black.** Assert `(255,255,255)` and `(0,0,0)` entries
   are unchanged in the shiny palette, while at least one creature (mid-tone)
   entry **did** change (rotation still happens).
5. Keep `test_cross_view_shinies_share_one_rotated_palette` passing (three views
   → three identical rotated shiny palettes; index 0 / white / black equal the
   normals').

Run `pytest` from the repo root (flat package). ~21 `ml` tests skip in the
sandbox without torch — expected and correct; do **not** install torch.

## Done looks like

Across all six views of a stage: the background is uniformly `(200,200,168)` at
palette index 0; frame 2 / back lock to frame 1's palette with their backgrounds
mapped to index 0 (not white); and shiny hue rotation never alters the key,
white, or black. Full `pytest` green (with the usual sandbox `ml` skips).
`main.py` is confirmed already correct (`reference_path=sprite.png` for the back
sprite, `generate_shiny` per view) and left untouched.

## Assumptions

- **[picked]** `quantize_to_reference` gains the `_flatten_background_to_key`
  pre-step (reusing the front-sprite helper) rather than a bespoke border-map,
  so front, frame 2, and back all key the background identically.
- **[picked]** The flatten step runs **after** resize+enhance and **before**
  `quantize`, mirroring `_quantize_gen3`'s ordering (enhance before flatten so
  the key stays byte-exact).
- **[picked]** `generate_shiny` pins **index 0** plus any `(255,255,255)` /
  `(0,0,0)` entry by exact match, independent of `_is_achromatic`. The key is
  pinned by index (0), white/black by colour. The `_is_achromatic` check may
  remain for other bright/dark creature entries or be superseded — either is
  acceptable as long as key/white/black are provably never rotated.
- **[picked]** `_flatten_background_to_key` requires RGB; the spec assumes the
  enhanced image is RGB on the real path and that a defensive `convert("RGB")`
  (only if not already RGB) is acceptable to satisfy that precondition without
  changing behaviour on RGB inputs.
- **[picked]** The pre-existing `build_frame2` tests that encode the *old*
  no-flatten assumption (`test_build_frame2_near_identical_candidate_falls_back`,
  `test_build_frame2_in_band_candidate_is_accepted`, and the
  `_key_background_sprite` helper) may need updating to the new flattened
  behaviour; that is in scope, while `build_frame2`'s own logic/thresholds stay
  unchanged.
- **[confirmed by code]** `main.py` already threads `reference_path=sprite_path`
  into the back sprite's `generate_sprite_img2img` and calls `generate_shiny`
  for front, frame 2, and back — no `main.py` change needed.
- **[confirmed by code/CLAUDE.md]** Both functions are pure PIL, so their tests
  belong in `tests/test_sprites.py` and run under the torch-free sandbox.
