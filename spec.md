# Spec: Wire the Gen-3 palette contract into `postprocess` and rewrite the legacy palette tests

## Summary

`fakemon-forge` generates 96×96 Gen-3-style sprites; the post-processing that
turns a raw Stable-Diffusion RGB image into a saved `P`-mode sprite lives in
`fakemon_forge/sprites.py`. Prior slices of #11 already built the new **Gen-3
palette contract** and its pure-PIL machinery:

```
index 0 : transparency key  (200, 200, 168)   — always, exactly, at index 0
          black             (0, 0, 0)          — always reserved
          white             (255, 255, 255)    — always reserved
creature: <= 13 colours
total   : <= 16 colours
```

- `_KEY_COLOR = (200, 200, 168)` and `_flatten_background_to_key(rgb) -> rgb`
  (border flood-fill + global near-background sweep, gradient-border fallback).
- `_MAX_CREATURE_COLORS = 13` and `_quantize_gen3(rgb) -> P` (resize + enhance,
  flatten background to key, quantize the creature region alone to ≤13, nudge
  creature colours off the key, assemble the deterministic palette
  `[key, black, white, ...creature]` with the key at index 0, and map with
  dither off).

Today, however, `postprocess()` still runs the **old adaptive path**:
`resize → ImageEnhance.Color(1.1) → ImageEnhance.Contrast(1.1) →
quantize(colors=16)`, producing a fresh adaptive 16-colour palette every call.

This slice (**3/4 of #11**, assuming slices 1–2 are merged) **flips
`postprocess()` to emit the Gen-3 contract** by delegating to `_quantize_gen3`,
and **rewrites the legacy tests** that encoded the old “≤16 adaptive”
expectation. This is the behaviour-changing integration for the **front sprite**
(`sprite.png`) and its animation-frame flow. It is a small, focused change: one
function body plus targeted test edits.

Explicitly **out of scope** (later slices): changing the internals of
`quantize_to_reference` or `generate_shiny`; making frame-2 / back-sprite
*backgrounds* land on index 0 (they still nearest-colour-map today).

## Inputs

- `postprocess(image: Image.Image)` — a raw RGB image from the SD pipeline
  (`generate_sprite` passes the 768×768 txt2img output; the adaptive branch of
  `generate_sprite_img2img` passes the raw img2img output; the ML-test
  `_frame1_file` fixture passes a hand-drawn 96×96 RGB). Assumed RGB (mirroring
  `_quantize_gen3` / `_flatten_background_to_key`, which do not convert mode).
- No new parameters, config, or module-level constants are introduced. All
  tunables (`_KEY_COLOR`, `_KEY_TOLERANCE`, `_MAX_CREATURE_COLORS`,
  `_KEY_COLLISION_DISTANCE`, `build_frame2`’s `low`/`high`) already exist.

## Outputs

- `postprocess` returns an `Image.Image` in **`P` mode, 96×96**, obeying the
  Gen-3 contract:
  - palette index 0 is exactly `(200, 200, 168)`;
  - palette indices 1 and 2 are exactly `(0, 0, 0)` and `(255, 255, 255)`,
    reserved even if the creature uses neither;
  - at most `_MAX_CREATURE_COLORS` (13) distinct **creature** colours are used,
    so at most 16 colours total;
  - every background (key) pixel decodes to index 0;
  - no creature colour lands within `_KEY_COLLISION_DISTANCE` of the key.
- Signature and return type are unchanged, so `generate_sprite`,
  `generate_sprite_img2img` (adaptive branch), and the `_frame1_file` fixture
  keep working with no caller edits.
- The saved front sprite `sprite.png` therefore carries the contract palette.

## Behavior

1. **`postprocess` becomes a thin wrapper over `_quantize_gen3`.** New body:
   return `_quantize_gen3(image)`. `_quantize_gen3` already performs the
   `resize(96×96, NEAREST) → Color(1.1) → Contrast(1.1)` pre-steps internally
   (before flattening, so enhancement can’t shift the key off its byte-exact
   value), so the resize/enhance lines currently in `postprocess` are **removed,
   not kept** — double-applying them is a bug to avoid. A short docstring should
   state that `postprocess` produces the Gen-3 contract and delegates to
   `_quantize_gen3` (one source of truth for the contract).
2. **Callers are unchanged.** `generate_sprite`, `generate_sprite_img2img`
   (adaptive branch), `generate_back_sprite`, and the ML `_frame1_file` fixture
   all reach `postprocess` and keep their behaviour; only the palette content of
   the produced image changes.
3. **`main.py` is untouched.** It calls `generate_sprite` /
   `generate_sprite_img2img` / `generate_frame2`, which reach `postprocess`
   internally; `tests/test_main.py` mocks the sprite functions and asserts only
   wiring (`reference_path`, init-from-front-sprite), never palette content, so
   nothing there breaks.
4. **`quantize_to_reference` and `generate_shiny` internals are not touched.**
   `quantize_to_reference` keeps copying its reference’s palette verbatim, so any
   test that builds a reference via `postprocess(...)` still gets that exact
   palette back. `frame2` / `back` continue to *nearest-colour-map* against
   frame 1’s palette (their backgrounds landing on index 0 is the next slice);
   in this slice they may fall back to `procedural_squash`, which copies frame 1
   and therefore already carries the key background — so structural tests stay
   green.

## Edge cases

- **All-background input** (no creature): `_quantize_gen3` still assembles
  `[key, black, white]`, every pixel maps to index 0. `postprocess` inherits
  this — no crash, palette head is `key, black, white`.
- **Fully-random / noisy input** (e.g. the old `_noisy_image()` 512×512): its
  1-px border ring is not near-uniform, so `_flatten_background_to_key` takes
  the **gradient/vignette fallback** — it keys only the dominant border colour
  and prints a warning to stderr, then `_quantize_gen3` quantizes the rest as
  “creature”. The result is still a valid ≤16-colour P image, but there is no
  coherent “background maps to index 0” guarantee. **Consequence for tests:** a
  test asserting “every background pixel is index 0” must feed an image with a
  real, near-uniform background (e.g. `_noisy_border_sprite()`,
  `_multicolor_creature()`, or `_sprite_rgb()`), not the fully-random noise
  image.
- **Creature colour near the key** (within `_KEY_COLLISION_DISTANCE`): nudged
  off the key by `_nudge_off_key`, so it stays distinct from the transparency
  background (already covered by `_quantize_gen3` tests).
- **Frame-2 acceptance band shift (key pitfall).** `build_frame2` accepts a
  candidate only when its palette-locked, recentred `difference_ratio` to frame 1
  lands in `[low=0.02, high=0.30]`. Two interacting changes matter now:
  - Frame 1’s background is uniformly the key (index 0) by construction, so
    background pixels are identical between frames and the diff-ratio is now
    driven almost entirely by the creature region.
  - `quantize_to_reference` does **not** flatten a candidate’s background to the
    key; a dark background such as `_sprite_rgb`’s `(40, 40, 60)` nearest-maps to
    reserved **black (index 1)**, not the key, so a candidate built from a plain
    `_sprite_rgb` recolour differs from frame 1 across most of the frame and is
    **rejected** (ratio above `high`) → falls back to `procedural_squash`. This
    is expected in this slice (background-to-key for candidates is the next
    slice). It means the *in-band accept* test can no longer use a plain
    dark-background candidate; see “Tests”.

## Errors

- No new error paths. `postprocess` raises nothing it did not before;
  `_quantize_gen3` is deterministic pure PIL. The gradient-border fallback
  inside `_flatten_background_to_key` warns to stderr and never raises, so
  generation never fails on a hard-to-key background.
- Existing `ValueError`s elsewhere (`quantize_to_reference` non-P reference,
  `procedural_squash` / `recenter_to_anchor` / `build_frame2` non-P `frame1`)
  are unchanged.

## Constraints & dependencies

- Pure PIL (Pillow) — no torch/diffusers touched by this slice. Deterministic:
  median-cut quantize + fixed arithmetic, dither off.
- Test slicing (repo convention in `CLAUDE.md`): everything here except the ML
  fixture note is pure PIL and lives in `tests/test_sprites.py`. Run `pytest`
  from the repo root (flat package). ~21 `ml`-marked tests auto-skip in the keep
  sandbox without torch — **expected and correct**; do **not** `pip install`
  torch/diffusers to “fix” skips. The full suite (including `ml`) runs on the
  host.
- Keep `postprocess`’s signature and return type intact so all callers and the
  `_frame1_file` fixture keep working.

## Tests to rewrite / add

All in `tests/test_sprites.py` (the light, no-torch file) unless noted.

- **`test_postprocess_at_most_16_colors` → rewrite to a contract test.** Rename
  to something like `test_postprocess_obeys_gen3_contract` and feed a
  background-bearing sprite (e.g. `_noisy_border_sprite()` for a clean border, or
  `_multicolor_creature()` when exercising the 13-colour budget) — **not** the
  fully-random `_noisy_image()` (see Edge cases). Assert: `palette[0:3] ==
  [200,200,168]`; `palette[3:6] == [0,0,0]`; `palette[6:9] == [255,255,255]`;
  distinct **creature** colours (used colours minus the three reserved) ≤ 13;
  total distinct used colours ≤ 16; and every border/background pixel decodes to
  index 0. (These mirror the existing `_quantize_gen3` tests, asserted at the
  public integration point.)
- **Keep passing (adjust only if needed):** `test_postprocess_resizes_to_96x96`,
  `test_postprocess_output_is_palette_mode`,
  `test_postprocess_does_not_mutate_input` — all still hold, since
  `_quantize_gen3` returns a fresh 96×96 P image and does not mutate its input.
- **`quantize_to_reference` light tests** that build their reference via
  `postprocess(...)`: re-verify they still pass. They should, because
  `quantize_to_reference` copies the reference palette verbatim:
  `test_quantize_to_reference_reuses_reference_palette`,
  `test_quantize_to_reference_at_most_16_colors` (contract is ≤16),
  `test_quantize_to_reference_shares_palette_across_inputs`,
  `test_back_sprite_locks_to_reference_frame_palette`,
  `test_cross_view_shinies_share_one_rotated_palette`. Optional cleanup: where a
  reference is built from `postprocess(_noisy_image())`, switching the source to
  a background-bearing image (e.g. `_noisy_border_sprite()`) avoids the
  gradient-border stderr warning; not required for correctness.
- **Frame-2 acceptance band retune:**
  - `test_build_frame2_in_band_candidate_is_accepted` **must be reworked**: a
    plain dark-background `_sprite_rgb` recolour now nearest-maps its background
    to black and is rejected. Rebuild the candidate so its **background locks to
    the key (index 0)** — i.e. give it a `_KEY_COLOR` (200,200,168) background
    (or otherwise a background that nearest-maps to index 0) so the only
    difference from frame 1 is the recoloured creature region, yielding an
    in-band ratio in `[0.02, 0.30]`. Keep the assertions: output ≠
    `procedural_squash(frame1)`, shares frame 1’s palette, and
    `0.02 <= difference_ratio(out, frame1) <= 0.30`.
  - `test_procedural_squash_differs_within_acceptance_band`: re-check that a
    genuine squash of the contract sprite still lands in `[0.02, 0.30]` (the
    background is now uniformly index 0 in both frames, so the ratio is driven by
    the squashed creature region). Expected to remain in-band; if measurement
    shows drift, retune the assertion bounds and/or `build_frame2`’s `low`/`high`
    defaults, preserving the semantics (below `low` = texture shimmer, above
    `high` = teleport/identity-drift). Keep `0.0 < ratio`.
  - `test_build_frame2_near_identical_candidate_falls_back` and
    `test_build_frame2_wildly_different_candidate_falls_back`: still pass (both
    reject → squash). The former’s inline comment (“ratio < low”) is now
    inaccurate — the dark-background candidate is rejected because it lands
    *above* `high`, not below `low`; update the comment to match, but the
    assertion (`== procedural_squash(frame1)` data) still holds.
  - `test_build_frame2_no_candidate_returns_squash`,
    `test_build_frame2_always_shares_palette_96x96`,
    `test_build_frame2_rejects_non_palette_frame1`: unchanged, still pass.
- **`tests/test_sprites_ml.py` (auto-skipped without torch):** the `_frame1_file`
  fixture and the P-mode / 96×96 / PNG structural assertions still hold; only
  palette-*content* expectations could need updating, and none of the current ml
  tests assert palette content in a way the contract breaks. In particular
  `test_frame2_falls_back_to_squash_on_garbage_candidate` still passes (feeding
  frame 1 back as RGB now has a key background that locks to index 0, reproducing
  frame 1 → ratio below `low` → squash). Do **not** add torch-triggering tests to
  the light file. These are verified on the host, not in the keep sandbox.

## Assumptions

Marked ⟨D⟩ where a default was picked (not confirmed by code/tests/docs);
⟨C⟩ where confirmed by the codebase.

- ⟨D⟩ **`postprocess` becomes a thin wrapper: `return _quantize_gen3(image)`.**
  Chosen to keep one source of truth for the contract and avoid duplicating the
  flatten/quantize/assembly logic. (Explicitly the task’s stated intent.)
- ⟨C⟩ **The resize/enhance pre-steps are removed from `postprocess`**, because
  `_quantize_gen3` already applies them before flattening; keeping them would
  double-apply the enhancement (confirmed by reading `_quantize_gen3`).
- ⟨C⟩ **`main.py` needs no change** and `tests/test_main.py` needs no change —
  the sprite functions are mocked there and only wiring is asserted (confirmed
  by reading both files).
- ⟨D⟩ **The rewritten `postprocess` contract test uses a background-bearing
  sprite** (`_noisy_border_sprite()` / `_multicolor_creature()`), not the
  fully-random `_noisy_image()`, so the “background → index 0” assertion is
  meaningful and doesn’t hit the gradient-border fallback.
- ⟨D⟩ **The in-band frame-2 accept test’s candidate is given a `_KEY_COLOR`
  background** so its background locks to index 0 and only the recoloured
  creature drives an in-band diff. Chosen over loosening the band, to preserve
  the band’s semantics. (Making candidate backgrounds key-flatten automatically
  is the next slice.)
- ⟨D⟩ **`build_frame2`’s `low`/`high` defaults stay `0.02`/`0.30`** unless a
  measured squash/accept ratio falls out of band on the host, in which case they
  (and/or the test bounds) are retuned — they are documented eyeball
  placeholders open to a later ML-tuning slice. Preferred fix order:
  reconstruct the test candidate first, retune bounds only if the real squash
  itself drifts.
- ⟨D⟩ **Known out-of-scope limitation:** `generate_shiny` hue-rotates every
  chromatic palette entry, and the key `(200,200,168)` is chromatic
  (luminance ≈ 196 < 215), so it *would* be rotated in shiny palettes — the
  transparency key is not currently preserved through shininess. This does not
  break `test_cross_view_shinies_share_one_rotated_palette` (all views still
  share one rotated palette). Preserving the key through `generate_shiny` is a
  later slice; this slice does not touch `generate_shiny`.
