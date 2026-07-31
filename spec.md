# Spec: `icon.py` — party-menu icon post-processing (`sprite_small.png`)

## Summary

Add a new pure-Pillow module `fakemon_forge/icon.py` that turns an
already-generated front sprite into a Gen-3-style party / team-selection menu
icon: a **32x64** PNG laid out as **two vertically stacked 32x32 animation
frames**, with an **opaque** teal-green background and a small (<= 16 colour)
palette.

This is slice 1/3 of issue #21 (via #26). It builds **only** the deterministic
post-processing step. There is **no** Stable Diffusion / torch / diffusers
involvement, so the whole module and its tests run in the keep sandbox (regular
test file, not marked `ml`). `main.py` and the rest of the pipeline are **not**
touched here — wiring the icon into per-stage generation is a later slice.

The public entry point is:

```python
def generate_icon(source_path: str, output_path: str) -> None
```

"Done and correct" means: `fakemon_forge/icon.py` exists exporting a tested
`generate_icon`; `tests/test_icon.py` exercises it with a synthetic `P`-mode
sprite fixture and asserts the full output contract below; `pytest` passes
(new icon tests actually run in-sandbox because they need no torch).

## Inputs

- `source_path: str` — filesystem path to an existing front sprite produced by
  the pipeline. The pipeline's sprites are **`P`-mode** PNGs whose native size is
  768x768, whose palette **index 0** is the transparency key
  `_KEY_COLOR = (200, 200, 168)` (imported from `fakemon_forge.sprites`), and
  whose background (both the outer backdrop and interior keyed pockets) decodes
  to index 0. The creature occupies the non-index-0 palette entries.
- `output_path: str` — filesystem path to write the resulting `sprite_small.png`.

No other parameters. Behaviour is fully determined by the source pixels — no
randomness, no seed, no config.

## Outputs

A single PNG written to `output_path`:

- Size **(32, 64)**, format `"PNG"`.
- Two stacked 32x32 frames: **frame 1** occupies rows 0..31 (top), **frame 2**
  occupies rows 32..63 (bottom).
- **Fully opaque** — every pixel has an opaque value and the file carries **no**
  transparency: if saved as `P`-mode there is no `transparency` chunk / no
  reserved transparency index in use (this mirrors how `sprites.py` already
  saves `P`-mode PNGs opaque — the key colour is just a normal opaque colour a
  downstream ROM tool keys against, not a PNG alpha channel).
- **<= 16 distinct colours** across the whole 32x64 image (fits a 16-colour
  palette): up to 15 quantized creature colours plus the teal background.
- Background colour is teal-green **(96, 152, 128)**, forced onto **palette
  index 0**, and dominates the image.

Returns `None`.

## Behavior

`generate_icon(source_path, output_path)`:

1. **Open + validate.** Open `source_path`. If its mode is not `"P"`, raise
   `ValueError` (see Errors). No conversion-away-from-`P` is attempted — the
   creature-vs-background split relies on index 0 being the key.

2. **Build frame 1 (32x32, opaque, teal background at index 0).**
   - Perform a **single** high-quality downscale from the source's native size to
     32x32: convert to `RGB` and do one `LANCZOS` resample 768 -> 32 (single
     resample so a 1px outline is never chained through multiple resamples).
   - Determine which output pixels are **background**: the source's keyed region
     (index 0) plus anything that still reads as background after the downscale.
     Background pixels are mapped to the teal index; the creature region is
     mapped to its quantized colours.
   - Quantize the **creature region** to **up to 15 colours** using PIL adaptive
     (median-cut) quantization with **dither OFF** (matching how `sprites.py`
     quantizes deterministically). Fewer than 15 resulting colours is fine.
   - Assemble a palette with the **teal background (96, 152, 128) forced onto
     index 0** and the up-to-15 creature colours after it. Every background pixel
     decodes to index 0; no creature colour is allowed to collide with / be
     mistaken for the background. The result is a 32x32 `P`-mode (or equivalent
     opaque) image with **no transparency** — no alpha holes.

3. **Build frame 2 procedurally (1px down-shift, zero extra generation).**
   Frame 2 is frame 1 shifted **down 1px**: the top row (row 0) is filled with
   the teal background, rows 1..31 equal frame 1's rows 0..30, and frame 1's
   bottom row (row 31) is cropped away. Purely a copy/paste in the frame-1
   palette space — introduces no new colours. (Reference-verified: the real
   Gen-3 frame 2 differs from a 1px-shifted frame 1 by only ~64/1024 edge
   pixels, so the procedural shift is faithful.)

4. **Stitch + save.** Compose frame 1 on top and frame 2 on the bottom into the
   final **32x64** image and save it to `output_path` as PNG, opaque, sharing a
   single <= 16-colour palette with teal at index 0.

Helpers are private module-level functions (small, docstringed, pure,
deterministic), mirroring the style in `sprites.py` (e.g. `_quantize_gen3`,
`procedural_squash`, `stitch_spritesheet`).

## Edge cases

- **Creature-region is empty** (source is entirely index 0 / all background):
  frame 1 is entirely teal; frame 2 (a down-shift of an all-teal frame) is also
  entirely teal. Output is still 32x64, opaque, single teal colour, valid PNG.
- **Fewer than 15 creature colours** after quantization: allowed — palette is
  "up to" 15 creature colours plus teal; no padding to exactly 16 is required.
- **Creature colour close to teal:** a creature colour must not become
  indistinguishable from / remap into the background index 0. Frame 1's teal must
  remain a distinct entry at index 0 so background pixels resolve there and the
  frame-2 down-shift can backfill row 0 with exactly that colour. (Mechanism is
  an implementation detail; the observable contract is "teal is present at index
  0 and background pixels decode to it".)
- **Source larger/smaller than 768:** the downscale targets 32x32 regardless of
  the exact source size (the spec's 768 -> 32 is the expected path; any
  `P`-mode source is resampled to 32 in one step). Test fixtures use a small
  (e.g. 96px) `P`-mode sprite for speed.
- **Bottom row of frame 1** is intentionally dropped by the frame-2 shift; the
  top row of frame 2 is intentionally synthetic (teal). Both are required by the
  reference contract, not bugs.

## Errors

- **Non-`P`-mode input** (`Image.open(source_path).mode != "P"`): raise
  `ValueError` (message naming the actual mode, mirroring
  `generate_shiny` / `procedural_squash`: `f"Expected palette-mode ..., got {mode}"`).
  The pipeline caller is expected to wrap `generate_icon` in a
  warn-and-continue block (per the per-view degradation pattern in
  `stitch_spritesheet` / sprite generation) — this module itself does not catch;
  it raises and lets the caller degrade. Wiring of that caller is out of scope
  for this slice.
- File-not-found / unreadable `source_path`: the underlying `Image.open` /
  filesystem error propagates unchanged (not specially handled here).

## Constraints & dependencies

- **Pure Pillow only.** No torch, diffusers, transformers, compel, or any ML
  import — not even function-local. This keeps the module and `tests/test_icon.py`
  runnable in the slim keep sandbox (Pillow + pytest + mistralai), so the icon
  tests **must not** be marked `ml` and must live in a **regular** test file.
- **Deterministic.** No `random`, no time, no seed; same input bytes -> same
  output bytes.
- Reuse `_KEY_COLOR` from `fakemon_forge.sprites` rather than re-hardcoding
  `(200, 200, 168)`, to stay in sync if the pipeline's key changes.
- Match `sprites.py` conventions: module-level tunable constants with
  explanatory comments, `P`-space operations that avoid re-quantizing, small
  pure docstringed helpers.
- Add a short in-code comment noting the **per-mon palette** deliberately
  deviates from authentic Gen-3 (which shares 3 fixed palettes across all
  species; ROM-insertion tools remap), so a future reader does not "fix" it.

## Assumptions

Items marked **[chosen default]** were picked here (no existing code/test/doc
confirms them); items marked **[from codebase]** are grounded in existing code.

- **[chosen default]** Public signature is
  `generate_icon(source_path: str, output_path: str) -> None` (as suggested by
  the issue).
- **[chosen default]** Output is saved as `P`-mode PNG with teal at index 0 and
  **no** transparency chunk (consistent with how `sprites.py` already saves
  opaque `P`-mode sprites). An alternative opaque `RGB` save would also satisfy
  the contract; `P`-mode is chosen to match the pipeline and keep the
  distinct-colour count trivially bounded.
- **[chosen default]** Downscale filter is `LANCZOS`, a **single** 768 -> 32
  resample. No post-downscale 1px dark-outline re-stamp is added in this slice
  (explicitly deferred by the parent issue); ship the plain high-quality
  downscale — outline work can be a later tweak if readability needs it.
- **[chosen default]** Creature quantization is PIL adaptive (median-cut) with
  **dither OFF**, up to **15** creature colours, matching `sprites.py`'s
  deterministic quantization approach. (`sprites.py` uses a 13-creature +
  key/black/white budget; the icon uses up to 15 creature + teal because it has
  no reserved black/white slots — teal is the only reserved entry.)
- **[chosen default]** Background detection after downscale: the source's index-0
  (key) region defines the background, propagated to the 32x32 result; any
  downscaled pixel that still reads as the key/background also maps to teal. The
  exact tolerance/threshold mechanism (e.g. reusing a `_KEY_TOLERANCE`-style
  radius, or masking from the source's index-0 pixels before resampling) is an
  implementation detail left to the implementer, constrained only by the
  observable output contract (opaque, teal at index 0, background dominates).
- **[chosen default]** Frame 2 is a strict 1px down-shift with row 0 backfilled
  teal and row 31 of frame 1 dropped — no img2img, no squash, no recentring
  (unlike `build_frame2` for the battle sprites). The reference note that real
  frame 2 differs by only ~64/1024 edge pixels justifies this.
- **[chosen default]** Assume the input is a `P`-mode sprite with the background
  on index 0 (the key). Non-`P`-mode input raises `ValueError`; a `P`-mode image
  whose index 0 is *not* the key is out of contract and not specially handled
  (the pipeline only ever produces key-at-index-0 sprites).
- **[chosen default]** Per-mon palette (up to 15 quantized colours + teal),
  deliberately deviating from authentic Gen-3's 3 shared fixed palettes; noted in
  an in-code comment. ROM-insertion tools remap, so this is acceptable for the
  generator's output.
- **[from codebase]** `_KEY_COLOR = (200, 200, 168)` and native sprite size 768,
  `P`-mode with background on index 0, and the "raise `ValueError` on non-`P`
  input" convention all come from `fakemon_forge/sprites.py`
  (`_quantize_gen3`, `generate_shiny`, `procedural_squash`).
- **[from codebase]** Tests with no ML import go in a regular test file and run
  in-sandbox; torch-touching tests are marked `ml` and auto-skip
  (`tests/conftest.py`, `CLAUDE.md`).
- **[from codebase / scope]** `main.py` and pipeline wiring are **not** modified
  in this slice; `generate_icon` is only invoked directly by its own tests here.
