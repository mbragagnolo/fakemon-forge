# Spec: Background-flatten-to-key helper for the Gen 3 palette

## Summary

`fakemon-forge` generates 96×96 Gen-3-style Fakemon sprites. Sprite
post-processing lives in `fakemon_forge/sprites.py`, where `postprocess()`
today does `resize → colour/contrast enhance → quantize(colors=16)`, producing
an *adaptive* palette. The Stable-Diffusion backgrounds are near-white noise
that smears across several palette entries (e.g. `254,254,252` / `255,255,253`
/ …), so there is **no single dedicated background colour** a ROM tool can key
transparency off of.

The larger goal (#11) is an authentic Gen-3 palette contract where the
background is a single dedicated **transparency key colour, RGB `(200, 200,
168)`** — measured across the reference sheets. This slice (**1/4 of #11**,
issue **#12**) builds only the **first building block**: a pure, standalone,
tested helper that flattens a sprite's background to that key colour. It does
**not** wire anything into `postprocess()` (a later slice does that), and it
changes **no existing behaviour** — `postprocess`, `quantize_to_reference`,
`generate_shiny`, `procedural_squash`, `build_frame2`, and `main` are all
untouched.

Concretely this slice adds, in `fakemon_forge/sprites.py`:

1. A module constant `_KEY_COLOR = (200, 200, 168)` near the other palette
   constants (`_PALETTE_COLORS = 16`, etc.), plus a tolerance constant.
2. A pure helper `_flatten_background_to_key(image: Image.Image) -> Image.Image`
   that takes an **RGB** image and returns a **new RGB** image with the
   background replaced by `_KEY_COLOR`, leaving the creature untouched, using a
   two-stage border-flood-fill + global-sweep design with a gradient-border
   fallback.

### Explicitly out of scope

- **Wiring into `postprocess` / `quantize_to_reference`** — this helper is a
  reusable primitive only; no call site is added in this slice. Later slices of
  #11 do the wiring and the palette/quantize integration.
- **Palette / quantization changes** — the helper operates purely in RGB space
  and does not quantize, resize, or touch any 16-colour palette. It does not
  guarantee `_KEY_COLOR` survives a later `quantize`; that is a downstream
  slice's concern.
- **Alpha / true transparency** — the helper paints an opaque RGB key colour,
  not an alpha channel. Actual transparency keying in a ROM tool is downstream.
- **ML / diffusers / torch** — the helper is pure PIL; it triggers no
  `import torch` and is therefore a *light* (torch-free) addition, tested in
  `tests/test_sprites.py` per `CLAUDE.md`'s test-slicing rule.

## Inputs

### New helper: `_flatten_background_to_key(image: Image.Image) -> Image.Image`

- `image: Image.Image` — an **RGB** Pillow image (any size; in practice the
  raw SD output or a 96×96 sprite). The helper reads pixel data via
  `image.load()` / `image.getdata()` and border rows/columns; it does **not**
  mutate this input (see Behavior / Edge cases).
  - **[picked]** The helper assumes RGB mode and does not convert. Callers pass
    RGB (matching `_run_img2img`, which produces RGB, and `postprocess`, which
    receives RGB). A non-RGB input yields undefined channel semantics; a later
    wiring slice is responsible for handing it RGB. See Assumptions.

### New module constants (near `_PALETTE_COLORS = 16`)

- `_KEY_COLOR = (200, 200, 168)` — the Gen-3 transparency key colour (RGB).
- `_KEY_TOLERANCE` (name **[picked]**) — a per-pixel distance threshold used
  both by the border flood fill (`thresh`) and by the global near-background
  sweep to decide "is this pixel background?". A small, sensible default (e.g.
  a Euclidean RGB distance around **30**, or an equivalent per-channel band) —
  **eyeball placeholder, tunable**, documented as such in a comment mirroring
  the existing `amount_px` / `low` / `high` "tunable eyeball placeholder"
  convention.

## Outputs

- Returns a **new** `Image.Image` in **RGB** mode, same size as the input, in
  which:
  - Every pixel that was part of the background (outer background reachable by
    flood fill from the borders, **and** any enclosed background pockets caught
    by the global sweep) is exactly `_KEY_COLOR = (200, 200, 168)`.
  - Every creature pixel (outside tolerance of the detected background colour)
    is byte-for-byte unchanged from the input.
- No side effects, no file writes, no stdout. The **only** thing the helper may
  write is a single warning line to **stderr** in the gradient-border fallback
  case (see Behavior / Errors).

## Behavior

`_flatten_background_to_key(image)` proceeds in these steps. It is fully
deterministic — no randomness, no time, no I/O beyond the stderr warning.

1. **Copy the input.** Work on `out = image.copy()` (or build a fresh RGB image);
   the original `image` is never mutated. All subsequent edits target `out`.

2. **Sample the border ring.** Collect the pixels of the outermost ring — the
   top and bottom rows and the left and right columns (1 px wide by default).
   These are treated as "known background" samples.

3. **Detect the background colour** from the border ring rather than assuming
   pure white (SD sometimes paints gradients/vignettes/tints):
   - Compute a representative background colour `bg` — **[picked]** the
     per-channel **mean** (rounded to int) of the border ring, which is robust
     to the near-white noise. (A dominant/mode colour is an acceptable
     alternative; mean chosen for noise-robustness and simplicity.)

4. **Uniformity check (gradient/vignette guard).** Decide whether the border
   ring is *near-uniform*:
   - **[picked]** The ring is near-uniform if the fraction of border pixels
     within `_KEY_TOLERANCE` of `bg` is at least a high threshold (e.g. ~90%),
     or equivalently if the per-channel spread of the ring stays within the
     tolerance band. Threshold is a tunable eyeball placeholder.
   - **If NOT near-uniform** (a gradient/vignette background): **fall back** —
     do **not** flood-fill or globally sweep with a single `bg` (that could eat
     the creature). Instead key only the **dominant** border colour: replace
     pixels within `_KEY_TOLERANCE` of the dominant border colour with
     `_KEY_COLOR`, emit a warning to **stderr**
     (`print(..., file=sys.stderr)`), and return `out`. **Do not raise; do not
     fail generation.** Then skip steps 5–6.

5. **Border flood fill (outer background).** For the near-uniform case, flood
   from the border ring inward, replacing near-background pixels with
   `_KEY_COLOR`:
   - Use `PIL.ImageDraw.floodfill(out, seed, _KEY_COLOR, thresh=_KEY_TOLERANCE)`
     seeded from border pixels (e.g. the four corners, or every border pixel
     still near `bg`), so the connected outer background region is filled with
     the key while the creature (a tolerance "wall" away from `bg`) is left
     intact. `thresh` provides the tolerance that an exact match cannot (SD's
     white background is noisy). This keys the **outer** background only.

6. **Global near-background sweep (enclosed pockets).** After the flood fill,
   iterate all pixels of `out` and replace any pixel still within
   `_KEY_TOLERANCE` of the detected `bg` **and** not already `_KEY_COLOR` with
   `_KEY_COLOR`. This catches **enclosed** background pockets (gaps between
   legs, under arms, the hole of a ring-shaped creature) that the border flood
   could not reach — as authentic Gen-3 sprites have their interior gaps keyed
   too.
   - The distance metric matches the flood-fill tolerance semantics (per-pixel
     distance from `bg`), so the outer flood and the sweep agree on what counts
     as background.

7. **Return `out`** (RGB, same size).

### Reuse / structure notes

- Reuses `PIL.ImageDraw` (already imported in the test file; `ImageDraw` is
  imported into `sprites.py` for `floodfill`) and `sys` (already imported at the
  top of `sprites.py` for the existing stderr paths).
- Uses `_KEY_COLOR` / `_KEY_TOLERANCE` constants rather than magic numbers,
  mirroring how `_SPRITE_SIZE` / `_PALETTE_COLORS` are used.
- The tolerance and uniformity threshold carry the same "tunable eyeball
  placeholder" comment style as `procedural_squash`'s `amount_px` and
  `build_frame2`'s `low`/`high`.

## Edge cases

- **Enclosed background pocket** (ring-shaped creature with a background hole
  inside) → the border flood cannot reach the pocket, but the **global sweep**
  keys it, so pocket pixels become `_KEY_COLOR`. (Directly tested.)
- **Noisy near-white border** (the common SD case: `254,254,252` /
  `255,255,253` / …) → `bg` is the mean of the noise, `thresh`/tolerance
  absorbs the per-pixel variation, and every border pixel ends exactly at
  `_KEY_COLOR`. (Directly tested: every border pixel `== (200,200,168)`.)
- **Non-uniform / gradient / vignette border** → the uniformity check fails;
  the helper keys only the dominant border colour, warns to stderr, and returns
  a valid RGB image **without raising**. (Directly tested via `capsys`.)
- **Creature colour close to the key colour** → creature pixels within tolerance
  of `bg` would be keyed; but `bg` is detected from the border (near-white),
  and `_KEY_COLOR` `(200,200,168)` is olive-ish, so a creature is only at risk
  when it is itself near the *border background* colour. Accepted for this
  primitive; a caller can pre-mask if needed. Not mitigated in this slice.
- **All-background image** (no creature) → flood + sweep key everything to
  `_KEY_COLOR`; returns an all-key RGB image. Not an error.
- **All-creature image** (no near-background pixels) → `bg` is whatever the
  border is; if the border is not near-background of anything, the flood fills
  little/nothing and the sweep matches little/nothing; result is essentially
  the input unchanged. Not an error.
- **Input not mutated** → the original image's size and pixel data are preserved
  (the helper works on a copy). (Directly tested.)

## Errors

- The helper **does not raise** on the difficult (gradient/vignette) border
  case — that is the explicit robustness requirement: it warns to stderr and
  returns a best-effort result so sprite generation never fails on it.
- No new `ValueError`/`sys.exit` paths are introduced. (Unlike
  `quantize_to_reference` / `procedural_squash`, which guard on `P`-mode input,
  this helper does not validate mode; see Assumptions — RGB is assumed by
  contract, matching how the later wiring slice will call it.)
- The stderr warning is the only diagnostic output; wording is a short
  human-readable notice (e.g. that a non-uniform/gradient border was detected
  and only the dominant border colour was keyed).

## Constraints & dependencies

- Change is confined to `fakemon_forge/sprites.py`: two constants and one pure
  function. No other module changes; no CLI/`main` changes; no signature changes
  to any existing function.
- **Pure PIL, torch-free.** The helper triggers no `import torch` / diffusers,
  so per `CLAUDE.md` it and its tests are the *light* slice: tests go in
  `tests/test_sprites.py`, which runs in the keep sandbox (pytest + Pillow +
  mistralai only). In that sandbox ~21 `ml` tests report as **skipped** — that
  is expected and correct; do **not** install torch to "fix" the skips.
- Requires `PIL.ImageDraw` (for `floodfill`) and `sys` (for the stderr warning);
  `sys` is already imported at the top of `sprites.py`, and `ImageDraw` is added
  to the existing `from PIL import ...` line.
- Deterministic: no `Math.random`/time/RNG (irrelevant here — pure PIL), so
  tests are stable and no seeding is needed.
- Backward compatibility: because nothing calls the helper yet, no existing
  test or behaviour can change; the full `pytest` run must stay green (with the
  usual `ml` skips in the sandbox).

## Tests

All tests are **light** and go in `tests/test_sprites.py` (torch-free), added
under a new section header matching the file's existing
`# ---- name() ----` divider convention, importing `_flatten_background_to_key`
and `_KEY_COLOR` from `fakemon_forge.sprites`. They follow the existing
synthetic-image helper style (`_rgb_image`, `_noisy_image`, `_sprite_rgb`,
building shapes with `ImageDraw`).

1. **Noisy border → every border pixel is exactly the key.** Build a 96×96 RGB
   with a **noisy near-white border** and a **solid creature blob** in the
   middle (colour far from white, e.g. a filled ellipse). After
   `_flatten_background_to_key`, assert **every border pixel** equals
   `(200, 200, 168)` exactly, and the creature-blob pixels are **unchanged**
   from the input.
2. **Enclosed pocket → global sweep keys it.** Build a ring-shaped creature
   (e.g. an outer filled ellipse/disc with a background-coloured hole punched
   in the centre) over a near-background field. After the helper, assert the
   **pocket** (interior hole) pixels are `(200, 200, 168)` — proving the global
   sweep reaches enclosed background the outer flood cannot.
3. **Input not mutated.** Capture the original image's size and pixel data
   (e.g. `list(img.getdata())`), call the helper, and assert the original is
   unchanged in size and data (the helper returns a fresh image).
4. **Non-uniform (gradient) border → warns, does not raise, returns valid RGB.**
   Build an image whose border is a **gradient/vignette** (not near-uniform).
   Assert the call **does not raise**, returns an RGB image of the same size,
   and that a **warning was emitted to stderr** (assert via `capsys` on
   `capsys.readouterr().err`).

Optional supporting assertions (if cheap): the output mode is `"RGB"` and the
output size equals the input size in every case.

Run `pytest` from the repo root (flat package layout). In the sandbox ~21 `ml`
tests report as **skipped** — expected and correct; do not install torch.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase or the
issue text.

- **[confirmed]** Helper name `_flatten_background_to_key`, RGB→RGB signature,
  `_KEY_COLOR = (200, 200, 168)`, two-stage (border-flood + global-sweep)
  design, border-detected background colour, and the gradient-border
  warn-don't-raise fallback — all specified by issue #12.
- **[confirmed]** This is a standalone primitive; `postprocess`,
  `quantize_to_reference`, `generate_shiny`, and `main` are **not** modified in
  this slice (wiring is a later slice of #11).
- **[confirmed]** Tests belong in `tests/test_sprites.py` (light/torch-free),
  not `tests/test_sprites_ml.py`, because the helper triggers no
  `import torch` / diffusers — per `CLAUDE.md`'s test-slicing rule.
- **[picked]** Background colour is detected as the **per-channel mean** of the
  border ring (robust to near-white noise). A dominant/mode colour is an
  acceptable alternative; mean chosen for simplicity and noise-robustness. The
  gradient-fallback keys the **dominant** border colour specifically (per the
  issue's wording).
- **[picked]** `_KEY_TOLERANCE` default is a small per-pixel distance
  (≈ Euclidean 30, or an equivalent per-channel band) — an eyeball placeholder,
  explicitly documented as tunable, mirroring the existing
  `amount_px`/`low`/`high` placeholder convention.
- **[picked]** The border **ring width is 1 px**. Wider rings (2–3 px) are a
  tunable alternative if 1 px proves too noisy; 1 px chosen as the simplest
  default.
- **[picked]** The uniformity test is "≥ ~90% of border pixels within
  `_KEY_TOLERANCE` of the mean `bg`" (equivalently, per-channel spread within
  the tolerance band). Threshold tunable.
- **[picked]** Flood fill via `PIL.ImageDraw.floodfill(..., thresh=...)` seeded
  from the border (corners and/or near-`bg` border pixels); the global sweep is
  a per-pixel distance check against `bg`. Both use the same tolerance so they
  agree on "background".
- **[picked]** The helper assumes **RGB** input and does **not** convert or
  validate mode (no `ValueError` guard), unlike the `P`-mode-guarded functions,
  because the later wiring slice controls the call site and passes RGB. If a
  guard is later wanted it is additive and non-breaking.
- **[picked]** The helper paints an **opaque** `_KEY_COLOR`; no alpha channel /
  true transparency is produced (downstream ROM-tool concern).
- **[confirmed]** No randomness/time is involved (pure PIL), so the helper is
  deterministic and tests need no seeding.
