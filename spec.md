# Spec: Palette-lock quantize helper (`quantize_to_reference`)

## Summary

Add a pure-PIL helper to `fakemon_forge/sprites.py` that quantizes an image
against a **reference** palette instead of building a fresh adaptive one.

Today `postprocess(image)` resizes to 96×96 (NEAREST), bumps colour/contrast
via `ImageEnhance`, then calls `image.quantize(colors=16)` — which invents a
brand-new adaptive 16-colour palette every call, so no two sprites share the
same 16 colours. The authentic Gen 3 model requires a *single shared 16-colour
palette* across a creature's whole sprite set (front animation frames + back
sprite all index into one palette).

This slice adds the foundational primitive that shared-palette work will build
on: given a reference `P`-mode image, quantize any input so its output reuses
the reference's **exact** palette. `postprocess` is left untouched — this is an
additive function, not a replacement.

This is slice 1/4 of #1. It introduces only the primitive; wiring it into the
sprite-generation flow (front frames, back sprite, callers, CLI) is out of
scope for this slice.

## Inputs

- `image: Image.Image` — the image to quantize. Typically RGB straight from the
  diffusion pipeline (`result.images[0]`), at generation size (e.g. 768×768),
  but any size/mode Pillow can resize and enhance is acceptable. The helper
  does its own resize + enhance, so callers pass the raw image.
- `reference: Image.Image` — a palette-mode (`P`) PIL image whose palette the
  output must adopt. In practice produced by `postprocess(...)` on the first
  sprite of the set, whose palette becomes the locked palette for the rest.

Suggested signature: `quantize_to_reference(image, reference)`.

## Outputs

- Returns a new `P`-mode `Image.Image` of size 96×96 (`_SPRITE_SIZE`).
- `result.getpalette()` is byte-for-byte identical to `reference.getpalette()`.
- Pixel indices reference at most 16 (`_PALETTE_COLORS`) distinct palette
  entries (bounded by the reference palette populated by `postprocess`, which
  quantizes to 16 colours).
- Neither `image` nor `reference` is mutated.

## Behavior

1. Validate `reference` is `P`-mode (see Errors).
2. Resize `image` to `(_SPRITE_SIZE, _SPRITE_SIZE)` with `Image.NEAREST`.
3. Apply `ImageEnhance.Color(...).enhance(1.1)` then
   `ImageEnhance.Contrast(...).enhance(1.1)` — identical constants/order to
   `postprocess`, so an image run through either path gets the same pre-steps.
4. Quantize against the reference palette via
   `image.quantize(palette=reference)`, which reuses the reference's palette
   rather than deriving a new one.
5. Return the resulting `P`-mode image.

Because every image quantized against the same `reference` produces the same
palette, two different inputs against one reference yield two outputs with
identical palettes — the property this slice exists to provide.

## Edge cases

- **Two different inputs, same reference** → both outputs carry the reference's
  palette exactly (equal `getpalette()`). This is the core guarantee.
- **Reference produced by `postprocess`** → its palette may be padded to 256
  entries by Pillow; the output's palette matches it byte-for-byte regardless
  of how many entries are actually used, and distinct indices used stay ≤ 16.
- **Non-RGB input** (e.g. `RGBA`, `L`): Pillow's `quantize(palette=...)` and the
  resize/enhance steps convert as needed; not a targeted use case for this slice
  but should not raise. (Assumption — see below.)
- **Input already 96×96**: resize is a no-op; behaviour unchanged.

## Errors

- If `reference.mode != "P"`, raise `ValueError` with a clear message, matching
  the style already used in `generate_shiny`, e.g.
  `raise ValueError(f"Expected palette-mode reference image, got {reference.mode}")`.
  Rationale: `Image.quantize(palette=...)` requires a `P`-mode palette image;
  failing early with a readable message beats Pillow's internal error.
- No other input validation is added in this slice (input `image` mode/size are
  handled by resize/enhance/quantize).

## Constraints & dependencies

- Pure PIL only — no torch/diffusers, no network, no filesystem I/O. The
  function must run in the torch-free keep sandbox.
- Reuse existing module constants `_SPRITE_SIZE` (96) and `_PALETTE_COLORS`
  (16) rather than hard-coding.
- Must not mutate inputs. Pillow's `resize`/`ImageEnhance`/`quantize` all return
  new images, so following the `postprocess` pattern (rebind local names, never
  operate in place) satisfies this.
- Do **not** modify `postprocess`; the adaptive path stays intact.

## Tests

Add to `tests/test_sprites.py` (runs in the torch-free sandbox; no
`@pytest.mark.ml`). Build the reference with `postprocess(_noisy_image())` so it
has a real 16-colour palette. Cover:

- Output mode is `P` and size is `(96, 96)`.
- Output palette equals the reference palette exactly
  (`out.getpalette() == reference.getpalette()`).
- Output uses at most 16 distinct colour indices (mirror the existing
  `test_postprocess_at_most_16_colors` idiom for counting distinct indices).
- Two different input images quantized against the same reference produce
  outputs with identical palettes.
- Inputs are not mutated (e.g. reference/input size and mode unchanged after the
  call), following `test_postprocess_does_not_mutate_input`.

No `ml`-marked tests are added; the full suite stays green on the host and the
light slice stays green without torch.

## Assumptions

Marked items are defaults chosen here, not confirmed by existing code/tests/docs.

- **[picked]** Function name is `quantize_to_reference(image, reference)`, per
  the issue's suggestion. If a different name is preferred, only the name
  changes, not the behaviour.
- **[picked]** The helper performs its own resize + enhance (rather than
  assuming the caller pre-processed), keeping callers simple and matching
  `postprocess` parity. This is the issue's chosen default.
- **[picked]** `reference` is required to be `P`-mode; a non-`P` reference
  raises `ValueError` (message style borrowed from `generate_shiny`). This
  mirrors Pillow's own requirement for `quantize(palette=...)`.
- **[picked]** The enhance constants (`1.1`, `1.1`) and order (Color then
  Contrast) are copied verbatim from `postprocess` for parity; they are not
  re-tuned in this slice.
- **[picked]** Non-RGB inputs are tolerated (delegated to Pillow) but not a
  targeted use case; no extra handling or validation is added for them.
- **[picked]** Palette equality is asserted via `getpalette()` byte-for-byte,
  including any 256-entry padding Pillow adds — callers care that the palettes
  are *identical*, not about the used-entry count.
- **[confirmed by issue]** Wiring this helper into the actual generation flow
  (choosing a reference frame; applying it to back/front sprites, callers, CLI)
  is deferred to later slices of #1 and is out of scope here.
