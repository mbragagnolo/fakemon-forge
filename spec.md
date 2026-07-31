# Spec: `footprint.py` — 16×16 Pokédex footprint renderer (Pillow-only)

## Summary

Add a new, self-contained module `fakemon_forge/footprint.py` that renders a
**Gen-3 Pokédex footprint** from a stage's finished front sprite. The footprint
is the tiny monochrome "foot stamp" the Gen-3 Pokédex shows next to a species:
a single stylized foot (a tapered pad plus a few toe marks).

The output must match the official Gen-3 format exactly:

- a **16×16 PNG**,
- **RGBA**, where every pixel is either **opaque black `(0, 0, 0, 255)`** or
  **fully transparent (`alpha == 0`)** — *no other colours, no partial alpha*,
- content is a single stylized foot derived from the sprite's bottom
  "contact patch", with the toe count/style keyed off the creature's primary
  type.

This module is **Pillow-only** — it must not import torch/diffusers, directly
or transitively, so it runs in the slim sandbox container. It is a leaf module:
**nothing wires it into the app in this slice** (CLI/`main.py`/`writer.py`/
`export_ini.py` are untouched). This slice ships the module and its tests
standing alone; wiring is a later slice.

This is slice 2/3 of #20 and depends on no earlier slice (`keep-depends-on:
none`).

### Relationship to `sprites.py`

Modern pipeline sprites (`sprite.png`) are **P-mode (palette) PNGs at native
768×768** with a flat backdrop stored as the **most common palette index**
(the Gen-3 transparency key at index 0). `footprint.py` reuses two private
helpers from `sprites.py` for background/content detection, mirroring
`procedural_squash` / `recenter_to_anchor`:

- `_background_index(image)` — background palette index = the most common index
  (`max(image.getcolors(maxcolors=w*h))[1]`).
- `_content_bbox(image, background)` — bbox of the non-background region via
  `image.point(lambda p: 255 if p != background else 0).getbbox()`, returning
  `None` when the image is all background.

**Import safety (verified):** a top-level `import fakemon_forge.sprites` pulls
in **neither torch nor diffusers** — those are function-local imports in
`sprites.py`. Confirmed at spec time:
`python3 -c "import sys, fakemon_forge.sprites; print('torch' in sys.modules, 'diffusers' in sys.modules)"` →
`False False`. Therefore `footprint.py` will
`from fakemon_forge.sprites import _background_index, _content_bbox` rather than
copying the helper logic. (If that ever became unsafe, the fallback is to
replicate the two ~2-line helpers locally; not needed now.)

## Inputs

Public function:

```python
def generate_footprint(
    sprite_path: str,
    output_path: str,
    *,
    types: list[str],
    size_fraction: float = 0.9,
    blank: bool = False,
) -> None
```

- `sprite_path` — path to the stage's `sprite.png`: a **P-mode**, native-768,
  flat-backdrop front sprite. Read only when `blank` is false.
- `output_path` — filesystem path where the 16×16 `footprint.png` is written.
- `types` (keyword-only) — the stage's type list, e.g. `["Fire"]` or
  `["Water", "Flying"]`. The **first (primary) type** keys the toe lookup;
  remaining types are ignored. An empty list is treated as "no listed type"
  (0 toes / plain pad).
- `size_fraction` (keyword-only, default `0.9`) — fraction of the 16×16 canvas
  that the footprint's **long axis** spans. Default 0.9 → the foot's longer
  dimension is ~14 px.
- `blank` (keyword-only, default `False`) — when true, write the all-transparent
  16×16 footprint and skip **all** derivation (no sprite read).

Returns `None`; the function's effect is writing the PNG at `output_path`.

## Outputs

A single file at `output_path`:

- PIL image mode **RGBA**, size exactly **(16, 16)**.
- Every one of the 256 pixels is exactly `(0, 0, 0, 255)` **or** fully
  transparent (`alpha == 0`). No greys, no anti-aliased edges, no non-black
  colours, no partial alpha. This is the hard format/colour contract the tests
  assert.
- Opaque black marks on a fully transparent background form a single stylized
  foot (tapered pad + optional toe marks).

## Behavior

Order of operations inside `generate_footprint`:

1. **Blank fast-path.** If `blank` is true → build an all-transparent 16×16
   RGBA image, save it to `output_path`, and return. `sprite_path` is **not**
   opened (so a nonexistent/irrelevant path is fine when `blank=True`).

2. **Load + validate.** Open `sprite_path`. Require **P-mode**; if
   `image.mode != "P"`, raise `ValueError` (see Errors), matching the
   `sprites.py` convention.

3. **Locate content.** Compute `background = _background_index(image)` and
   `bbox = _content_bbox(image, background)`. If `bbox is None` (all-background
   sprite), write the **blank** (all-transparent) output and return.

4. **Contact patch.** Take the bottom **15%** of the content bbox's rows (the
   region where feet touch the ground) and collect the non-background pixels in
   that band. This is the raw silhouette of whatever is touching the ground.

5. **Foot blobs.** Compute connected components of the contact patch. Discard
   noise blobs whose area is below **0.2%** of the band's pixel area. From the
   survivors, pick the **widest** blob as "the foot". If no blob survives, write
   the blank output and return.

6. **Pad shape.** Derive pad dimensions from the chosen blob:
   - `width_ratio` = blob width / content-bbox width.
   - Pad width scales with `width_ratio`, **clamped to a minimum** so a foot is
     always legible.
   - A very thin leg (tiny `width_ratio`) intentionally yields a **small tall
     oval** that reads as a hoof print — this is desired output, not an error.
   - The pad is **tapered**: wider "shoulders" near the toes, narrower at the
     base (the classic footprint teardrop/pad silhouette).

7. **Toe count/style — primary-type lookup with rare contour override.** The
   first type keys a small table defined in `footprint.py`. Representative
   entries (exact contents are an implementation tunable — see Assumptions):
   - Dragon / Fighting / Fire → **3 claw wedges**
   - Flying → **3 bird prongs**
   - Normal / Ground → **4 round toes**
   - Water / Poison / Ghost / Psychic → **0** (plain or webbed pad)
   - any unlisted primary type (or empty `types`) → **0**

   **Contour override (expected to fire rarely):** if the chosen blob's **top
   contour** yields a *strong* bump signal — **≥2 bumps** with a mostly-genuine
   contour (bump-count/quality thresholds are tunables) — the contour-derived
   toe count **overrides** the table. This lets an unusually toe-shaped
   silhouette speak for itself.

8. **Render.**
   - Draw the **tapered pad** (wide shoulders, narrower base) **supersampled**
     (e.g. 4×) then downsample to 16 px, giving a crisp but rounded pad.
   - Stamp the **toe marks directly on the 16×16 grid** — *not* through the
     supersample/downsample step. (Supersampled toes merge unpredictably into
     blobs at 16 px; stamping at final resolution keeps them distinct.)
   - Guarantee a **≥1 px transparent gap row** between the toe tips and the pad
     edge so toes read as separate marks, never fused to the pad.
   - Scale so the footprint's **long axis spans `size_fraction`** of the 16 px
     canvas, roughly centered.

9. **Binarize + save.** Ensure the final image is RGBA with every pixel either
   `(0, 0, 0, 255)` or fully transparent (any supersample/downsample greys are
   thresholded to one or the other — no partial alpha survives). Save to
   `output_path` as PNG.

## Edge cases

- **`blank=True`** → all 256 pixels transparent; `sprite_path` never opened.
- **All-background sprite** (`_content_bbox` returns `None`) → all-transparent
  output (no crash).
- **Contact patch empty / only sub-threshold noise blobs** → all-transparent
  output (no crash).
- **Very thin leg** (tiny `width_ratio`) → deliberate small tall oval (hoof
  print); not an error.
- **Empty `types` list / unknown primary type** → 0 toes (plain pad), unless the
  contour override fires.
- **Primary type with toes vs. type with none** (e.g. `["Normal"]` vs
  `["Water"]`) → both must still satisfy the colour/format contract; exact toe
  pixel positions are a visual tunable and are *not* asserted.
- **Sprite larger/smaller than 768** — the code keys off the content bbox and
  percentages, not a hard 768 assumption, so odd sizes still produce a valid
  16×16 output. (Native 768 is the expected input.)

## Errors

- **Non-P-mode `sprite_path`** (e.g. an RGB or RGBA PNG) → raise
  `ValueError`, message matching the `sprites.py` convention
  (`f"Expected palette-mode ... got {image.mode}"`, so it matches
  `pytest.raises(ValueError, match="palette-mode")`). Only enforced on the
  non-blank path — `blank=True` never reads the sprite, so it never raises on
  mode.
- **Missing `sprite_path` on the non-blank path** → the underlying
  `Image.open` raises `FileNotFoundError` (not caught/reshaped here).
- No other exceptions are raised for "empty" derivations — those degrade to a
  blank footprint rather than erroring (see Edge cases).

## Constraints & dependencies

- **Pillow-only.** No torch/diffusers import, transitively or otherwise.
  Importing `footprint` must leave `torch`/`diffusers` out of `sys.modules`.
- May import the private helpers `_background_index` / `_content_bbox` from
  `fakemon_forge.sprites` (verified safe — those don't pull the ML stack).
- Reuses the `sprites.py` `ValueError` convention for non-P-mode input.
- Output contract is exact: 16×16 RGBA, two-valued (opaque black / fully
  transparent).
- **Tests location (per `CLAUDE.md`):** because this module touches no ML code
  and fakes nothing via `sys.modules`, its tests go in a **regular test file
  `tests/test_footprint.py`** with **no `@pytest.mark.ml`**. They must **pass,
  not skip**, in the slim (no-torch) sandbox. The pre-existing ~21 `ml` tests
  still skip when torch is absent (expected).

### Reference metrics (the contract; official assets are NOT in the repo — copyrighted)

- Output size exactly **(16, 16)**.
- Venusaur reference: content spans ~**13×12 px** (~80% of canvas),
  **112 black px**, 3 claw bumps merged into the pad top.

These are targets the tunables aim at; tests assert the *format/colour*
contract, not these exact pixel counts.

## Tests: `tests/test_footprint.py` (regular suite, no `ml` marker)

Build small synthetic P-mode sprites in memory (or via `tmp_path`) — e.g.
`Image.new("P", (768, 768))` with a flat background index and a drawn blob near
the bottom, saved to `tmp_path`. Assert the numeric/format contract, **not** a
golden image:

- **Output size** — the written image is 16×16.
- **Colour contract** — every pixel's `(r, g, b, a)` is in
  `{(0, 0, 0, 255)}` or has `alpha == 0`, and nothing else appears.
- **`blank=True`** — all 256 pixels transparent, and prove no sprite read
  happens by passing a **nonexistent/irrelevant path** (call must not raise).
- **All-background sprite** — a P sprite that is entirely the background index →
  all-transparent output.
- **Clear foot blob** — a sprite with a distinct bottom blob → the output has at
  least one opaque-black pixel (a footprint was rendered).
- **Non-P-mode input** — an RGB image passed as the sprite → raises `ValueError`
  (`match="palette-mode"`).
- **Type variation** — a primary type with toes (`["Normal"]`) vs one with none
  (`["Water"]`): both outputs stay within the colour contract. **Do not** assert
  exact toe pixels (visual tunable).

Helper suggestion (mirrors `test_sprites.py` style): a small factory that
returns a P-mode 768 sprite with a chosen flat background index and an
optional rectangular/elliptical blob near the bottom, plus an assertion helper
that walks all 256 RGBA pixels and checks the two-value contract.

## Assumptions

Every item here is a default chosen for this headless slice (no live user), not
something already fixed by code/tests/docs — except where noted as
**verified**.

- **[picked]** Exact `type → toe` table contents (which types map to 3 claws /
  3 prongs / 4 toes / 0) and the toe-mark shapes (wedge/dot/prong) are
  implementation tunables chosen during implementation. Tests assert the
  format/colour contract, not specific toe pixels. The representative mapping in
  Behavior §7 is the starting point.
- **[picked]** Contour-override thresholds (what counts as "≥2 bumps" and a
  "mostly-genuine" contour) are tunables, expected to fire rarely; not asserted
  by tests.
- **[picked]** Contact-patch band = bottom **15%** of the content bbox; noise
  cutoff = **0.2%** of the band area; "the foot" = the **widest** surviving
  blob. These are the numbers named in the issue and adopted verbatim.
- **[picked]** Supersample factor **4×** for the pad; toes stamped at final
  16 px resolution with a **≥1 px** gap row. Factor is a tunable; the
  stamp-toes-at-final-res rule and the gap are contract-level (they exist to
  keep pixels distinct).
- **[picked]** `size_fraction` default **0.9**; interpreted as the fraction the
  **long axis** occupies. Binarization threshold that maps downsample greys to
  black-or-transparent is an implementation detail chosen so the two-value
  contract always holds.
- **[picked]** Empty `types` and unknown primary types both → 0 toes (plain
  pad).
- **[picked]** Missing sprite file on the non-blank path is left to
  `Image.open`'s `FileNotFoundError`; not caught or reshaped.
- **[verified]** Importing `fakemon_forge.sprites` at module top level pulls in
  neither torch nor diffusers, so `footprint.py` imports `_background_index` /
  `_content_bbox` from it rather than duplicating them.
- **[verified — `CLAUDE.md`]** Tests live in `tests/test_footprint.py`, carry no
  `ml` marker, and must pass (not skip) in the slim container.
- **[scope]** This slice ships only the module + its tests. No CLI, `main.py`,
  `writer.py`, `export_ini.py`, or spritesheet wiring — that is an explicitly
  later slice.
