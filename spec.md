# Spec: Lock the back sprite to the shared front-frame palette + cross-view shiny consistency

## Summary

`fakemon-forge` already produces, per stage, a front sprite `sprite.png` and a
second front-animation frame `sprite_frame2.png` that **share one exact
16-colour palette** (frame 2 is palette-locked to frame 1 via
`quantize_to_reference` / `build_frame2` in `fakemon_forge/sprites.py`), plus
shiny variants. The **back sprite** (`sprite_back.png`) is the last view still
carrying its **own adaptive palette**: it is produced by
`generate_sprite_img2img`, whose final step is `postprocess(candidate)` — an
*adaptive* 16-colour `quantize` that builds a fresh palette every call.

This slice (4/4 of #1) brings the back sprite into the **same shared palette**
as the two front frames, completing the authentic Gen-3 model of *one palette
for the whole sprite set, one rotated palette for the whole shiny set*. It has
two parts:

1. **`sprites.py`** — give the back-sprite generation path a way to re-quantize
   its img2img result against a reference `P`-mode image's exact palette
   (frame 1) instead of an adaptive palette, by adding an optional
   `reference_path` parameter to `generate_sprite_img2img`. When
   `reference_path` is given, the raw img2img candidate is locked with the
   existing `quantize_to_reference(candidate, reference)` rather than
   `postprocess`. When it is omitted, behaviour is byte-for-byte unchanged
   (adaptive `postprocess`), so the front-sprite img2img path and all existing
   tests are untouched.
2. **`main.py`** — pass `sprite.png` (frame 1, the just-written front sprite) as
   the back sprite's `reference_path`, so the back sprite locks to frame 1's
   palette regardless of which image seeded the img2img (the user's drawing in
   the img2img path, `sprite.png` in the txt2img path).

Because the back sprite then shares frame 1's exact palette, and
`generate_shiny` is name-keyed and **rotates only the palette** (preserving
achromatic entries), `sprite_back_shiny.png` — already derived via
`generate_shiny(back_path, …)` — automatically uses the **same rotated palette**
as `sprite_shiny.png` and `sprite_frame2_shiny.png`. All three views' shinies
become consistent for free; no shiny-path change is required.

### Explicitly out of scope

- **`.ini` / writer changes** — verified unnecessary (as in the prior slices).
  `export_ini` emits Gen 3 data fields, `writer.py` writes only
  `stats.json` / `entry.md`; neither references sprite filenames.
- **The img2img call itself** — the pipeline invocation (`_run_img2img`),
  `strength=0.65`, and `extra_tags=["backside"]` are unchanged. Only the
  post-generation quantization step gains a reference-locked branch.
- **Recentering / animation-band logic** — the back sprite is a *different view*,
  not an animation frame of the front, so `build_frame2` /
  `recenter_to_anchor` / the acceptance band do **not** apply to it. Only the
  palette is shared; geometry is whatever img2img produced.
- **Colour-fidelity guarantees** — the back sprite's colours may degrade when
  they land far from frame 1's 16 colours. Per the issue this is the authentic
  Gen-3 constraint and is accepted, not mitigated.

## Inputs

### Changed: `generate_sprite_img2img(prompt, types, image_path, output_path, *, pipeline, extra_tags=None, seed=None, strength=0.8, reference_path=None)`

All existing parameters are unchanged. One new keyword-only parameter is added
at the end (so existing positional/keyword calls are unaffected):

- `reference_path: str | None = None` (keyword-only) — path to a `P`-mode
  reference image whose exact 16-colour palette the generated sprite must adopt.
  When `None` (the default, and every current call except the new back-sprite
  one), the sprite is quantized adaptively via `postprocess` exactly as today.
  When set, the raw img2img candidate is locked to that palette via
  `quantize_to_reference`. **[picked]** name/shape — see Assumptions.

### `main.py` per-stage back-sprite call

No new CLI arguments. Inside the existing
`for stage, stage_dir in zip(stages, stage_dirs)` loop, the existing back-sprite
block gains one keyword argument:

- `reference_path = sprite_path` — i.e. `str(stage_dir / "sprite.png")`, the
  front sprite written earlier in the same loop iteration. This is **always**
  `sprite.png` (frame 1), independent of `init_image` (which is the user's
  `args.image` in the img2img path, or `sprite_path` in the txt2img path).

## Outputs

- **`generate_sprite_img2img` with `reference_path` set** → returns `None`; side
  effect is writing a 96×96 `P`-mode PNG at `output_path` whose palette is
  byte-for-byte equal to the reference image's palette (guaranteed by
  `quantize_to_reference`).
- **`generate_sprite_img2img` with `reference_path=None`** → unchanged: a 96×96
  `P`-mode PNG with an adaptive ≤16-colour palette (via `postprocess`).
- **`main`** per stage — the same set of files as today
  (`sprite.png`, `sprite_frame2.png`, `sprite_frame2_shiny.png`,
  `sprite_back.png`, `sprite_shiny.png`, `sprite_back_shiny.png`), but now:
  - `sprite_back.png` shares `sprite.png`'s exact 16-colour palette (was: its
    own adaptive palette).
  - `sprite_back_shiny.png` uses the same rotated palette as `sprite_shiny.png`
    and `sprite_frame2_shiny.png` (automatic consequence; no code change in the
    shiny blocks).

## Behavior

### `generate_sprite_img2img(...)` in `sprites.py`

1. Run the img2img pipeline exactly as today via the existing internal helper
   `_run_img2img(prompt, types, image_path, pipeline=…, extra_tags=…, seed=…,
   strength=…)`, obtaining the raw RGB candidate (`result.images[0]`). This step
   is unchanged.
2. Quantize the candidate:
   - If `reference_path is None`: `sprite = postprocess(candidate)` (adaptive
     palette) — unchanged from today.
   - Else: open the reference as a `P`-mode image
     (`Image.open(reference_path)` — the saved front sprite is already `P`-mode;
     do **not** convert) and `sprite = quantize_to_reference(candidate,
     reference)`. `quantize_to_reference` already performs the same
     resize-to-96×96 + colour/contrast enhance pre-steps as `postprocess`, then
     `.quantize(palette=reference)`, so both branches feed identical input to
     quantization and differ only in adaptive-vs-fixed palette.
3. `sprite.save(output_path)` (PNG inferred from extension) — unchanged.

The choice is a single branch on `reference_path`; `_run_img2img`,
`postprocess`, `quantize_to_reference`, and the module constants are reused
rather than duplicated.

### `main.py` wiring

The existing back-sprite block becomes:

```
back_path = str(stage_dir / "sprite_back.png")
try:
    init_image = args.image if args.image else sprite_path
    generate_sprite_img2img(
        stage["sprite_prompt"], stage["types"], init_image, back_path,
        pipeline=img2img_pipeline, extra_tags=["backside"], seed=seed,
        strength=0.65, reference_path=sprite_path,
    )
except Exception as exc:
    print(
        f"Warning: back sprite generation failed for {stage['name']}: {exc}",
        file=sys.stderr,
    )
```

Only `reference_path=sprite_path` is added. The block still reaches this code
only after the front-sprite block succeeded (that block `continue`s on failure),
so `sprite_path` names an existing `P`-mode `sprite.png`. The back-shiny block
(`generate_shiny(back_path, stage["name"], back_shiny_path)`) is **unchanged** —
it now inherits the shared palette automatically.

## Edge cases

- **Front sprite generation failed** → the front-sprite `except` `continue`s to
  the next stage; the back-sprite block (and its `reference_path`) never runs
  for that stage, so there is never a missing/absent reference.
- **img2img returns colours far from frame 1's palette** →
  `quantize_to_reference` maps each pixel to the nearest of frame 1's 16 colours;
  the back sprite may look slightly off-palette / posterized. **Accepted** — this
  is the authentic Gen-3 shared-palette constraint, not a bug.
- **Back sprite content differs from the front** (it is a rear view) → only the
  palette is shared, not geometry; no recentering/animation-band logic is applied
  (that is `build_frame2`'s job for frame 2, not for the back view).
- **Cross-view shiny consistency** → `sprite.png`, `sprite_frame2.png`, and
  `sprite_back.png` now share one palette; `generate_shiny` rotates only the
  palette keyed on `name`, so `sprite_shiny.png`, `sprite_frame2_shiny.png`, and
  `sprite_back_shiny.png` share one rotated palette automatically.
- **Line mode (3 stages)** → the back-sprite block is inside the per-stage loop;
  each stage locks its own back sprite to its own `sprite.png`, and each stage's
  three shinies stay mutually consistent within that stage.
- **`reference_path=None` callers** (the front-sprite img2img call, and any other
  existing caller) → behaviour is identical to today (adaptive `postprocess`).

## Errors

- `generate_sprite_img2img` surfaces exceptions to its caller (it does not
  swallow them); `main` wraps the back-sprite call in the existing try/except and
  warns `Warning: back sprite generation failed for {name}: {exc}` — unchanged
  wording and structure.
- `quantize_to_reference` raises `ValueError` ("palette-mode reference image") if
  the reference is not `P`-mode. Because `main` always passes the already-saved
  `P`-mode `sprite.png`, this only fires on misuse and would be caught by the
  back-sprite `except`.
- A missing `reference_path` file (e.g. `sprite.png` never written) would raise
  in `Image.open`; this cannot happen after a successful front-sprite block, and
  if it somehow did it is caught by the back-sprite `except` (warn-and-continue).
- No new `sys.exit` paths; pipeline-load failure paths are unchanged.

## Constraints & dependencies

- The change lives in `fakemon_forge/sprites.py` (`generate_sprite_img2img`) and
  `fakemon_forge/main.py` (one added kwarg). It reuses the existing
  `quantize_to_reference`, `_run_img2img`, `postprocess`, and module constants;
  nothing is hard-coded or duplicated.
- `generate_sprite_img2img` performs a function-local `import torch` (via
  `_run_img2img` → `_make_generator`), so **any test that calls it is an `ml`
  test** and belongs in `tests/test_sprites_ml.py` (or carries
  `@pytest.mark.ml`), per `CLAUDE.md`'s test-slicing rule. The pure
  palette-lock/shiny assertions that go in `tests/test_sprites.py` must therefore
  exercise `quantize_to_reference` / `generate_shiny` **directly**, not through
  `generate_sprite_img2img`.
- `main.py` changes touch only the back-sprite call (one kwarg); no import
  changes, no new CLI args, no signature change to `main`. Because
  `test_main.py` mocks the sprite functions, the `main` wiring is testable
  without torch.
- **Backward compatibility:** the new parameter defaults to `None`, so all
  current `generate_sprite_img2img` calls and their `ml` tests (96×96, `P`-mode,
  PNG, single pipeline call, `strength`/`image`/`prompt_embeds` passthrough)
  must continue to pass unchanged. Only the new back-sprite call passes
  `reference_path`.
- Frame 1 (`sprite.png`) is the canonical palette source for the whole set
  (front frame 1, front frame 2, and back all lock to it). The front sprite
  itself is never reference-locked (it *defines* the palette).

## Tests

### light (`tests/test_sprites.py`, torch-free)

These exercise the shared-palette lock and shiny consistency **without** calling
`generate_sprite_img2img` (which would trigger `import torch`). Follow the
existing `postprocess` / `quantize_to_reference` / helper patterns.

- **Back-sprite palette lock**: given a back RGB image and a `P`-mode reference
  frame (build via `postprocess(_rgb_image())` / `postprocess(_noisy_image())`),
  `quantize_to_reference(back_rgb, reference)` yields a `P`-mode 96×96 image
  whose `getpalette()` equals the reference's exactly. (This is the pure core of
  the back-sprite lock; `quantize_to_reference` is already well-covered, so this
  test frames it as the back-sprite scenario and asserts palette equality.)
- **Cross-view shiny consistency**: build three `P`-mode images that share one
  palette (stand-ins for frame 1 / frame 2 / back — e.g. quantize three
  different RGB inputs against one reference so all three share its palette),
  save each, run `generate_shiny(path, name, out_path)` on each with the **same
  `name`**, reload the three outputs, and assert their three `getpalette()`
  results are **identical** to one another. (Optionally also assert each shiny
  palette differs from the shared original, i.e. rotation happened.)

### ml (`tests/test_sprites_ml.py`, auto-skipped without torch)

Follow the existing `_fake_img2img_pipeline` / `_stub_encode_prompt` patterns;
build the reference as a real `P`-mode 96×96 file (e.g. via `_frame1_file` /
`postprocess(_rgb_image())`, as sprites are saved).

- `generate_sprite_img2img(..., reference_path=<P-mode frame path>)` with a mock
  img2img pipeline writes an output file that is **`P`-mode** and whose
  `getpalette()` **equals the reference's** (proves the back sprite adopts the
  shared palette rather than an adaptive one).
- The saved reference-locked sprite is still 96×96 and PNG.
- **Regression**: `generate_sprite_img2img` **without** `reference_path`
  continues to produce a `P`-mode 96×96 PNG via adaptive `postprocess` (existing
  tests suffice; add one asserting the two branches diverge only in palette if
  desired — e.g. locked output's palette equals the reference while the
  unlocked output's need not).
- The pipeline is still invoked **exactly once** and with the unchanged
  `strength` / `image` / `prompt_embeds` / `extra_tags` passthrough when
  `reference_path` is supplied (the reference only affects post-quantization).

### light (`tests/test_main.py`, no torch — sprite fns mocked)

- The back-sprite `generate_sprite_img2img` call receives
  `reference_path == str(stage_dir / "sprite.png")` (frame 1), in **both** the
  txt2img path and the img2img path. In the img2img path, assert the back call's
  positional `image_path` (init) is `args.image` while its `reference_path` is
  `sprite.png` — i.e. the reference is frame 1 even though the init image is the
  user's drawing.
- The existing back-sprite assertions still hold:
  `extra_tags == ["backside"]`, `strength == 0.65`, and the img2img-path call
  count (front + back). Distinguish the front call (`reference_path` absent/`None`)
  from the back call (`reference_path == sprite.png`).
- The back-shiny wiring is unchanged (`generate_shiny(back_path, name,
  back_shiny_path)`); the existing shiny-count assertions
  (`test_generate_shiny_called_three_times_per_stage`,
  `test_line_mode_frame2_called_three_times`) remain valid, since no shiny call
  was added or removed — only the back sprite's palette changed.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** The lock is added as an optional `reference_path: str = None`
  keyword parameter on the **existing** `generate_sprite_img2img`, rather than a
  new dedicated `generate_back_sprite` function. Rationale: it is the minimal,
  lowest-risk change (existing `reference_path=None` callers and their tests are
  untouched), keeps the img2img call in one place, and matches how the front
  frames were locked (via `quantize_to_reference`). The issue explicitly permits
  either approach. Note: an unused `generate_back_sprite` (txt2img-based) already
  exists in `sprites.py` but is **not** the back-sprite path `main` uses (`main`
  calls `generate_sprite_img2img` for the back sprite); it is left untouched to
  avoid scope creep. **[confirmed]** that `generate_back_sprite` is currently
  unused by `main.py`.
- **[picked]** The parameter is a **path** (`reference_path`) rather than a
  pre-loaded `Image`, matching how `main` already threads file paths
  (`front_sprite_path`, `image_path`) and letting the function own the
  `Image.open`. The issue allowed `reference_path`/`reference`; path chosen for
  consistency.
- **[picked]** The parameter is placed **last** in the keyword-only signature and
  defaults to `None`, preserving every existing call site and test.
- **[picked]** When `reference_path` is set, the reference is opened without a
  mode conversion (the saved front sprite is already `P`-mode); a non-`P`-mode
  reference is left to raise via `quantize_to_reference` (caught by `main`'s
  back-sprite `except`).
- **[confirmed]** Frame 1 (`sprite.png`) is the canonical palette source: it is
  generated first in the loop and is saved `P`-mode by `postprocess`'s
  `.quantize`; front frame 2 already locks to it, and this slice locks the back
  to it too.
- **[confirmed]** The reference is always `sprite_path` (frame 1), independent of
  the img2img init image — in the img2img path the init is the user's drawing
  (`args.image`) while the palette reference must still be frame 1.
- **[confirmed]** `quantize_to_reference` already mirrors `postprocess`'s
  resize + colour/contrast pre-steps, so switching only the palette (adaptive →
  fixed reference) is the sole behavioural difference between the branches; it
  does not mutate its inputs.
- **[confirmed]** `generate_shiny` rotates only the palette keyed on `name` and
  preserves achromatic entries, so three views sharing one palette yield three
  identical rotated shiny palettes — `sprite_back_shiny.png` is consistent with
  `sprite_shiny.png` / `sprite_frame2_shiny.png` with **no** change to the shiny
  blocks (the back shiny is already `generate_shiny(back_path, …)`).
- **[confirmed]** `export_ini` / `writer.py` need no changes — they reference no
  sprite files (Gen 3 data fields / `stats.json` + `entry.md`).
- **[confirmed]** Anything calling `generate_sprite_img2img` triggers a real
  `import torch` (via `_run_img2img` → `_make_generator`) and is therefore an
  `ml` test; the pure palette/shiny assertions in `test_sprites.py` must call
  `quantize_to_reference` / `generate_shiny` directly, and the `main`-level
  wiring is torch-free because `test_main.py` mocks the sprite functions.
