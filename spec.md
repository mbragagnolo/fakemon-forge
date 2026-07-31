# Spec: img2img frame-2 candidate + CLI wiring (sprite_frame2.png / sprite_frame2_shiny.png)

## Summary

`fakemon-forge` is adding an Emerald-style **two-frame front animation**: every
front sprite becomes two frames. Two prior slices already landed the groundwork
in `fakemon_forge/sprites.py`:

- `quantize_to_reference(image, reference)` — quantize an image against a
  reference `P`-mode image's exact 16-colour palette.
- The pure-PIL frame-2 decision core: `build_frame2(frame1, candidate=None,
  low=0.02, high=0.30)` plus `procedural_squash`, `recenter_to_anchor`,
  `difference_ratio`. Given frame 1 (the finished 96×96 `P`-mode front sprite)
  and an optional candidate, it palette-locks + recenters the candidate, checks
  its pixel-difference against an acceptance band, and otherwise returns a
  bottom-anchored vertical squash of frame 1. It **always** returns a valid
  96×96 `P`-mode frame sharing frame 1's exact palette.

This slice (3/4 of #1) supplies the missing img2img **candidate** and wires the
whole thing into the CLI so every generated stage ships a `sprite_frame2.png`
(plus its shiny). It has two parts:

1. **`sprites.py`** — a new frame-2 generator `generate_frame2(...)` that runs
   the img2img pipeline from the finished front sprite at low strength with an
   animation tag, hands the **raw pre-quantization RGB** candidate to
   `build_frame2`, and saves the returned `P`-mode frame.
2. **`main.py`** — inside the per-stage loop, after the front sprite exists, a
   try/except block that calls `generate_frame2(...)` to write
   `sprite_frame2.png`, then `generate_shiny(...)` to write
   `sprite_frame2_shiny.png` — each wrapped in its own warn-and-continue block
   matching the existing back/shiny sibling blocks.

Because frame 2 shares frame 1's exact palette and `generate_shiny` is
name-keyed and rotates only the palette, `sprite_frame2_shiny.png` is
automatically consistent with `sprite_shiny.png` for free.

### Explicitly out of scope

- **Stitching** the two frames into a stacked double-height sprite sheet for ROM
  tooling — a separate future issue. This slice ships the standalone
  `sprite_frame2.png` / `sprite_frame2_shiny.png` only.
- **`.ini` / writer changes** — verified unnecessary. `export_ini` emits ROM
  pointer fields (`FrontAnimationTable=0`, `BackAnimTable=0`,
  `AnimDelayTable=0`), not sprite filenames; `writer.py` writes only
  `stats.json` / `entry.md`. Neither references the new files.
- **Tuning** the acceptance band / strength / animation tag — the prior slice's
  procedural fallback guarantees a valid frame regardless, so the defaults ship
  as-is and are tuned later by eye.

## Inputs

### New: `generate_frame2(prompt, types, front_sprite_path, output_path, *, pipeline, seed=None, strength=0.35, extra_tags=None)`

- `prompt: str` — the sprite prompt (same value passed to `generate_sprite` /
  `generate_sprite_img2img` for this stage; in `main` this is
  `stage["sprite_prompt"]`).
- `types: list[str]` — the stage's types, for type-tag prompting via
  `build_prompt` (same as the sibling generators).
- `front_sprite_path: str` — path to the finished front sprite (`sprite.png`),
  used **both** as the img2img init image **and** loaded as frame 1 (`P`-mode)
  for `build_frame2`.
- `output_path: str` — where to save the resulting `sprite_frame2.png`.
- `pipeline` (keyword-only) — an img2img pipeline (the same object `main`
  passes to the back-sprite `generate_sprite_img2img` call).
- `seed: int | None = None` (keyword-only) — RNG seed; `main` passes the stage
  seed so frame 2 reuses **frame 1's seed** for consistency.
- `strength: float = 0.35` (keyword-only) — low img2img strength; small
  deviation from the front sprite. **[picked]** default.
- `extra_tags: list[str] | None = None` (keyword-only) — animation tags;
  defaults to `["open mouth"]` when `None`. **[picked]** default.

### `main.py` per-stage additions

No new CLI arguments. The new block runs inside the existing
`for stage, stage_dir in zip(stages, stage_dirs)` loop, using values already in
scope:

- `stage["sprite_prompt"]`, `stage["types"]`, `stage["name"]`.
- `sprite_path` (the just-written `str(stage_dir / "sprite.png")`).
- `seed` (the stage's `random.randint(0, 2**32 - 1)`).
- `img2img_pipeline` — the same pipeline used for the back sprite (in the
  txt2img path this is `make_img2img_pipeline(pipeline)`; in the img2img path
  it is the loaded img2img pipeline itself).

## Outputs

- **`generate_frame2`** → returns `None`; side effect is writing a 96×96
  `P`-mode PNG at `output_path` whose palette is byte-for-byte equal to frame
  1's (guaranteed by `build_frame2`).
- **`main`** per stage, in addition to today's files, writes into `stage_dir`:
  - `sprite_frame2.png` — the frame-2 image (within the acceptance band vs
    `sprite.png`, sharing its exact 16-colour palette).
  - `sprite_frame2_shiny.png` — `generate_shiny` applied to `sprite_frame2.png`,
    keyed on `stage["name"]`; consistent with `sprite_shiny.png` by
    construction.

## Behavior

### `generate_frame2(...)` in `sprites.py`

1. Run the img2img pipeline from the **front sprite** (`front_sprite_path`) as
   the init image, using `build_prompt(prompt, types, extra_tags or ["open
   mouth"])`, `strength` (default `0.35`), and `_make_generator(seed)` — the
   same seed frame 1 used. This mirrors `generate_sprite_img2img`'s pipeline
   call (init image loaded as RGB, resized to `_GEN_SIZE`×`_GEN_SIZE` with
   `Image.LANCZOS`; `prompt_embeds`, `num_inference_steps=_NUM_STEPS`,
   `guidance_scale=_CFG_SCALE`, `strength` passed through).
2. Obtain the candidate as an **RGB image before adaptive quantization**. The
   existing `generate_sprite_img2img` bakes in `postprocess` (adaptive
   quantize) + save, which would double-quantize and produce an
   off-frame-1-palette image. **Preferred approach: factor out an internal
   helper** (e.g. `_run_img2img(prompt, types, image_path, *, pipeline,
   extra_tags, seed, strength) -> Image.Image`) that performs steps 1 and
   returns `result.images[0]` (the raw RGB pipeline output, no `postprocess`).
   Refactor `generate_sprite_img2img` to call this helper and then
   `postprocess` + save, so its existing behaviour/tests are unchanged.
   `generate_frame2` calls the same helper to get the raw candidate.
   (Alternative allowed by the issue: generate to a temp file and reload — but
   factoring out the raw-image step is preferred and avoids the extra
   quantize/round-trip.)
3. Load frame 1 from `front_sprite_path` as a `P`-mode image
   (`Image.open(front_sprite_path)` — the saved front sprite is already
   `P`-mode; do **not** convert).
4. Call `build_frame2(frame1, candidate)` (prior slice) with default `low` /
   `high`. This palette-locks + recenters the candidate, accepts it iff its
   difference from frame 1 is in `[low, high]`, else falls back to
   `procedural_squash(frame1)`.
5. Save the returned `P`-mode frame to `output_path` (PNG, inferred from
   extension, matching the other generators' `.save(output_path)`).

The pixel-art LoRA may ignore the semantic `"open mouth"` animation tag
entirely; that is exactly why `build_frame2`'s acceptance band + procedural
fallback guarantee a valid frame 2 regardless of what the pipeline returns.

### `main.py` wiring

Insert, inside the per-stage loop, **after** the front-sprite block succeeds
(the front-sprite block `continue`s on failure, so reaching this code means
`sprite_path` exists) and alongside the existing back / shiny / back-shiny
blocks. Suggested placement: after the front sprite, before or among the shiny
blocks (order is not load-bearing). Two nested-sibling `try/except`s, each
printing a `Warning: ...` to `stderr` and continuing — matching the existing
blocks exactly:

```
frame2_path = str(stage_dir / "sprite_frame2.png")
try:
    generate_frame2(
        stage["sprite_prompt"], stage["types"], sprite_path, frame2_path,
        pipeline=img2img_pipeline, seed=seed,
    )
except Exception as exc:
    print(f"Warning: frame 2 generation failed for {stage['name']}: {exc}",
          file=sys.stderr)

frame2_shiny_path = str(stage_dir / "sprite_frame2_shiny.png")
try:
    generate_shiny(frame2_path, stage["name"], frame2_shiny_path)
except Exception as exc:
    print(f"Warning: frame 2 shiny generation failed for {stage['name']}: {exc}",
          file=sys.stderr)
```

- `generate_frame2` must be added to the `from fakemon_forge.sprites import (…)`
  block in `main.py`.
- The frame-2 shiny reads `frame2_path`; if frame-2 generation failed and no
  file was written, `generate_shiny` will raise (file not found / non-`P`) and
  the shiny block's `except` will warn and continue — acceptable, consistent
  with how a failed back sprite feeds `generate_shiny(back_path, …)` today.

## Edge cases

- **Front sprite generation failed** → the existing front-sprite `except`
  `continue`s to the next stage; the frame-2 blocks never run for that stage.
- **img2img pipeline returns an off-band or garbage candidate** →
  `build_frame2` falls back to `procedural_squash(frame1)`; `generate_frame2`
  still writes a valid 96×96 `P`-mode `sprite_frame2.png`.
- **LoRA ignores the animation tag** → same as above; procedural fallback
  ensures a valid, in-band frame.
- **Frame-2 generation raises** (pipeline error, I/O) → warn-and-continue; the
  stage still gets its other sprites, and `export_ini` still runs.
- **Frame-2 shiny consistency** → since `sprite_frame2.png` shares
  `sprite.png`'s exact palette and `generate_shiny` hue-rotates only the
  palette keyed on `name`, `sprite_frame2_shiny.png` and `sprite_shiny.png` use
  the same rotated palette (consistent shinies) automatically.
- **Line mode (3 stages)** → the new blocks are inside the per-stage loop, so
  each of the 3 stages produces its own `sprite_frame2.png` /
  `sprite_frame2_shiny.png`.

## Errors

- `generate_frame2` surfaces exceptions to its caller (it does not swallow
  them); `main` wraps the call in try/except and warns. This matches every
  other generator (`generate_sprite`, `generate_sprite_img2img`,
  `generate_shiny`) and the existing warn-and-continue structure.
- `build_frame2` raises `ValueError` ("palette-mode") if `frame1.mode != "P"`;
  since `generate_frame2` loads the already-saved `P`-mode front sprite, this
  only fires on misuse and would be caught by `main`'s frame-2 `except`.
- No new `sys.exit` paths; the pipeline-load failure paths
  (`load_txt2img_pipeline` / `load_img2img_pipeline`) are unchanged.

## Constraints & dependencies

- `generate_frame2` lives in `fakemon_forge/sprites.py` next to the existing
  generators; it reuses `build_frame2`, `build_prompt`, `_make_generator`, and
  the module constants (`_GEN_SIZE`, `_NUM_STEPS`, `_CFG_SCALE`, `_SPRITE_SIZE`)
  rather than hard-coding.
- `generate_frame2` (and any factored-out `_run_img2img`) performs a
  function-local `import torch` via `_make_generator`, so **any test that calls
  it is an `ml` test** and belongs in `tests/test_sprites_ml.py` (or carries
  `@pytest.mark.ml`) — per `CLAUDE.md`'s test-slicing rule.
- `main.py` changes touch only import list + the per-stage loop; no new CLI
  args, no signature changes to `main`. Because `test_main.py` mocks the sprite
  functions, the `main` wiring is testable without torch.
- Reuse `img2img_pipeline` already in scope in `main` (do not load a new
  pipeline for frame 2).
- Preserve `generate_sprite_img2img`'s public behaviour when factoring out the
  raw-image helper — its existing `ml` tests (96×96, `P`-mode, PNG, single
  pipeline call, `strength`/`image`/`prompt_embeds` passthrough) must still
  pass unchanged.

## Tests

### ml (`tests/test_sprites_ml.py`, auto-skipped without torch)

Follow the existing `_fake_img2img_pipeline` / `_stub_encode_prompt` patterns.
Frame 1 fixtures should be real `P`-mode 96×96 images (build via `postprocess`
of an `_rgb_image`, as sprites are saved) so `build_frame2` gets a valid
palette-mode input.

- `generate_frame2` with a mock img2img pipeline creates the output file, and
  the saved image is 96×96, `P`-mode, PNG.
- The pipeline is invoked with the **low `strength`** (default `0.35`, i.e.
  `pipe.call_args.kwargs["strength"] == 0.35`).
- The pipeline is invoked with the animation **`extra_tags`** — assert the
  built prompt (via patched/inspected `_encode_prompt`, mirroring
  `test_img2img_encode_prompt_called`) includes the `"open mouth"` tag by
  default, and honours a caller-supplied `extra_tags`.
- The **same-seed** path is exercised (calling with `seed=<n>` runs without
  error / passes a generator through), mirroring how other tests drive
  `_make_generator`.
- The raw candidate is **not double-quantized**: e.g. `postprocess` is not
  applied to the frame-2 candidate before `build_frame2` (can be asserted by
  factoring `_run_img2img` and checking `generate_frame2` uses it, or by
  confirming the saved frame's palette equals frame 1's — the `build_frame2`
  contract).
- Existing `generate_sprite_img2img` tests continue to pass after the refactor
  (regression guard).

### light (`tests/test_main.py`, no torch — sprite fns mocked)

Add `generate_frame2` to the fixture patches (`ctx` and `ctx_line`), then:

- In the **txt2img path** and the **img2img path**, `main` calls
  `generate_frame2` once per stage with `output_path == str(stage_dir /
  "sprite_frame2.png")` and the stage's `sprite_path` as the front-sprite
  argument.
- `main` calls `generate_shiny` for the frame-2 shiny with `output_path ==
  str(stage_dir / "sprite_frame2_shiny.png")` and reading `frame2_path`.
- **Update existing call-count assertions** that this new call changes:
  - `test_txt2img_path_calls_generate_sprite` currently asserts
    `sprite_i2i.assert_called_once()` (back only) — frame 2 uses
    `generate_frame2` (separately patched), so the img2img count is unaffected,
    but verify.
  - Any `generate_shiny` call-count assertion must account for the **added
    frame-2 shiny call** (front shiny + back shiny + frame-2 shiny per stage).
    (Today no test asserts the total `generate_shiny` count directly, but if one
    is added or the count is used, it must include the frame-2 shiny.)
- Line mode: each of the 3 stages triggers a `generate_frame2` +
  frame-2-`generate_shiny` call (i.e. counts scale ×3), consistent with
  `test_line_mode_calls_sprite_three_times`.
- Frame-2 warn-and-continue: making `generate_frame2` raise prints a `Warning`
  to stderr and does not exit (mirroring `test_sprite_failure_warns_but_does_
  not_exit`, but note the front-sprite failure `continue`s whereas a frame-2
  failure must **not** skip the rest of the stage).

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** Function name `generate_frame2` and signature
  `generate_frame2(prompt, types, front_sprite_path, output_path, *, pipeline,
  seed=None, strength=0.35, extra_tags=None)` follow the issue's suggestion and
  the existing generator signatures.
- **[picked]** Default `strength=0.35` and `extra_tags=["open mouth"]` are the
  issue's starting suggestions and are eyeball placeholders; the LoRA may ignore
  the tag entirely, which the `build_frame2` acceptance band + procedural
  fallback (prior slice) cover. Tunable by eye; not yet confirmed by output.
- **[picked]** Frame 2 reuses **frame 1's seed** (the stage seed), matching how
  the back sprite already reuses the stage seed for front/back consistency.
- **[picked]** Preferred implementation factors an internal `_run_img2img`
  helper returning the **raw RGB** pipeline image (no `postprocess`), and
  `generate_sprite_img2img` is refactored to call it then `postprocess` + save —
  avoiding double-quantization while preserving its behaviour. The temp-file
  alternative is permitted but not chosen.
- **[picked]** Frame 2 uses `build_frame2`'s **default** `low=0.02` /
  `high=0.30` acceptance band (not re-exposed through `generate_frame2`).
- **[picked]** In `main`, frame 2 uses the **same `img2img_pipeline`** already
  in scope for the back sprite (not a new pipeline, not the txt2img pipeline).
- **[picked]** The frame-2 and frame-2-shiny steps are **two separate**
  warn-and-continue `try/except` blocks (like back/shiny today), so a frame-2
  generation failure still attempts nothing further for frame 2 but leaves the
  rest of the stage (and `export_ini`) intact.
- **[picked]** Warning messages read `Warning: frame 2 generation failed for
  {name}: {exc}` and `Warning: frame 2 shiny generation failed for {name}:
  {exc}`, mirroring the existing wording.
- **[picked]** Frame-2 blocks are placed after the successful front-sprite block
  within the per-stage loop; exact ordering relative to the back/shiny blocks is
  not load-bearing.
- **[confirmed]** `sprite.png` is saved `P`-mode (from `postprocess`'s
  `.quantize`), so loading it as frame 1 needs no conversion; `build_frame2`
  requires `P`-mode and guarantees the output shares its palette.
- **[confirmed]** `generate_shiny` rotates only the palette keyed on `name`, so
  same-palette sprites yield consistent shinies — `sprite_frame2_shiny.png`
  matches `sprite_shiny.png` for free.
- **[confirmed]** `export_ini` / `writer.py` need no changes — verified in the
  issue and by the modules' outputs (ROM pointer fields / `stats.json` +
  `entry.md`, no sprite filenames).
- **[confirmed]** Anything calling `generate_frame2` triggers a real
  `import torch` (via `_make_generator`) and is therefore an `ml` test per
  `CLAUDE.md`; the `main`-level wiring is torch-free because `test_main.py`
  mocks the sprite functions.
