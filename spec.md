# Spec: `--image` mode front+back sprite pair

## Summary

Issue #69, slice 8/9 of #61. Today `--image` mode calls `generate_sprite_img2img`
once against the child's drawing to produce a single front sprite; no back
sprite is produced at all (an earlier slice in this sequence deleted the old
img2img backside chain and did not replace it — see the regression flagged in
`tests/test_main.py::test_img2img_path_produces_no_back_sprite_call`). Separately,
an earlier slice gave txt2img mode (`--description` with no `--image`) a single
`generate_sprite_pair` call that renders one 1536x768 front+back canvas, splits
it, adaptively quantizes the front, and locks the back to the front's exact
palette (`fakemon_forge/sprites.py:624`, wired into `fakemon_forge/main.py:106`).

This slice brings `--image` mode onto that same shared canvas/split/postprocess
path. Two candidate designs are in play, gated on an empirical check that only
the real LoRA pipeline (GPU, on the host — unavailable in this sandbox; see
Assumptions) can answer:

- **Approach A — doubled init canvas.** Paste the child's (resized) drawing onto
  both halves of a fresh 1536x768 canvas, run that canvas through the SDXL
  img2img pipeline at `strength=0.8`, and feed the result through the exact same
  `_split_front_back_with_retry` / `postprocess` / `quantize_to_reference` path
  `generate_sprite_pair` already uses.
- **Approach B — vision-description-only fallback.** If Approach A's back half
  doesn't come out as a genuine rear view (a real risk: the model has only ever
  seen a front view in the init image), drop img2img entirely for `--image` mode.
  `--image` becomes purely an input to the existing `describe_image` vision step;
  the rest of the pipeline — including sprite generation — runs exactly like
  text-only generation, i.e. one txt2img `generate_sprite_pair` call built from
  the described prompt.

Both are fully specified below so implementation can proceed the moment the
validation step (this slice's first task, per the issue) picks one. **Do not
implement both** — the validation result determines which one lands, and the
choice must be stated in the PR description.

"Done" means: whichever approach the validation supports, `--image` mode
produces a front+back pair (`sprite.png` and `sprite_back.png`, same as
text-only runs) via `generate_sprite_pair`-equivalent logic, with no bespoke
split/post logic of its own in `main.py`.

## Inputs

- `args.image: str` — path to the child's drawing (jpg/png), already validated
  by `cli.validate_args` (exists, correct extension) before `main()` reaches the
  sprite-generation block. Unchanged by this slice.
- `args.description: str | None` — optional free-text description, combined with
  the vision output exactly as today (`fakemon_forge/main.py:56-61`). Unchanged.
- Per-stage `stage["sprite_prompt"]` / `stage["types"]` from `generate_fakemon`,
  as consumed by `generate_sprite_pair` today. Unchanged in shape.
- A `seed` (`random.randint(0, 2**32 - 1)`, per stage) exactly as today.
- (Approach A only) the loaded img2img `pipeline` (`load_img2img_pipeline()`),
  exactly as `--image` mode loads today.
- (Approach B only) the loaded txt2img `pipeline` / derived `img2img_pipeline`
  (`load_txt2img_pipeline()` + `make_img2img_pipeline(pipeline)`), exactly as
  `--description`-only mode loads today — now used unconditionally.

## Outputs

- `sprite.png` — front sprite, native 768x768, `P`-mode, Gen-3 16-colour
  contract (adaptive palette via `postprocess`). Same contract as today's
  txt2img front output.
- `sprite_back.png` — back sprite, native 768x768, `P`-mode, sharing
  `sprite.png`'s exact palette (`quantize_to_reference`). **New for `--image`
  mode** — today this file is never written for `--image` runs.
  - If the split canvas's back half comes back empty/background-only even
    after the palette lock, it is skipped with a `stderr` warning and
    `sprite_back.png` is not written — mirroring `generate_sprite_pair`'s
    existing degrade-gracefully contract. This is not a new failure mode, just
    the existing one now reachable from `--image` mode too.
- Every downstream consumer of `sprite_back.png` that already exists for
  txt2img mode (`sprite_back_shiny.png` via `generate_shiny`, the
  `sprite_back.png` cell in `stitch_spritesheet`) starts working for `--image`
  mode too, with no changes to those functions — they already tolerate a
  missing back file.
- No change to `sprite_chibi.png`, `sprite_frame2.png`, `sprite_frame2_shiny.png`,
  `sprite_shiny.png`, `sprite_small.png`, `footprint.png`: none of these read
  from the new front+back call in a way this slice touches.

## Behavior

### Decision gate (implementation's first task, not part of this spec's output)

Before wiring either approach into `main.py`, generate several samples with
Approach A's doubled-canvas img2img call against representative drawings /
prompts using the real pipeline (host, GPU) and inspect the back half of each
split canvas. Approach A is adopted only if a clear majority of samples produce
a back half that reads as a genuine rear-facing view (distinguishable pose/
silhouette from the front, not a mirrored or duplicated front, not visibly
garbled). Otherwise, fall back to Approach B. The outcome (which approach, and
a one-line reason) must be stated in the PR description — this is a recorded
decision from the parent issue, not a free choice to relitigate at
implementation time.

### Approach A — doubled init canvas (if validation passes)

New function in `fakemon_forge/sprites.py`, named to match the existing
`generate_sprite_pair` / `generate_sprite_img2img` convention —
`generate_sprite_pair_img2img(prompt, types, image_path, front_output_path, back_output_path, *, pipeline, seed=None, strength=0.8)`:

1. Build the init canvas: load `image_path` via `Image.open(...).convert("RGB")`,
   resize to `(_GEN_SIZE, _GEN_SIZE)` with `Image.LANCZOS` — identical to
   `_run_img2img`'s existing resize of the init image — then paste that single
   resized image at `(0, 0)` and again at `(_GEN_SIZE, 0)` on a fresh
   `_PAIR_WIDTH x _GEN_SIZE` (1536x768) RGB canvas. Both halves are byte-identical
   copies of the same resized drawing; no augmentation/jitter between them.
2. Run the img2img pipeline once against that canvas: `prompt=build_prompt(prompt)`,
   `negative_prompt=_NEGATIVE_PROMPT`, `image=<canvas>`, `strength=strength`
   (default `0.8`, matching the issue's "existing strength"),
   `num_inference_steps=_NUM_STEPS`, `guidance_scale=_CFG_SCALE`,
   `generator=_make_generator(seed)`. No `width=`/`height=` kwargs — mirrors
   every other img2img call site, which infers size from the init image.
3. Split + reroll-once + naive-fallback: reuse `_split_front_back_with_retry`
   unchanged. Its `regenerate` callable re-invokes the pipeline against the
   *same* built canvas with `seed + 1` (or `None` if unseeded) — the canvas
   itself is deterministic and only needs building once; a different seed still
   changes the img2img result via the diffusion generator's noise.
4. Postprocess: front half through `postprocess` (adaptive palette) and saved;
   back half through `quantize_to_reference(back_raw, front)` and saved, or
   skipped-with-warning if empty — byte-for-byte the same two calls
   `generate_sprite_pair` already makes.

`fakemon_forge/main.py` changes: the `if args.image: generate_sprite_img2img(...)
else: generate_sprite_pair(...)` branch (`main.py:100-109`) becomes a call to
`generate_sprite_pair_img2img(stage["sprite_prompt"], stage["types"], args.image, sprite_path, back_path, pipeline=pipeline, seed=seed)` in the `if args.image` arm. The
pipeline-loading branch (`main.py:70-75`) is unchanged — `--image` mode still
loads only the img2img pipeline. The chibi and frame2 calls, which already
run `generate_sprite_img2img` against `sprite_path` (not the raw drawing),
are untouched by this change.

### Approach B — vision-description-only fallback (if validation fails)

No new functions in `sprites.py`. `fakemon_forge/main.py` collapses both of its
`--image`-conditional branches to their `else` arm unconditionally:

- Pipeline loading (`main.py:70-75`): always `pipeline = load_txt2img_pipeline()`
  then `img2img_pipeline = make_img2img_pipeline(pipeline)`, regardless of
  `args.image`. `load_img2img_pipeline` becomes unused for the main sprite call
  (still fine to keep imported if the chibi/frame2 img2img calls need it — they
  don't; they already run on `img2img_pipeline`, i.e. the txt2img-derived one).
- Sprite generation (`main.py:100-109`): always
  `generate_sprite_pair(stage["sprite_prompt"], stage["types"], sprite_path, back_path, pipeline=pipeline, seed=seed)`, regardless of `args.image`.
- `describe_image(args.image, client=client)` keeps running exactly as today
  (`main.py:56-58`) and keeps feeding into `combined` / `generate_fakemon` exactly
  as today — the LLM-authored `sprite_prompt` is the only channel through which
  the drawing's content reaches the sprite renderer. The raw pixels of the
  drawing are never passed to any image-generation pipeline in this approach.
- `generate_sprite_img2img` remains used for the chibi enhancement (init image
  `sprite.png`) exactly as today, for both modes.

## Edge cases

- **`--image` with no `--description`.** Already valid per `cli.validate_args`
  (`test_validate_passes_with_image_only`). `combined` is vision-description-only
  in both approaches (unchanged from today's construction) — no special-casing
  needed here beyond what already exists.
- **Non-square drawings.** Both approaches that touch the raw drawing
  (Approach A's canvas build) stretch it to a 768x768 square via the same
  `Image.LANCZOS` resize `_run_img2img` already uses for the single-view
  img2img path — not a letterbox/pad. This preserves today's existing
  distortion behavior rather than introducing new letterboxing logic.
- **Split failure on the doubled canvas (Approach A).** Identical to
  `generate_sprite_pair`'s existing contract: reroll once with `seed + 1`, then
  fall back to a naive midline split with a `stderr` warning. Never raises for
  this reason.
- **Back half empty/background-only (Approach A).** Identical to
  `generate_sprite_pair`: skip saving `sprite_back.png` with a `stderr` warning;
  the stage continues (downstream steps already tolerate a missing back file).
- **Line mode (`--mode line`) with `--image`.** The image is the same drawing
  reused for every stage's sprite call, exactly as today (each stage still gets
  its own random `seed`). No per-stage image variation is introduced.

## Errors

- A genuine pipeline failure (e.g. `RuntimeError` from the mocked/real
  pipeline call) propagates out of `generate_sprite_pair_img2img` (Approach A)
  exactly as `generate_sprite_pair` and `generate_sprite_img2img` already do —
  `main.py`'s existing per-stage `try/except` around the sprite block
  (`main.py:99-115`) already prints a `Warning: sprite generation failed for
  {name}` message and `continue`s to the next stage; unchanged by this slice.
- No new error paths are introduced by either approach. `cli.validate_args`'s
  existing `--image` file-existence/extension checks are unaffected.

## Constraints & dependencies

- Same SDXL img2img pipeline, LoRA, and constants (`_GEN_SIZE`, `_PAIR_WIDTH`,
  `_NUM_STEPS`, `_CFG_SCALE`, `_NEGATIVE_PROMPT`, `strength=0.8`) as the rest of
  `sprites.py` — no new tunables introduced.
- Approach A depends on the img2img pipeline accepting a `1536x768` init image
  (double the single-view `768x768` init `_run_img2img` uses today) —
  unverified in this sandbox (no torch/diffusers/GPU here; the full ML stack
  runs on the host per `CLAUDE.md`).
- Tests: extend `tests/test_sprites_ml.py` (`ml`-marked, torch-requiring,
  auto-skipped here) with pipeline-call-shape assertions for whichever function
  is added/changed, mirroring the existing `generate_sprite_pair` test block
  (1536x768-equivalent init image, prompt/negative_prompt passthrough, reroll-
  once-uses-seed-plus-one, naive-fallback-warns, back-shares-front-palette,
  pipeline-error-propagates). Extend `tests/test_main.py` /
  `tests/test_cli.py` with mocked-pipeline call-shape assertions:
  - Approach A: update `test_img2img_path_calls_generate_sprite_img2img` and
    `test_img2img_path_produces_no_back_sprite_call` (the latter's premise —
    "no replacement mechanism is implemented yet" — is exactly what this slice
    fixes; both need rewriting, not just extending) to assert
    `generate_sprite_pair_img2img` is called once with `(front_path, back_path)`
    and that `generate_sprite_img2img` is called only for the chibi enhancement.
  - Approach B: assert `load_txt2img_pipeline` (not `load_img2img_pipeline`) is
    called for `--image` runs, `generate_sprite_pair` (not
    `generate_sprite_img2img`) is called once for the front+back pair, and that
    `describe_image`'s mocked return value reaches the same
    `generate_sprite_pair` call path `args.description`-only mode already
    exercises (i.e. via `stage["sprite_prompt"]`, since the LLM call already sits
    between vision output and sprite prompt — assert the existing `combined`
    construction test coverage rather than a new direct-passthrough claim).

## Assumptions

- **No GPU/ML stack in this sandbox** (`CLAUDE.md`; confirmed here — `torch`
  and `diffusers` are both unimportable and no LoRA weights are present under
  `models/`). The issue's required first task — empirically validating whether
  the doubled canvas yields a genuine back view — cannot be executed as part of
  writing this spec. This spec therefore fully designs both branches instead of
  picking one; the implementation phase (on the host, where the ML stack and
  GPU exist per `CLAUDE.md`) must run the validation first and implement only
  the winning branch, deleting/never building the other.
- **Validation protocol left to implementer discretion on specifics** (sample
  count, exact prompts/drawings used) since it is a qualitative visual judgment
  call, not a value obtainable by reading code. A default of ~5 varied samples
  with a "clear majority must read as a genuine back view" bar is suggested
  above as a reasonable, falsifiable default rather than an open-ended "look at
  it and decide."
- **Non-square drawing resize stays a stretch-to-square** (`Image.LANCZOS` to
  `(768, 768)`), not a new letterbox/pad step, to match `_run_img2img`'s
  existing single-view behavior exactly. Introducing letterboxing here would be
  a bigger behavior change than this slice's scope ("the only thing that should
  differ ... is how the init canvas is built").
- **Both drawing pastes are byte-identical** (no per-half tag/prompt variation,
  no mirroring/flipping one half) — the issue's plain-language description
  ("the same drawing appears twice, side by side") and "same prompt shape used
  elsewhere" are read literally: one prompt, one canvas, two identical pastes.
  A more elaborate scheme (e.g. tagging the right half "backside") is exactly
  the kind of bespoke logic the issue says `--image` mode should *not* need.
- **`generate_sprite_pair_img2img` takes no `extra_tags` parameter**, mirroring
  `generate_sprite_pair`'s existing signature (which also lacks one) rather than
  `generate_sprite_img2img`'s (which has one) — the front+back pair call has
  never taken caller-supplied tags in this codebase.
- **Scope**: this slice touches only the primary front+back sprite call for
  `--image` mode. It does not revisit the chibi caricature pass, frame 2, or
  shiny derivation, all of which already tolerate today's file layout and
  require no changes under either approach.
