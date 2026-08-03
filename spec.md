# Spec: Verify the chibi icon enhancement against the SDXL img2img pipeline

## Summary

Slices 1-6 of #61 swapped `fakemon_forge/sprites.py`'s generation pipelines
from SD1.5 to SDXL: `load_img2img_pipeline()` / `make_img2img_pipeline()` now
construct a `diffusers.StableDiffusionXLImg2ImgPipeline`, and `build_prompt`
was rewritten to a plain string formula (`"gen3, {sprite_prompt}[, extra
tags], white background"`) instead of the old `_TYPE_TAGS`-based vocabulary.
`fakemon_forge/main.py`'s per-stage chibi enhancement — an optional img2img
pass (`generate_sprite_img2img(..., pipeline=img2img_pipeline,
extra_tags=_CHIBI_TAGS, seed=seed)`, `_CHIBI_TAGS = ["chibi", "big head",
"small body"]`) that feeds a caricatured render into `generate_icon` instead
of the plain front sprite, falling back to the plain sprite on any exception —
was written against the old SD1.5 pipeline and never re-verified against the
new SDXL one.

This slice is **verification-only**: having read `build_prompt` and the
`generate_sprite_img2img` call chain (`generate_sprite_img2img` →
`_run_img2img` → `_run_img2img_on_image` → `build_prompt(prompt,
extra_tags)`), `extra_tags` support was **not** dropped by the SDXL rewrite —
it is threaded through unchanged, and `tests/test_sprites.py` already has a
passing unit test (`test_build_prompt_multiple_extra_tags_joined_with_comma`)
asserting `build_prompt`'s exact-`_CHIBI_TAGS`-shaped output. **No production
code change is required.** The only work this slice does is add a test that
exercises the chibi call path (`generate_sprite_img2img` with
`extra_tags=_CHIBI_TAGS`) against an SDXL-img2img-shaped pipeline mock, so the
combination is confirmed by test rather than by code reading alone.

### Explicitly out of scope

- **Any change to `main.py`'s chibi call site or `_CHIBI_TAGS`** — both are
  confirmed correct as-is (see Assumptions/Behavior).
- **Any change to `build_prompt` / `generate_sprite_img2img` / `_run_img2img`**
  — `extra_tags` passthrough already works; this slice adds coverage, not code.
- **Tuning the chibi tags, `strength`, or judging whether the SDXL LoRA
  actually produces a recognisable caricature** — per the original chibi spec
  (`keep/aab47792`) and this issue, the LoRA was never trained with
  chibi/caricature tags in mind, and whether the *rendered image* looks like a
  caricature is a GPU-only, human-eyeball judgement out of scope for an
  automated test. The exception-fallback to the plain (now
  `k_centroid`-downscaled) sprite is the documented, sufficient safety net for
  a poor render; this slice does not add new special-casing for that outcome.
- **`icon.py`'s `k_centroid` downscale** — unchanged, already shipped by an
  earlier slice in this sequence.

## Inputs

No new inputs. Existing values already in scope, unchanged:

- `_CHIBI_TAGS = ["chibi", "big head", "small body"]` (`main.py`) — the
  `extra_tags` passed to the chibi render.
- The chibi call site in `main.py`'s per-stage loop:
  `generate_sprite_img2img(stage["sprite_prompt"], stage["types"],
  sprite_path, chibi_path, pipeline=img2img_pipeline, extra_tags=_CHIBI_TAGS,
  seed=seed)`, where `img2img_pipeline` is, by this point in the sequence,
  either `load_img2img_pipeline()`'s return value (`--image` path) or
  `make_img2img_pipeline(pipeline)`'s return value (description-only path) —
  both real `StableDiffusionXLImg2ImgPipeline` instances at runtime.
- `build_prompt(sprite_prompt, extra_tags=None)` (`sprites.py`) — already
  accepts and folds `extra_tags` into the plain prompt string; unchanged.

## Outputs

No new outputs. The behavior under test remains:

- On success: `sprite_chibi.png` — a 768x768 `P`-mode PNG (adaptive palette,
  `reference_path` omitted) produced from `sprite.png` via the SDXL img2img
  pipeline with the chibi tags folded into the prompt; `generate_icon` builds
  `sprite_small.png` from it.
- On the chibi call raising: `icon_source` falls back to `sprite_path`
  (`sprite.png`); `generate_icon` builds `sprite_small.png` from the plain
  sprite instead. `sprite_chibi.png` may be absent.

## Behavior

Confirmed by reading the code (no change needed):

1. `generate_sprite_img2img(prompt, types, image_path, output_path, *,
   pipeline, extra_tags=None, seed=None, strength=0.8, reference_path=None)`
   calls `_run_img2img(prompt, types, image_path, pipeline=pipeline,
   extra_tags=extra_tags, seed=seed, strength=strength)`, which calls
   `_run_img2img_on_image(..., extra_tags=extra_tags, ...)`, which calls
   `pipeline(prompt=build_prompt(prompt, extra_tags), negative_prompt=...,
   image=init, num_inference_steps=..., guidance_scale=..., generator=...,
   strength=strength)`. `extra_tags` is threaded through every hop unchanged —
   this is generic plumbing shared by every `generate_sprite_img2img` caller
   (chibi, back-sprite `["backside"]`, frame 2's `["open mouth"]`), not
   special-cased per caller.
2. `build_prompt("...", ["chibi", "big head", "small body"])` returns
   `"gen3, ..., chibi, big head, small body, white background"` — verified by
   the existing `test_build_prompt_multiple_extra_tags_joined_with_comma`
   (which uses exactly this tag list, written against `_CHIBI_TAGS`'s literal
   values).
3. The chibi call in `main.py` is unchanged from the prior (SD1.5-targeting)
   slice: same tags, same `pipeline=img2img_pipeline` (now SDXL), same
   `seed=seed`, `reference_path` omitted (own adaptive palette, matching the
   original chibi spec's choice — the chibi render need not share the front
   sprite's palette).
4. The only thing that changed under this call site since it was written is
   *what* `img2img_pipeline` is a runtime instance of (SD1.5 -> SDXL
   `Img2ImgPipeline`) and *what* `build_prompt` does internally (old
   `_TYPE_TAGS` formula -> plain string formula) — neither changes the
   `extra_tags` contract `generate_sprite_img2img` and `main.py` rely on.

## Edge cases

- **Chibi img2img raises** (pipeline crash, torch/OOM, bad init, the LoRA
  simply not understanding "chibi"/"big head"/"small body" and producing a
  garbage or unrecognisable image that some downstream step chokes on, etc.)
  -> caught by the existing inner `try/except` in `main.py`; falls back to
  `icon_source = sprite_path` silently (no warning — this is the documented,
  optional-enhancement behavior, unchanged by this slice). Already exercised
  by `tests/test_main.py::test_chibi_render_failure_falls_back_to_plain_downscale`.
- **The LoRA renders something that doesn't read as a caricature but doesn't
  raise either** (a real risk called out in the original chibi research) —
  explicitly accepted, not a bug: nothing in `generate_sprite_img2img` /
  `main.py` inspects render *quality*, only whether the call raised. No new
  handling is added for this outcome per the issue's explicit instruction.
- **`extra_tags` empty/`None` for other callers** (e.g. the plain front-sprite
  img2img call, which passes no `extra_tags`) — unaffected; `build_prompt`'s
  `if extra_tags:` branch is unchanged for the no-tags case.

## Errors

No new error paths. Unchanged:

- `generate_sprite_img2img` propagates any exception from the pipeline call
  (or from `Image.open`/quantization) to its caller; it does not swallow
  errors itself.
- `main.py`'s chibi call sits in an inner `try/except Exception` whose `except`
  branch sets `icon_source = sprite_path` with no warning printed (contrast
  with the outer per-block `except`s elsewhere in `main`, which do warn) —
  this asymmetry is intentional and pre-existing (the chibi pass is an
  optional enhancement, not a required step).

## Constraints & dependencies

- `generate_sprite_img2img` performs a function-local `import torch` (via
  `_run_img2img` -> `_make_generator`), so any test that calls it directly is
  an `ml` test and belongs in `tests/test_sprites_ml.py` per `CLAUDE.md`'s
  slicing rule; it is auto-skipped without torch installed (expected in the
  keep sandbox).
- `tests/test_main.py` mocks `generate_sprite_img2img` at the `main` module
  level, so its existing chibi tests exercise wiring (call args/kwargs,
  fallback control flow) without touching torch/diffusers at all; that
  coverage is unaffected by this slice.
- The "pipeline-swap" mocking seam already established for SDXL-pipeline
  behavior in this test suite is the plain `_fake_img2img_pipeline(image)`
  `MagicMock`-with-`.images`-attribute helper in `tests/test_sprites_ml.py`,
  reused by every existing `generate_sprite_img2img` / `generate_frame2` test
  in that file (e.g. `test_img2img_conditioning_image_passed_to_pipeline`,
  `test_img2img_reference_path_pipeline_called_once_with_passthrough`). These
  already exercise the SDXL-shaped img2img call (`image=`, `strength=`, no
  `width`/`height`, unlike the txt2img/pair calls) generically; this slice
  reuses that exact seam for the chibi tag combination rather than inventing a
  new one or importing the real `diffusers.StableDiffusionXLImg2ImgPipeline`
  class (which would require diffusers importable at test-module load time —
  breaking collection in the torch/diffusers-less sandbox, since
  `pytest_collection_modifyitems` skips `ml` items only *after* collection).
- `_CHIBI_TAGS` must be imported from `fakemon_forge.main` (as
  `tests/test_main.py` already does) rather than re-declared as a literal in
  the new test, so a future edit to the constant is automatically reflected.

## Tests

### `tests/test_sprites_ml.py` (new test; `ml`-marked via the file's
`pytestmark`, needs torch, auto-skipped in the keep sandbox)

Add a test alongside the existing `generate_sprite_img2img` tests:

- Import `_CHIBI_TAGS` from `fakemon_forge.main`.
- Using the existing `_fake_img2img_pipeline` seam (a `MagicMock` pipeline
  returning a plain RGB image via `.images`), call
  `generate_sprite_img2img("fire lizard", [], str(init_img), str(out),
  pipeline=pipe, extra_tags=_CHIBI_TAGS)`.
- Assert the prompt actually sent to the pipeline
  (`pipe.call_args.kwargs["prompt"]`) contains every tag in `_CHIBI_TAGS`
  (`"chibi"`, `"big head"`, `"small body"`) — proving the SDXL-shaped img2img
  call (the same call shape every other test in this file already exercises:
  `image=`, `strength=`, `generator=`, no `width`/`height`) correctly folds the
  chibi tags into the prompt end to end, not just at the `build_prompt` unit
  level.
- Assert `pipe.call_count == 1` and the saved output exists / is `P`-mode
  (mirroring the file's existing `test_img2img_*` assertions), confirming the
  call still completes and produces a valid sprite file with the chibi tags
  applied.
- Do **not** assert anything about the rendered image's *content* (caricature
  proportions) — that is a GPU/human-eyeball judgement explicitly out of
  scope (see Edge cases).

### `tests/test_main.py` (no change required, verification only)

- `test_chibi_render_feeds_the_icon` and
  `test_chibi_render_failure_falls_back_to_plain_downscale` already cover the
  happy path and the exception-fallback path at the wiring level (`extra_tags
  == _CHIBI_TAGS`, init/output paths, `icon_source` selection). Confirm both
  still pass unmodified — they do not depend on which concrete pipeline class
  `img2img_pipeline` is at runtime, since `generate_sprite_img2img` is fully
  mocked in this file.

### Regression check

- Run the full suite (`pytest`); confirm no existing test needs updating and
  the ~21 `ml`-marked tests remain the only skips in the torch-less sandbox
  (per `CLAUDE.md`).

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[confirmed]** `extra_tags` support in the prompt-building path was **not**
  dropped by the SDXL rewrite of `build_prompt`/pipelines: `build_prompt`
  still takes `extra_tags: list[str] | None = None` and folds it in
  (`sprites.py:56-60`), `generate_sprite_img2img` -> `_run_img2img` ->
  `_run_img2img_on_image` all thread `extra_tags` through unchanged
  (`sprites.py:672-733`), and `tests/test_sprites.py` already has a passing
  unit test built from the literal `_CHIBI_TAGS` values
  (`test_build_prompt_multiple_extra_tags_joined_with_comma`). Therefore this
  slice makes **no production code change**, per the issue's own conditional
  ("if it was dropped, restore... that is the only production-code change
  this slice should need").
- **[confirmed]** `main.py`'s chibi call site and `_CHIBI_TAGS` are unchanged
  from the version written in `keep/aab47792` (the original chibi-icon spec);
  `git log -S _CHIBI_TAGS` shows no edits to the constant or call site since.
- **[picked]** The new SDXL-verification test belongs in
  `tests/test_sprites_ml.py` (not `tests/test_main.py`), because that is where
  the *real* `generate_sprite_img2img` function runs against an
  SDXL-img2img-shaped pipeline call; `tests/test_main.py`'s chibi tests mock
  `generate_sprite_img2img` away entirely and so cannot demonstrate anything
  about the pipeline shape. This also matches `CLAUDE.md`'s rule that any test
  invoking `generate_sprite_img2img` for real belongs in `test_sprites_ml.py`.
- **[picked]** "Mock at the same seam already used for the pipeline-swap
  tests" is read as: reuse the existing generic `_fake_img2img_pipeline`
  `MagicMock` seam that every other SDXL-shaped img2img test in
  `test_sprites_ml.py` already uses (chosen over importing the real
  `diffusers.StableDiffusionXLImg2ImgPipeline` class as a `spec=`, which would
  require diffusers to be importable at test-collection time and would break
  collection of the whole file in the torch/diffusers-less sandbox — `ml` marks
  are only skipped by `conftest.py` *after* collection, not before). The
  sys.modules-injection seam used in `tests/test_sprites.py` for
  `make_img2img_pipeline`/`load_img2img_pipeline` (which *does* reference the
  real class name via a mocked `diffusers` module) verifies pipeline
  *construction*, a different concern already covered by earlier slices; this
  slice is about the chibi *call*, which is pipeline-class-agnostic by design.
- **[confirmed]** No caricature-quality assertion is added: the original chibi
  spec (`keep/aab47792`) and this issue both explicitly flag that whether the
  LoRA (SD1.5 originally, SDXL now) produces recognisable chibi proportions
  from `extra_tags` is an unverified, GPU-dependent tunable — not something an
  automated test can or should assert on pixel content.
