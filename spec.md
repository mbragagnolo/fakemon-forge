# Spec: SDXL + kohya LoRA shim pipeline loaders (`sprites.py`)

## Summary

Slice 4/9 of #61. Swap `sprites.py`'s pipeline-loading, LoRA-application,
scheduler, and prompt-building machinery from the SD1.5 stack
(`Lykon/dreamshaper-8` + `pksp768_V2-1` LoRA + `compel` + DPM++ Karras) to the
SDXL stack (`Laxhar/noobai-XL-1.1` + `pkspbf_nb_v1` "back&front" LoRA +
Euler Ancestral). This is infrastructure-only: no change to sprite
post-processing (`postprocess`, `_quantize_gen3`, palette locking, frame2
band-acceptance), and no change to the front+back single-canvas/split
generation flow — that's the next slice (per the issue, which describes a new
1536x768 canvas + split as a later step).

Done and correct means: `load_txt2img_pipeline()` / `load_img2img_pipeline()`
build `StableDiffusionXLPipeline` / `StableDiffusionXLImg2ImgPipeline`
instances from `_BASE_MODEL_ID`, with the kohya-format LoRA applied via a
manual shim (bypassing diffusers' broken `load_lora_weights` rank detection
for this LoRA format), an `EulerAncestralDiscreteScheduler`, and the
CUDA-mandatory offload/tiling defaults — while `compel`, `_TYPE_TAGS`,
`_encode_prompt`, and every old SD1.5-specific name are fully gone from both
`sprites.py` and `pyproject.toml`.

## Inputs

- `_BASE_MODEL_ID: str` — HF hub id `"Laxhar/noobai-XL-1.1"`, passed to
  `.from_pretrained`.
- `_LORA_PATH: Path` — `models/loras/pkspbf_nb_v1.safetensors`, resolved the
  same way as today (`Path(__file__).parent.parent / "models" / "loras" / ...`).
  `models/` is already gitignored; this file is never committed and must be
  manually downloaded by whoever runs the real (non-mocked, non-CI) pipeline.
- `_LORA_SCALE` — kept as today's tunable (`0.7`); unaffected by this slice
  except that it's now passed to `pipe.fuse_lora(lora_scale=_LORA_SCALE)` in
  the new shim, exactly as the old `_apply_lora` already did.
- `build_prompt`'s caller-supplied `sprite_prompt: str` (LLM-authored, from
  `generator.py`) and optional `extra_tags: list[str] | None` (e.g.
  `["backside"]`, `["open mouth"]`, `["chibi", "big head", "small body"]` —
  used today by `generate_back_sprite`, `generate_frame2`, and `icon.py`'s
  chibi pass respectively).
- Pipeline call inputs are unchanged in kind: `width`/`height` (txt2img),
  `image`/`strength` (img2img), `num_inference_steps`, `guidance_scale`,
  `generator` (seeded via `_make_generator`).

## Outputs

- `load_txt2img_pipeline() -> StableDiffusionXLPipeline` (or `sys.exit(1)` on
  any load failure, message on stderr — shape unchanged from today).
- `load_img2img_pipeline() -> StableDiffusionXLImg2ImgPipeline` (same
  error-handling shape).
- `make_img2img_pipeline(txt2img_pipe) -> StableDiffusionXLImg2ImgPipeline`,
  built from the txt2img pipeline's components (SDXL equivalent of today's
  SD1.5 component reuse).
- `build_prompt(sprite_prompt, extra_tags=None) -> str` — a plain string, no
  type vocabulary.
- `_NEGATIVE_PROMPT: str` (new constant) — passed as `negative_prompt=` on
  every pipeline call.
- Loaded pipelines have: the LoRA fused in (`fuse_lora` called), scheduler
  swapped to `EulerAncestralDiscreteScheduler`, and — CUDA path only —
  `enable_model_cpu_offload()` and VAE tiling enabled.

## Behavior

### Prompt building

`build_prompt(sprite_prompt: str, extra_tags: list[str] | None = None) -> str`
drops the `types` parameter and all `_TYPE_TAGS` logic. Any type wording is
now the caller's responsibility (baked into the LLM-authored `sprite_prompt`
upstream in `generator.py` — out of scope for this slice; `generator.py` is
not touched here).

Formula, keeping `extra_tags` support (needed so `generate_back_sprite`,
`generate_frame2`, and `icon.py`'s chibi pass can still differentiate their
prompts — the issue's exact literal `f"gen3, {sprite_prompt}, white
background"` is the `extra_tags=None` case):

- No `extra_tags`: `f"gen3, {sprite_prompt}, white background"`
- With `extra_tags`: `f"gen3, {sprite_prompt}, {', '.join(extra_tags)}, white background"`

`generate_sprite` / `generate_back_sprite` / `generate_sprite_img2img` /
`generate_frame2` keep accepting a `types: list[str]` parameter (so
`main.py`'s call sites and its own tests, which pass `stage["types"]`, don't
need to change in this slice) but stop forwarding it into `build_prompt`.

### Call sites: `prompt=`/`negative_prompt=` instead of `prompt_embeds=`

Per the issue's scope-boundary judgment call, `generate_sprite` and
`_run_img2img` (shared by `generate_sprite_img2img` and `generate_frame2`)
are rewritten to call the pipeline with:

```
prompt=build_prompt(...), negative_prompt=_NEGATIVE_PROMPT, ...
```

instead of `prompt_embeds=_encode_prompt(...)`. `_encode_prompt` and the
`compel` import are deleted outright — SDXL calls take plain strings for
both conditioning directions, so no replacement encoding helper is needed.

### LoRA shim (`_apply_lora`)

Ported per the issue's verified spike shape, adapted to this module's
existing naming (`pipe`, `_LORA_PATH`, `_LORA_SCALE`):

- Loads `state_dict, network_alphas, metadata` via
  `StableDiffusionXLLoraLoaderMixin.lora_state_dict(str(_LORA_PATH),
  return_lora_metadata=True, unet_config=pipe.unet.config)` — passing
  `unet_config` explicitly is what triggers kohya's SGM block-name
  remapping that `load_lora_weights` would otherwise do internally (and
  crash on, per the issue, with an `IndexError` in rank detection for this
  LoRA format).
- `pipe.load_lora_into_unet(state_dict, network_alphas=network_alphas,
  unet=pipe.unet, metadata=metadata, _pipeline=pipe)`.
- For each of the two SDXL text encoders, strips the `"{prefix}.text_model."`
  wrapper level from keys **only for `text_encoder` (te1,
  `CLIPTextModel`)** — `text_encoder_2` (te2, `CLIPTextModelWithProjection`)
  keeps its keys untouched, since te2 names its modules *with* the
  `text_model.` wrapper and te1 does not (this asymmetry, verified against
  each encoder's `named_modules()`, is why the fix is a per-encoder flag, not
  a blanket transform — same shape as today's single-encoder SD1.5
  `_drop_text_model`, now parameterized by prefix and generalized to run
  twice).
- `pipe.fuse_lora(lora_scale=_LORA_SCALE)`, as today.

### Scheduler

`pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)`
replaces `_set_dpmpp_karras` (which is deleted, along with the
`DPMSolverMultistepScheduler` import/use).

### Base pipeline loading (`_load_base_pipeline`)

Same shape as today (build from `_BASE_MODEL_ID` via `pipe_cls.from_pretrained`,
apply LoRA, set scheduler, move to device), plus the new CUDA-mandatory
memory-saving calls:

- **CUDA**: `pipe.enable_model_cpu_offload()`, then enable VAE tiling. The
  tiling call name is whichever the pinned `diffusers` version exposes on
  `StableDiffusionXLPipeline`/`StableDiffusionXLImg2ImgPipeline` — see
  Assumptions; spec defaults to `pipe.enable_vae_tiling()` with
  `pipe.vae.enable_tiling()` as the documented fallback if that method isn't
  present on the pinned version. These are **mandatory defaults whenever CUDA
  is available**, not opt-in flags, to stay inside the 8GB VRAM budget noted
  in `research-sprite-generation.md`.
  - Note: `enable_model_cpu_offload()` already moves the pipeline to the
    right device internally; today's code does `pipe.to(device)`
    unconditionally after `_load_base_pipeline` returns. `.to("cuda")` is a
    no-op after offload is enabled (diffusers documents this combination as
    safe/idempotent), so the existing `pipe.to(device)` call is left
    unchanged rather than special-cased, keeping the diff minimal.
- **CPU** (`safety_checker` kwarg is also dropped — SDXL pipelines don't
  accept it, unlike the SD1.5 ones): fp32, no offload call — same fallback
  shape as today, only the model id/pipeline classes changed.

`load_txt2img_pipeline` builds `StableDiffusionXLPipeline`;
`load_img2img_pipeline` builds `StableDiffusionXLImg2ImgPipeline`. Both keep
today's try/except-then-`sys.exit(1)`-with-stderr-message shape, unchanged.

`make_img2img_pipeline` builds a `StableDiffusionXLImg2ImgPipeline` from the
txt2img pipeline's `.components`, mirroring today's
`StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)` pattern.

## Edge cases

- **No CUDA (CPU-only dev/CI machine)**: fp32 dtype, no
  `enable_model_cpu_offload()`/tiling calls — identical fallback shape to
  today, just SDXL classes. Already covered by the existing
  `test_load_uses_float32_when_no_cuda` / `test_load_moves_pipeline_to_cpu_when_no_cuda`-style
  tests (rewritten for the new classes/ids).
- **`extra_tags=None` vs `[]`**: both must produce the plain (no-extra-tags)
  formula — treat falsy the same as `None` (mirrors how `build_prompt`
  already treats `extra_tags` today via `extra_tags or []`).
- **Model/LoRA load failure** (missing `models/loras/pkspbf_nb_v1.safetensors`
  file, HF hub auth/network failure, OOM on `.from_pretrained`): caught by
  the existing broad `except Exception` in `load_txt2img_pipeline`/
  `load_img2img_pipeline`, printed to stderr, `sys.exit(1)` — unchanged
  behavior, new underlying exception sources.
- **te1 vs te2 key-prefix asymmetry**: the shim must not blanket-apply
  `_drop_text_model` to both encoders — doing so for te2 would corrupt
  keys that are already correctly prefixed, silently breaking that encoder's
  LoRA weights.
- **Mocked-pipeline tests never touch a real HF hub or LoRA file** — `sys.modules`
  injection (torch/diffusers faked wholesale) and `MagicMock` pipes stand in,
  consistent with the existing `test_load_*` pattern in `test_sprites.py`.

## Errors

No new error *types* are introduced. `load_txt2img_pipeline`/
`load_img2img_pipeline` keep catching any `Exception` from the load path
(now including LoRA-shim-specific failures — bad state-dict shape, missing
`unet.config`, etc. — in addition to the old OOM/missing-weights cases),
printing `f"Error: failed to load model: {exc}"` to stderr, and exiting with
status 1. `generate_shiny`'s and other unrelated `ValueError`s are untouched.

## Constraints & dependencies

- `compel` is removed from `pyproject.toml`'s `dependencies`; no code in
  `sprites.py` (or elsewhere — confirmed no other module imports `compel` or
  `_encode_prompt`) references it after this slice.
- `diffusers`, `transformers`, `accelerate`, `torch` version floors in
  `pyproject.toml` are left unchanged — the issue doesn't ask for a version
  bump, and the existing `_apply_lora` already relied on
  `return_lora_metadata=True` / `metadata=` kwargs on `load_lora_into_unet`/
  `load_lora_into_text_encoder`, so the floor was already implicitly assuming
  a fairly recent `diffusers`; revisiting the floor is out of scope here.
- `models/loras/pkspbf_nb_v1.safetensors` is a manual download (Civitai model
  378602, "Pokemon Sprite XL PixelArt back&front", login required) — a code
  comment at `_LORA_PATH`'s definition documents this (mirroring how the
  file it replaces was handled); full README wording is a later slice per
  the issue.
- `_GEN_SIZE` (768), `_NUM_STEPS` (30), `_CFG_SCALE` (7), `_SPRITE_SIZE` (768)
  are untouched — this slice doesn't touch generation resolution/step/CFG
  tuning, only the loader/LoRA/scheduler/prompt-string machinery.
- No change to `postprocess`, `_quantize_gen3`, `quantize_to_reference`,
  `split_front_back_canvas`, `build_frame2`, or any other post-processing
  function.
- Test placement follows this project's `CLAUDE.md` convention: tests that
  fake `torch`/`diffusers` wholesale via `sys.modules` injection (no real
  `import torch`) stay in `tests/test_sprites.py` unmarked; tests that
  exercise `generate_sprite`/`generate_sprite_img2img`/`generate_frame2`
  (which hit a real `import torch` inside `_make_generator` even with a
  mocked pipeline) stay in `tests/test_sprites_ml.py` under `pytestmark =
  pytest.mark.ml`, auto-skipped here (no torch in this sandbox) and expected
  to run on the host.

## Assumptions

- **VAE tiling method name**: the issue flags this as something to verify
  against "the pinned `diffusers` version," but `diffusers` isn't installed
  in this sandbox (confirmed: `import diffusers` fails here), so it can't be
  checked directly. Default to `pipe.enable_vae_tiling()` (the SDXL
  pipeline-level convenience wrapper), falling back to `pipe.vae.enable_tiling()`
  if the former is absent on the pinned version. Implementer should confirm
  against the actual installed version on the host before merging, and can
  keep this as a one-line `hasattr` fallback rather than a hard-coded choice
  if that turns out cheaper.
- **`build_prompt` keeps an `extra_tags` parameter and drops only `types`.**
  The issue's literal replacement formula
  (`f"gen3, {sprite_prompt}, white background"`) only shows the
  no-extra-tags case and doesn't otherwise mention `extra_tags`; dropping
  `extra_tags` entirely would silently break `generate_back_sprite`
  (`"backside"`), `generate_frame2` (default `"open mouth"`), and
  `icon.py`'s chibi img2img pass (`_CHIBI_TAGS`), none of which this issue
  asks to change. Chosen fix: keep `extra_tags`, append it as a
  comma-separated clause before `"white background"`.
- **`generate_sprite`/`generate_back_sprite`/`generate_sprite_img2img`/
  `generate_frame2` keep their public `types: list[str]` parameter**, now
  unused internally (not forwarded to `build_prompt`), rather than removing
  it from their signatures. Removing it would ripple into `main.py`'s call
  sites and `test_main.py`/`test_stages_e2e.py`, which are outside this
  slice's stated scope (pipeline loaders + LoRA application).
  Type wording, per the issue, is meant to already live inside the caller's
  `sprite_prompt` text — making `types` fully redundant here — but wiring
  that into `generator.py`'s LLM prompt is explicitly not this slice.
- **Test file placement for `test_load_*`/`test_load_img2img_*`.** The issue's
  Tests section says these tests "belong in `tests/test_sprites_ml.py`," but
  they already live in `tests/test_sprites.py` today, unmarked, and fake
  `torch`/`diffusers` entirely via `sys.modules` injection rather than
  triggering a real import — exactly the category this project's `CLAUDE.md`
  says belongs in the *regular* test files, not `test_sprites_ml.py`. Since
  `CLAUDE.md` instructions take precedence, and the new kohya shim is just as
  mockable this way (`StableDiffusionXLLoraLoaderMixin`,
  `pipe.unet.config`, etc. are all `MagicMock` attributes), this spec keeps
  `test_load_*`/`test_load_img2img_*` in `tests/test_sprites.py`, rewritten
  in place for the new model id/classes/scheduler, rather than moving them.
  Only tests that exercise `generate_sprite`/`generate_sprite_img2img`/
  `generate_frame2` end-to-end (needing real `_make_generator` ->
  `import torch`) stay under `test_sprites_ml.py`'s `ml` marker.
- **`test_encode_prompt_*` / `test_type_tags_included_in_encoded_prompt` are
  deleted, not rewritten**, since there's no successor concept (`prompt=`/
  `negative_prompt=` passthrough is asserted by new, differently-named
  tests, e.g. `test_pipeline_called_with_prompt_string` /
  `test_pipeline_called_with_negative_prompt`) — "rewrite" in the issue is
  read loosely here, not as "keep the same test names/shape."
  `test_build_prompt_*` tests in `test_sprites.py` that assert type-tag
  inclusion (not explicitly named in the issue's Tests section, which
  focuses on the ml-marked file) are likewise rewritten to match the new
  plain-string/`extra_tags` formula.
- **`_NEGATIVE_PROMPT` constant name and content** are not specified by the
  issue beyond "e.g." wording; this spec fixes it as `_NEGATIVE_PROMPT =
  "worst quality, low quality, blurry, watermark, signature, text, jpeg
  artifacts"`, matching the issue's suggested text verbatim.
- **`safety_checker=None` kwarg is dropped** from the `.from_pretrained` call
  in `_load_base_pipeline` — SDXL pipeline classes don't accept a
  `safety_checker` argument (unlike the SD1.5 ones), so passing it would
  raise a `TypeError` at load time. The issue doesn't call this out
  explicitly but it follows directly from switching pipeline classes.
- **Scope is exactly this slice** (loaders, LoRA shim, scheduler, prompt
  string, and the `prompt=`/`negative_prompt=` call-site rewrite needed to
  test them end-to-end) — the 1536x768 front+back single-canvas generation
  and split-based `generate_sprite`/`generate_back_sprite` rewrite is
  correctly left for the next slice per the issue, and this spec does not
  attempt it.
