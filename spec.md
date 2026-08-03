# Spec: Generate front+back sprite pairs from one SDXL canvas, delete the backside chain

## Summary

Today, a stage's front and back sprites are two independent generations
chained by tag convention, not by construction:

1. **Front**: `generate_sprite(...)` — one `pipeline(...)` txt2img call at
   768x768, `postprocess`-quantized (adaptive palette), saved to `sprite.png`.
   (In the `--image` path this step is `generate_sprite_img2img` seeded from
   the user's drawing instead — see "Explicitly out of scope".)
2. **Back**: `generate_sprite_img2img(..., extra_tags=["backside"], strength=0.65,
   reference_path=sprite_path)` — an img2img pass over the just-written front
   sprite, tagged `"backside"`, quantized against the front's exact palette via
   `quantize_to_reference`, saved to `sprite_back.png`.

Per `research-sprite-generation.md` (2026-08-03 discovery spike), this two-step
chain is a known failure mode: "back sprites aren't back views... img2img +
`backside` tag doesn't rotate the subject." The now-fused LoRA
(`pkspbf_nb_v1.safetensors`, "Pokemon Sprite XL PixelArt back&front") instead
renders a genuine front+back pair **side by side in one 1536x768 canvas** from
a single denoising pass — identity-consistent by construction, no tag needed.
A pure splitter, `split_front_back_canvas`, already exists to cut such a
canvas into a front half and a back half (or report no clean cut column).

This slice wires the two together: a new `generate_sprite_pair(...)` function
in `fakemon_forge/sprites.py` that runs one txt2img call, splits it (with a
one-time reroll and a naive-midline fallback if the split fails), postprocesses
the front adaptively, and locks the back to the front's exact palette via the
existing `quantize_to_reference` — then `main.py`'s `--description`-only stage
loop is rewired to call it instead of the old two-step chain. `generate_back_sprite`
and the `extra_tags=["backside"]` call site are deleted.

### Explicitly out of scope

- **`--image` (img2img front) mode.** The issue's own framing — "today's
  **front-txt2img**-then-backside-img2img two-step" — and its "Done when"
  criterion ("one txt2img call produces both views **for txt2img mode**")
  both describe the `--description`-only path, where the front already comes
  from a bare txt2img call that the new 1536x768 call can directly replace.
  When `--image` is given, the front instead comes from
  `generate_sprite_img2img(..., args.image, ...)` — an img2img pass seeded
  from the user's drawing. There is no described way to fold that into a
  single 1536x768 txt2img canvas without discarding the reference image, so
  this slice leaves that call untouched.
  **Consequence, flagged explicitly:** because the back sprite in `--image`
  mode was produced by the very call site this issue says to delete
  ("the old backside-img2img call path... entirely"), and no replacement is
  specified for that mode, `--image` runs stop producing `sprite_back.png` /
  `sprite_back_shiny.png` after this slice. This is a real regression against
  issue #10 ("back sprite inits from front sprite, not the user's drawing").
  See Assumptions for why this is the picked default and what a follow-up
  slice needs to close.
- **`generate_sprite`** (the plain single-canvas txt2img helper) is left in
  place, unused by `main.py` after this slice, mirroring this codebase's own
  precedent of leaving a superseded generator function in place rather than
  deleting it preemptively (see Assumptions).
- **`generate_sprite_img2img`'s `reference_path` parameter** is left in place
  (still a generic, tested capability), even though no call site passes it
  after this slice. See Assumptions.
- `quantize_to_reference`'s contract, `generate_shiny`, `stitch_spritesheet`'s
  cell layout, `export_ini`/`writer.py` — all unaffected, per the issue.

## Inputs

### New: `generate_sprite_pair(prompt, types, front_output_path, back_output_path, *, pipeline, seed=None)`

- `prompt: str` — `stage["sprite_prompt"]` (or `args.description` for stage 1
  today's call already threads through inconsistently — see Assumptions).
- `types: list[str]` — accepted for call-site parity with `generate_sprite`
  (main.py passes `stage["types"]`); not forwarded into the prompt, matching
  `build_prompt`'s existing contract (type wording already lives in `prompt`).
- `front_output_path: str`, `back_output_path: str` — where the two P-mode
  PNGs are saved. `back_output_path` is only written when a non-empty back
  half is found (see Edge cases).
- `pipeline` — a **txt2img** pipeline (`StableDiffusionXLPipeline`-shaped),
  called with `prompt=`, `negative_prompt=`, `width=`, `height=`,
  `num_inference_steps=`, `guidance_scale=`, `generator=` kwargs, returning
  `.images[0]` — exactly `generate_sprite`'s existing call contract, so
  `main.py`'s already-loaded `pipeline` (the txt2img pipeline in the
  no-`--image` branch) is passed through unchanged.
- `seed: int | None = None` (keyword-only) — seeds the initial `pipeline`
  call via `_make_generator(seed)`, matching every other generator function
  in the module.

### `main.py` per-stage sprite block

No new CLI flags. Inside the existing per-stage loop, the `--description`-only
branch's front generation (`generate_sprite`) and the (previously mode-agnostic)
back-sprite block are collapsed into one call to `generate_sprite_pair`. The
`--image` branch's front call is unchanged.

## Outputs

- **`generate_sprite_pair`** → returns `None`. Side effects:
  - Always writes `front_output_path`: a 768x768 `P`-mode PNG, adaptively
    quantized via `postprocess` — byte-for-byte the same post-processing
    `generate_sprite` used to apply.
  - Writes `back_output_path` **iff** a non-empty back half was found: a
    `P`-mode PNG sharing the front's exact palette (via `quantize_to_reference`),
    sized to the front's size (mirroring how `quantize_to_reference` already
    resizes to its reference's size for the frame-2 lock).
  - Never raises for a split failure or an empty back half — both degrade
    with a `stderr` warning, matching `_flatten_background_to_key`'s
    "best-effort result + warn, never raise" convention. A `pipeline(...)`
    call itself raising (real inference failure) **does** propagate — matching
    every existing generator function in this module, which never swallow
    pipeline errors.
- **`main`**, `--description`-only stage: same file set as today
  (`sprite.png`, `sprite_back.png`, `sprite_frame2.png`, three shiny variants,
  `spritesheet.png`, `footprint.png`), but `sprite_back.png` now comes from the
  paired canvas instead of a backside-tagged img2img pass, and is absent (with
  a stderr warning already emitted by `generate_sprite_pair`) on the rare
  empty-back-half case instead of always being present.
- **`main`**, `--image` stage: unchanged front-sprite output; `sprite_back.png`
  / `sprite_back_shiny.png` are no longer produced (see "Explicitly out of
  scope").

## Behavior

### `generate_sprite_pair(...)` in `sprites.py`

1. Build the prompt via the existing `build_prompt(prompt)` (no `extra_tags` —
   the parameter is dropped entirely; there is no longer a tag-driven view
   variant to request).
2. Call `pipeline(prompt=..., negative_prompt=_NEGATIVE_PROMPT, width=1536,
   height=768, num_inference_steps=_NUM_STEPS, guidance_scale=_CFG_SCALE,
   generator=_make_generator(seed))`, take `result.images[0]` as `canvas`
   (a 1536x768 RGB image — front on the left, back on the right, per the
   research spike's verified orientation).
3. Resolve `(front_raw, back_raw)` via a new private helper,
   `_split_front_back_with_retry(canvas, regenerate)`, so the retry/fallback
   decision is a small pure-PIL function independent of `pipeline`/`torch`:
   - Try `split_front_back_canvas(canvas)`. If it returns a pair, that's
     `(front_raw, back_raw)` — done.
   - If it returns `None`: call `regenerate()` (the caller-supplied zero-arg
     callable that reruns the pipeline with `seed + 1`, or a plain unseeded
     call again if `seed is None` — see Assumptions) to get a fresh `canvas`,
     and try `split_front_back_canvas` on **that** — a full regeneration, not
     a re-split of the same pixels.
   - If it's still `None`: split at the naive midline
     (`canvas.crop((0, 0, w // 2, h))` / `canvas.crop((w // 2, 0, w, h))`) and
     `print(..., file=sys.stderr)` a warning — never raise. This mirrors
     `_flatten_background_to_key`'s gradient-border fallback shape exactly:
     best-effort result, stderr warning, no exception.
   - `generate_sprite_pair` calls this with
     `regenerate=lambda: pipeline(prompt=..., negative_prompt=_NEGATIVE_PROMPT,
     width=1536, height=768, num_inference_steps=_NUM_STEPS,
     guidance_scale=_CFG_SCALE, generator=_make_generator(seed + 1 if seed is
     not None else None)).images[0]` — i.e. the only `pipeline`/`_make_generator`
     (torch) calls happen in `generate_sprite_pair` itself; the helper it
     calls is torch-free and unit-testable with a stub `regenerate`.
   - At most **two** `pipeline(...)` calls total per `generate_sprite_pair`
     invocation (initial + at most one reroll).
4. `front = postprocess(front_raw)` (adaptive palette, `size` defaults to
   `_SPRITE_SIZE`, unchanged from `generate_sprite`'s post-step);
   `front.save(front_output_path)`.
5. Lock the back half to the front's exact palette *in memory* —
   `back = quantize_to_reference(back_raw, front)` — reusing `front` directly
   rather than round-tripping through `front_output_path` on disk (unlike the
   old `reference_path=...` call, which needed the round trip because it was
   a separate function invocation).
6. **Empty-back check:** because `quantize_to_reference` locks to `front`'s
   exact palette, and the Gen-3 contract (`_quantize_gen3`) always puts the
   transparency key at **palette index 0**, "the back half is empty or
   entirely background" reduces to: every pixel of `back` decodes to index 0.
   Concretely, this is the same test `_content_bbox` already answers
   elsewhere in the module — `_content_bbox(back, background=0) is None` —
   just with the background index taken from the Gen-3 contract instead of
   computed via `_background_index` (which would also work here, since index
   0 dominates an all-background image, but is unnecessary generality for a
   contract that already guarantees the index).
   - If empty: `print(..., file=sys.stderr)` a warning and **do not** save
     `back_output_path` (front is already written).
   - Else: `back.save(back_output_path)`.

### `main.py` wiring

Illustrative (not literal code to paste — see Constraints for exact
call-site coordinates):

```
sprite_path = str(stage_dir / "sprite.png")
back_path = str(stage_dir / "sprite_back.png")
try:
    if args.image:
        generate_sprite_img2img(
            stage["sprite_prompt"], stage["types"], args.image, sprite_path,
            pipeline=pipeline, seed=seed,
        )
    else:
        generate_sprite_pair(
            stage["sprite_prompt"], stage["types"], sprite_path, back_path,
            pipeline=pipeline, seed=seed,
        )
except Exception as exc:
    print(f"Warning: sprite generation failed for {stage['name']}: {exc}", file=sys.stderr)
    continue

# ... chibi/icon block: unchanged, still reads sprite_path ...

# The old back-sprite block (lines ~137-150 today) is deleted outright: its
# job is now folded into generate_sprite_pair for the --description path,
# and --image mode has no replacement this slice (see "Explicitly out of
# scope").
```

This satisfies both preservation requirements verbatim:
- **Front-generation failure still `continue`s.** If the initial
  `pipeline(...)` call inside `generate_sprite_pair` (or the img2img front
  call) raises, that *is* a front-generation failure — there is no front
  sprite for chibi/icon/frame2/shiny to build on — so the existing
  `except: warn; continue` fires exactly as it does today.
- **A back-only failure warns and continues to icon/frame2/shiny.** Both
  back-only degradations (split-failure-after-reroll, empty-back-half) are
  handled *inside* `generate_sprite_pair` with a `stderr` warning and no
  exception, so from `main.py`'s perspective the call simply returns
  normally and execution falls through to the chibi/icon block exactly as
  the old, independently-caught back-sprite `try/except` allowed.

`main.py`'s import list drops `generate_sprite` (no longer referenced there)
and gains `generate_sprite_pair`; `generate_sprite_img2img` stays imported
(still used for the `--image` front call and the chibi call).

## Edge cases

- **Clean split on the first canvas** → normal path, one `pipeline` call.
- **No clean split on the first canvas, clean split after reroll** → two
  `pipeline` calls, no warning (the reroll finding a clean gap is the
  documented happy path for that branch, not a degradation).
- **No clean split even after reroll** → two `pipeline` calls, naive midline
  split, one `stderr` warning. Front and back are still both attempted from
  that midline crop.
- **Back half empty/background-only** (from *any* of the three split paths
  above) → front is written, back is not, one `stderr` warning naming the
  skipped path. Downstream steps (`generate_shiny(back_path, ...)`,
  `stitch_spritesheet`, `export_ini`) already tolerate a missing
  `sprite_back.png` today — `stitch_spritesheet` explicitly leaves that cell
  on the transparency key, and the back-shiny step's own `try/except` in
  `main.py` catches `Image.open`'s `FileNotFoundError` and warns, exactly as
  it already does for a fully-failed back sprite today.
- **`seed=None`** → the initial call is unseeded (`_make_generator(None)`,
  same as every other generator function in the module); the reroll cannot
  add 1 to a seed that doesn't exist, so it is also a plain unseeded call —
  already stochastic, so a second draw is a legitimate "reroll" in spirit
  even without the `+1`.
- **`--image` stage** → front sprite generation and the chibi/icon/frame2/shiny
  pipeline are otherwise unaffected; `sprite_back.png` is simply never
  created, degrading the same way a fully-failed back-sprite call degrades
  today (missing file, warned-and-skipped downstream).

## Errors

- `generate_sprite_pair` propagates any exception the underlying
  `pipeline(...)` call raises (both the initial call and the reroll) —
  it does not catch pipeline/inference errors, matching `generate_sprite`
  and `generate_sprite_img2img` today. `main.py`'s existing front-sprite
  `except: warn; continue` is the catch point.
- `quantize_to_reference` still raises `ValueError` if handed a non-`P`-mode
  reference; unreachable here in practice because `front` is always the
  `postprocess` output from step 4 (guaranteed `P`-mode), one call earlier in
  the same function.
- No new `sys.exit` paths.

## Constraints & dependencies

- `generate_sprite_pair` performs a function-local `import torch` (via
  `_make_generator`, called directly by `generate_sprite_pair` for both the
  initial and reroll pipeline calls) — so **any test exercising the actual
  pipeline call** is an `ml` test (`tests/test_sprites_ml.py` /
  `@pytest.mark.ml`), per `CLAUDE.md`. `_split_front_back_with_retry` itself
  calls no torch and takes its `regenerate` callable as a plain argument, so
  it belongs in `tests/test_sprites.py` with a stub `regenerate`.
- `main.py`'s sprite block is roughly lines 97-150 today; the front branch is
  97-111, the (deleted) back-sprite block is 137-150. `write_output`,
  `export_ini`, footprint sizing, and the shiny/spritesheet blocks (152-221)
  are untouched.
- `split_front_back_canvas` and its tunables (`_SPLIT_SEARCH_LOW/HIGH`,
  `_KEY_TOLERANCE`) are reused as-is; this slice adds no new tunables besides
  the canvas size (`1536x768`, spec'd literally by the issue) and the naive
  midline fallback (`w // 2`, no new constant needed — it is not an "eyeball
  placeholder" like the module's tunables, it is a structural fallback).
  `Image.crop((0, 0, w // 2, h))` on an odd `w` rounds down (e.g. 1535 -> 767
  front / 768 back); no special-casing needed since `_GEN_SIZE`/canvas width
  are fixed constants, not user input.

## Tests

### `tests/test_sprites.py` (torch-free)

- `_split_front_back_with_retry(canvas, regenerate)`:
  - Clean split on the first canvas: `regenerate` is never called (assert via
    a stub that raises if invoked, or a call counter).
  - First canvas has no clean split, second (the `regenerate()` result) does:
    the returned halves come from the **second** canvas, not the first —
    build two visually distinguishable canvases (à la `_split_canvas` in the
    existing `split_front_back_canvas` tests) and assert on pixel content.
  - Neither canvas splits cleanly: returns the naive `w // 2` crop of the
    *second* canvas (the one `regenerate` returned, since it was already
    called), and `capsys`-style `stderr` is non-empty — following
    `test_flatten_gradient_border_warns_without_raising`'s existing pattern of
    asserting `err` truthy rather than exact wording.
- Empty-back-half check, exercised directly as pure PIL logic (build a `front`
  via `postprocess`, build a `back_raw` that is pure background color, run it
  through `quantize_to_reference` + the index-0 bbox check) — this can be
  asserted either by calling a small extracted helper directly, or by
  constructing the equivalent inline in the test if no such helper is
  factored out. Either is acceptable; the point is the branch is not left
  solely covered by an `ml` test.

### `tests/test_sprites_ml.py` (torch, auto-skipped in this sandbox)

- `generate_sprite_pair` calls `pipeline` with `width=1536`, `height=768`,
  and the other existing kwargs (`prompt`, `negative_prompt`,
  `num_inference_steps`, `guidance_scale`, `generator`) — same assertion
  shape as the deleted `generate_sprite` tests it replaces coverage for.
  A fake pipeline returning a pre-built side-by-side canvas (front/back
  distinguishable by color, mirroring `_split_canvas`) exercises the full
  happy path: exactly one `pipeline` call, `front_output_path` is `P`-mode
  768x768 PNG, `back_output_path` is `P`-mode PNG sharing the front's exact
  `getpalette()`.
- Reroll integration: a fake pipeline whose first call returns a
  no-clean-split canvas and second call returns a clean-split canvas —
  `pipeline.call_count == 2`, second call's `generator` seed reflects
  `seed + 1`, output halves come from the second canvas.
- Regression: `generate_sprite_pair` never calls `pipeline` a third time even
  when both canvases fail to split (naive-fallback path, integration-level
  version of the torch-free unit test above).
- Delete: `test_extra_tags_included_in_prompt`'s `generate_sprite(...,
  extra_tags=["backside"])` example is unaffected (that test exercises
  `generate_sprite`, which is untouched); no forced update, though swapping
  its example tag away from the literal string `"backside"` is a reasonable
  optional cleanup since that string no longer means anything special.
- The whole `generate_sprite_img2img(reference_path=...)` — "back-sprite
  palette lock" test block (today's lines ~260-326) stays as-is:
  `reference_path` is not removed (see Assumptions), so its existing
  generic-capability coverage remains valid even though no production call
  site exercises it after this slice.

### `tests/test_main.py`

- `ctx` fixture: replace `patch("fakemon_forge.main.generate_sprite")` with
  `patch("fakemon_forge.main.generate_sprite_pair")` (main.py's import list
  changes accordingly).
- `test_txt2img_path_calls_generate_sprite` → rewrite against
  `generate_sprite_pair`: asserts it is called once with `sprite_path` and
  `back_path` (`str(stage_dir / "sprite_back.png")`) as the two output-path
  arguments; drop the old assertions that inspected `sprite_i2i` calls for a
  `extra_tags == ["backside"]` entry (that call site no longer exists). The
  chibi assertion (`extra_tags == _CHIBI_TAGS` among `sprite_i2i` calls)
  stays — chibi generation is unaffected.
- `test_txt2img_back_sprite_reference_is_frame1` → delete. It specifically
  asserted the deleted `reference_path=sprite_path` backside call; there is
  no equivalent `main.py`-level assertion to make since
  `generate_sprite_pair` locks the back to the front internally, not via a
  `main.py`-supplied path.
- `test_img2img_path_calls_generate_sprite_img2img` → update the expected
  `call_count` from `3` (front + chibi + back) to `2` (front + chibi only).
- `test_img2img_back_sprite_inits_from_front_sprite` → delete or rewrite as
  an explicit "no back call happens in `--image` mode" assertion. **Flag for
  the implementer/reviewer:** this test existed specifically as regression
  coverage for issue #10; removing it removes that regression's test
  coverage, not just its assertions about a call shape that no longer
  exists. This is a direct, visible consequence of the "Explicitly out of
  scope" decision above, not an incidental test cleanup.
- Any other test asserting `sprite_i2i.call_count` in the `--image` path
  (e.g. around line 621) needs the same `3 -> 2` adjustment.

### `tests/test_stages_e2e.py`

- The `forge` fixture's `patch("fakemon_forge.main.generate_sprite")` becomes
  `patch("fakemon_forge.main.generate_sprite_pair")`; `generate_sprite_img2img`
  stays patched (still used for `--image` front + chibi). No assertions in
  this file inspect sprite call shapes today (it asserts on-disk stage
  structure), so no further changes expected there.

### `tests/test_cli.py`

- No sprite functions are imported or asserted on in this file (`cli.py`
  doesn't reference `sprites.py`); no changes expected.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** `--image` mode's back-sprite generation is dropped in this
  slice rather than adapted, because (a) the issue's own framing and "Done
  when" list both scope the paired-canvas replacement to "txt2img mode," and
  (b) no replacement mechanism for an img2img-seeded pair is described
  anywhere (research doc, issue text). This is a real regression against
  issue #10 that a follow-up slice in #61 should close — candidate approaches
  worth a future spike: compositing the user's drawing into the left half of
  a blank 1536x768 canvas and running the paired-canvas prompt through
  img2img at high denoise (the research doc's own "img2img from a plain
  white canvas at denoise 1.0" recommendation), or a dedicated single-view
  back-generation img2img pass with an in-prompt (not tag-based) rear-view
  phrase. Flagging this clearly rather than silently reintroducing #10's bug
  or silently expanding this slice's scope to design that mechanism.
- **[picked]** `generate_sprite` (the function, not its main.py call site) is
  left in `sprites.py` even though it becomes unused there, mirroring this
  codebase's own explicit precedent (see `spec.md`'s git history — the
  back-sprite palette-lock slice left `generate_back_sprite` "unused... to
  avoid scope creep") until an issue explicitly asks to remove it, as this
  one explicitly does for `generate_back_sprite` but not for `generate_sprite`.
- **[picked]** `generate_sprite_img2img`'s `reference_path` parameter and its
  `quantize_to_reference` branch are left in place, for the same reason:
  the issue's delete list names `generate_back_sprite` and "the old
  backside-img2img call path" (the `main.py` call site), not the generic
  capability on `generate_sprite_img2img` itself, which remains a tested,
  reusable feature even with zero current callers.
- **[picked]** `generate_sprite_pair`'s reroll increments `seed` by exactly 1
  (`seed + 1`), and falls back to an unseeded second call when `seed is None`
  — the issue specifies "seed + 1" for the seeded case but is silent on the
  unseeded one; incrementing `None` has no sensible meaning, and an unseeded
  pipeline call is already non-deterministic, so a second plain call already
  satisfies "a full regeneration, not just a re-split."
- **[picked]** The empty-back-half check uses the Gen-3 contract's guaranteed
  key-at-index-0 rather than computing `_background_index(back)` — simpler
  and doesn't depend on the key happening to be the *most common* index
  (it always is, by contract, once locked to the front's palette, but
  checking index 0 directly is more precisely what "background" means here).
- **[picked]** Function name `generate_sprite_pair` and helper name
  `_split_front_back_with_retry` — not specified by the issue; chosen for
  consistency with existing naming (`generate_sprite`, `generate_frame2`,
  `split_front_back_canvas`).
- **[confirmed]** `split_front_back_canvas` never raises — it returns `None`
  on failure to find a full-height background run — so the reroll logic can
  treat "no split" as a plain `None` check, not an exception path.
  (`tests/test_sprites.py`, `test_split_no_full_height_background_run_returns_none`,
  `test_split_degenerate_search_window_returns_none`.)
- **[confirmed]** `quantize_to_reference` performs its own resize-to-reference-
  size + colour/contrast enhance + background-flatten pipeline internally, so
  `generate_sprite_pair` does not need to pre-process `back_raw` before
  calling it — matching how the deleted `reference_path=...` branch of
  `generate_sprite_img2img` used it.
- **[confirmed]** `_flatten_background_to_key`'s gradient-border fallback is
  the established "degrade, don't fail" pattern this issue explicitly asks
  the split/reroll/naive-fallback logic to mirror (best-effort result +
  `stderr` warning via bare `print(..., file=sys.stderr)`, never raise).
- **[confirmed]** `stitch_spritesheet` already tolerates a missing
  `sprite_back.png` (skips that cell, leaves it on `_KEY_COLOR`), and the
  back-shiny `main.py` block already tolerates a missing `sprite_back.png`
  (an `Image.open` failure caught by its own `try/except`, warn-and-continue)
  — so no changes are needed to either for the new empty-back-half or
  `--image`-mode-has-no-back cases; both were already reachable degradations
  before this slice (a fully-failed back-sprite call left the same gap).
