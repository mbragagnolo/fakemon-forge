# Bug: `generate_frame2` still img2imgs from the raw front sprite instead of a squash-init

## Summary

`generate_frame2` (in `fakemon_forge/sprites.py`) is supposed to img2img from
`procedural_squash(frame1)` at `strength=0.30` so the SDXL pipeline is forced
to clean up a real structural pose change (per issue #67 / slice 6 of #61).
Instead it still img2imgs from the raw, unmodified front sprite at
`strength=0.35` — the pre-fix recipe that a cited research spike found reads
as colour jitter rather than motion. `build_frame2`'s acceptance band and
fallback (out of scope for this slice) are unaffected and working correctly.

## Repro steps

1. Stub `torch` in `sys.modules` (real torch isn't installed in this sandbox
   per `CLAUDE.md`) so `_make_generator`'s function-local `import torch`
   resolves without the real dependency.
2. Build a small P-mode `frame1` fixture image and a fake `pipeline` callable
   that records its kwargs and echoes back `kwargs["image"]` as the "generated"
   candidate.
3. Call `sprites.generate_frame2("fire lizard", [], frame1_path, out_path, pipeline=pipe)`.
4. Compare the recorded `strength` and `image=` kwargs against:
   - the raw front sprite, loaded/resized the same way `_run_img2img` does it
   - `procedural_squash(frame1)`, converted to RGB and resized the same way

Reproduced with a throwaway script (not committed, no production files
edited):

```
strength passed: 0.35
init == raw front sprite: True
init == squashed frame1: False
```

## Expected vs. actual

- **Expected**: the img2img init image handed to `pipeline` is
  `procedural_squash(frame1)` (converted to RGB, resized to `_GEN_SIZE` the
  same way the current init image is prepared), and `strength` is `0.30`.
- **Actual**: the init image is the raw front sprite loaded straight from
  `front_sprite_path` (unchanged from frame 1's actual pixels), and
  `strength` is `0.35`.

## Root cause

**Confirmed.** `generate_frame2` (`fakemon_forge/sprites.py:720-739`) calls:

```python
candidate = _run_img2img(
    prompt, types, front_sprite_path, pipeline=pipeline,
    extra_tags=extra_tags or ["open mouth"], seed=seed, strength=strength,
)
```

`_run_img2img` (`fakemon_forge/sprites.py:672-692`) always builds its init
image by `Image.open(image_path).convert("RGB").resize(...)` — i.e. it opens
whatever path it's given, verbatim. `generate_frame2` passes
`front_sprite_path` straight through, so the init image is the unmodified
front sprite, not a squashed version of it. Separately, `generate_frame2`'s
`strength` parameter still defaults to `0.35` (`sprites.py:722`), not the
`0.30` the issue calls for.

Because the init image barely differs from the target look (it *is* the
target look), img2img at low strength has very little structural signal to
work from, so accepted candidates end up being near-identical
recolours/texture shimmer — consistent with the near-zero mask-shift the
issue's research spike measured for the status-quo recipe. This is not a
crash or exception; `build_frame2`'s acceptance band (`[0.02, 0.30]`) still
does its job of rejecting genuinely bad candidates and falling back to
`procedural_squash`, so the bug is a quality/behavior gap, not a functional
failure — the shipped frame 2 quietly falls back to (or barely differs from)
the squash fallback far more often than intended, without any visible error.

`build_frame2`'s band/fallback logic itself was verified unchanged and
correct — not part of this bug.

## Affected files

- `fakemon_forge/sprites.py` — `generate_frame2` (lines ~720-739): wrong init
  image source, wrong `strength` default.
- `tests/test_sprites_ml.py` — `test_frame2_pipeline_called_with_low_strength`
  (line 512-517) currently asserts `strength == 0.35`, encoding the buggy
  value; will need updating alongside the fix. No test currently asserts what
  init image is handed to the pipeline, which is why this shipped unnoticed.

## Regression info

Not a regression. `git log -S "def generate_frame2"` shows the function was
introduced once, in commit `1d4b680` ("keep: implement (keep/d5a31c58)"), with
this exact recipe (raw front sprite as init, `strength=0.35`) from day one,
and has not been modified since. The squash-init/`strength=0.30` recipe this
issue asks for was never implemented — it's a pending fix from a research
spike that isn't itself present in this repo, not a regression from prior
working behavior.

## Proposed fix approach

In `generate_frame2`:

1. Build the img2img init image as `procedural_squash(frame1)` — this
   requires opening `frame1 = Image.open(front_sprite_path)` *before* the
   `_run_img2img` call (currently it's opened after, only for
   `build_frame2`), converting the squashed result to RGB, and resizing it to
   `_GEN_SIZE` the same way `_run_img2img` currently resizes its init image.
2. Since `_run_img2img` takes an `image_path` and opens it itself, either (a)
   write the squashed RGB image to a temp file and pass that path, or more
   cleanly (b) refactor `_run_img2img` (or add a sibling helper) to accept a
   pre-built PIL image directly, so `generate_frame2` can pass the squashed
   in-memory image without a round-trip through disk. Option (b) matches the
   issue's framing ("build the img2img init image... instead of loading
   `front_sprite_path` directly") better and avoids an unnecessary temp file.
3. Change `generate_frame2`'s `strength` default from `0.35` to `0.30`.
4. Keep `extra_tags=extra_tags or ["open mouth"]` as-is.
5. Leave `build_frame2` (band `[0.02, 0.30]`, `procedural_squash` fallback)
   completely untouched.
6. Update `test_frame2_pipeline_called_with_low_strength` to assert `0.30`,
   and extend the `test_frame2_*` suite in `tests/test_sprites_ml.py` to
   assert the `image=` kwarg passed to the pipeline matches
   `procedural_squash(frame1)` (converted/resized), not the raw front sprite.
