# Spec: Wire per-stage `sprite_small.png` icon generation into `main.py`

## Summary

Wire the existing `fakemon_forge/icon.py::generate_icon` into the per-stage
loop of `fakemon_forge/main.py` so that every stage directory that gets a front
`sprite.png` also gets a `sprite_small.png` (the 32x64 Gen-3 party-menu icon).

The icon is built with the **plain downscale straight from `sprite.png`** — no
chibi img2img pass. This guarantees the baseline the parent issue (#21) calls
for: *an icon always exists whenever the front sprite exists*. The chibi
enhancement is a later slice and is explicitly out of scope here.

"Done and correct" means:

- After a stage's front `sprite.png` is successfully generated, `main` calls
  `generate_icon(str(stage_dir / "sprite.png"), str(stage_dir / "sprite_small.png"))`
  exactly once, writing `sprite_small.png` into that stage dir.
- The call sits in its own `try/except` following the same warn-to-stderr,
  do-not-raise, do-not-skip-the-rest-of-the-stage pattern as the other per-view
  blocks (back, frame2, shiny, spritesheet).
- It is called once per stage in the single-stage path and 3x in `--mode line`.
- `stitch_spritesheet` / `_SHEET_LAYOUT` are unchanged: `sprite_small.png` is a
  per-stage-dir artifact only and does **not** appear in `spritesheet.png`.
- `pytest` passes in the slim keep sandbox (these tests mock `generate_icon`;
  no torch involved).

## Inputs

- Nothing new from the CLI or environment. The slice consumes existing loop
  state:
  - `stage_dir` — the per-stage output `Path` from `write_output(stages)`.
  - `sprite_path` = `str(stage_dir / "sprite.png")` — the already-generated,
    verified-to-exist front sprite (the front block does `continue` on failure,
    so reaching the icon block means `sprite.png` was written).
  - `stage["name"]` — used only in the warning message.
- `generate_icon(source_path, output_path)` reads `source_path` (an existing
  `P`-mode sprite) and writes `output_path`. It raises `ValueError` on
  non-`P`-mode input; any such error is handled by the warn-and-continue wrapper.

## Outputs

- A new file `sprite_small.png` written into each stage directory alongside the
  existing views (`sprite.png`, `sprite_back.png`, `sprite_frame2.png`, shiny
  variants, `spritesheet.png`).
- On icon failure: a single stderr line
  `Warning: icon generation failed for {stage['name']}: {exc}` and no file (or a
  partial file, per `generate_icon`'s own behavior); the loop continues.
- No change to stdout's final `Done!` line, to return values, or to exit codes.

## Behavior

1. Add an import near the other `fakemon_forge` imports in `main.py`:
   `from fakemon_forge.icon import generate_icon` (importing the bound symbol
   `generate_icon`, matching the style of the other sprite-view imports and the
   test-patch target `fakemon_forge.main.generate_icon`).
2. Inside the existing `for stage, stage_dir in zip(stages, stage_dirs):` loop,
   **after** the front-sprite `try/except` block (the one that `continue`s on
   failure — so `sprite.png` is known to exist), add a new per-view block:

   - Compute `icon_path = str(stage_dir / "sprite_small.png")`.
   - `try:` call `generate_icon(sprite_path, icon_path)`.
   - `except Exception as exc:` print
     `f"Warning: icon generation failed for {stage['name']}: {exc}"` to
     `sys.stderr`, and fall through (no `continue`, no re-raise).

3. The block depends only on `sprite.png`; it is **not** gated on
   back/frame2/shiny succeeding. Its position relative to those other blocks
   does not matter as long as it is after the front-sprite block and inside the
   same loop iteration. Recommended placement: immediately after the front block
   (before the back block) so the icon is produced as early as its only
   dependency allows.
4. Runs once per stage, so `--mode line` (3 stages) produces 3 icons.

## Edge cases

- **Front sprite failed:** the front block already `continue`s, so the icon
  block is never reached — no icon, consistent with "icon exists iff front
  sprite exists".
- **`generate_icon` raises `ValueError` (non-`P`-mode source):** caught by the
  wrapper, warned, loop continues; later blocks in the same stage still run.
- **`generate_icon` raises any other `Exception` (I/O, Pillow error):** same
  warn-and-continue handling (catch broad `Exception`, mirroring sibling
  blocks).
- **Line mode / multiple stages:** independent per stage; one stage's icon
  failure does not affect other stages.
- **Spritesheet unaffected:** `_SHEET_LAYOUT` still lists exactly its 6 views;
  `sprite_small.png` is never stitched in.

## Errors

- Icon generation never aborts the run. All exceptions from `generate_icon` are
  caught by the block's `except Exception` and surfaced as a non-fatal
  `Warning:` line on stderr; `main` does not change its exit status because of
  an icon failure.
- No new fatal error paths, no new `sys.exit` calls, no new user-facing prompts.

## Constraints & dependencies

- Depends on `fakemon_forge/icon.py::generate_icon` (already merged in the prior
  slice). Pure Pillow — no torch/diffusers — so the change and its tests run in
  the slim sandbox.
- Must reuse the existing `sprite_path` variable and the established
  warn-and-continue idiom (broad `except Exception as exc`,
  `print(..., file=sys.stderr)`); no new helper, no refactor of the loop.
- Must not modify `stitch_spritesheet`, `_SHEET_LAYOUT`, or any other module.
- Message format must match the sibling blocks:
  `Warning: icon generation failed for {stage['name']}: {exc}` so existing
  assertions on `"Warning"` + stage name style hold.

## Assumptions

- **[picked default]** Filename is `sprite_small.png` (matches the reference
  asset name in the parent issue and `icon.py`'s docstring).
- **[picked default]** Import form is
  `from fakemon_forge.icon import generate_icon` (bound-symbol style, like the
  `from fakemon_forge.sprites import (...)` block); the test patch target is
  correspondingly `fakemon_forge.main.generate_icon`. The alternative
  `from fakemon_forge import icon` + `icon.generate_icon(...)` (patch
  `fakemon_forge.main.icon.generate_icon`) is equally valid; the bound-symbol
  form is chosen to match the surrounding imports and the simplest patch target.
- **[picked default]** The icon block is placed immediately after the
  front-sprite block (before the back block). Order among the per-view blocks is
  behaviorally irrelevant since the icon depends only on `sprite.png`; this
  placement is chosen for readability (produced as soon as its dependency
  exists).
- **[picked default]** The block catches broad `Exception` (not just
  `ValueError`), matching every sibling per-view block, so any Pillow/I/O error
  degrades gracefully rather than aborting the stage.
- **[confirmed]** This slice uses the plain downscale only; no chibi img2img
  pass and no spritesheet inclusion (both stated in the issue and deferred to a
  later slice).
- **[confirmed]** `generate_icon` already exists, is pure Pillow, and raises
  `ValueError` on non-`P`-mode input (verified in `fakemon_forge/icon.py`).

## Tests

Location: `tests/test_main.py` (regular file — all `main` collaborators are
mocked; no torch). Extend the `ctx` and `ctx_line` fixtures with
`patch("fakemon_forge.main.generate_icon")` (exposed under a dict key such as
`"icon"`), then add:

- **Called once per stage (single-stage path):** after
  `main(["--description", "fire lizard"])`, `ctx["icon"]` is called once, with
  positional args
  `(str(stage_dir / "sprite.png"), str(stage_dir / "sprite_small.png"))`.
- **Called 3x in `--mode line`:** using `ctx_line`,
  `generate_icon.call_count == 3`.
- **Failure warns but does not exit or skip the rest of the stage:** set
  `ctx["icon"].side_effect = RuntimeError("icon crash")`; `main(...)` must not
  raise; stderr contains `"Warning"` and `"Flamburr"`; a later block in the same
  stage still runs (e.g. `stitch_spritesheet` still called once, or the front
  shiny still written) — mirroring `test_frame2_failure_warns_but_does_not_exit`.
- **(Optional regression) spritesheet layout unchanged:** `_SHEET_LAYOUT` still
  lists its existing 6 views and `sprite_small.png` is not among them.
