# Spec: Wire footprint generation into main.py per-stage loop

## Summary
Wire `generate_footprint` (from `fakemon_forge/footprint.py`) into the per-stage
loop of `fakemon_forge/main.py`. After a stage's other views are produced and the
spritesheet is stitched, generate a `footprint.png` for the stage using a
`size_fraction` derived from the stage's position in the evolutionary line and a
`blank` flag derived from the stage's `levitates` value. The call is wrapped in
the same warn-and-continue `try/except` pattern used by every other per-view step
("degrade per view"): a failure prints a `Warning:` line to stderr and the run
continues.

"Done and correct" means: `main.py` emits exactly one `footprint.png` per stage
whose front sprite succeeded, with the correct `size_fraction` (0.6 / 0.75 / 0.9
for stages 1/2/3 of a 3-stage line; 0.9 for a single form) and correct `blank`
value (`stage.get("levitates", False)`), never aborts the run on a footprint
failure, does not add `footprint.png` to `spritesheet.png`, makes no `export_ini`
changes, and passes the full `pytest` suite (with the usual ~21 `ml` tests
skipped without torch, per `CLAUDE.md`).

## Inputs
Per stage, inside the existing loop (`for stage, stage_dir in zip(stages, stage_dirs)`):
- `stage` — a dict carrying at least:
  - `stage["stage"]` — integer tier position in the line (1, 2, or 3).
  - `stage["types"]` — list of type strings (primary type is `types[0]`).
  - `stage["name"]` — display name, used in the warning message.
  - `stage.get("levitates", False)` — optional boolean; a missing key degrades to
    `False` ("render normally"). Persisted in `stats.json` by `writer.py`.
- `stage_dir` — the `Path` to the stage's output directory; `footprint.png` is
  written as `stage_dir / "footprint.png"`.
- `stages` — the full list of stages for the current run; `len(stages)`
  distinguishes a 3-stage line (`== 3`) from a single form (`== 1`).
- `sprite_path` — `str(stage_dir / "sprite.png")`, the already-generated front
  sprite the footprint reads from (only when `blank` is false).

`generate_footprint` signature (unchanged, already merged):
`generate_footprint(sprite_path, output_path, *, types, size_fraction=0.9, blank=False) -> None`.

## Outputs
- `stage_dir / "footprint.png"` — a 16×16 RGBA PNG written by `generate_footprint`
  (opaque black on transparent, or all-transparent when `blank=True`). Written per
  stage whose front sprite succeeded.
- On failure: a line to `stderr` of the form
  `Warning: footprint generation failed for <name>: <exc>` and no aborting of the
  run.
- No change to `spritesheet.png`, `stats.json`, `entry.txt`, or the `.ini` export.

## Behavior
1. Add `from fakemon_forge.footprint import generate_footprint` to the imports in
   `main.py`.
2. In the per-stage loop, **after** the spritesheet-stitching `try/except` block
   (and before the loop advances to the next stage), compute `size_fraction` from
   the stage's position and call `generate_footprint`.
3. `size_fraction` mapping:
   - If `len(stages) == 3`: use a local mapping keyed off `stage["stage"]` —
     `{1: 0.6, 2: 0.75, 3: 0.9}`.
   - Otherwise (single form, `len(stages) == 1`, any tier including
     standard/legendary/mythical): `0.9`.
   - Implement as a small local dict lookup with a default of `0.9`, e.g.
     `size_fraction = {1: 0.6, 2: 0.75, 3: 0.9}.get(stage["stage"], 0.9) if len(stages) == 3 else 0.9`
     (exact expression left to the implementer, but semantics must match the
     mapping above).
4. The call:
   ```python
   footprint_path = str(stage_dir / "footprint.png")
   try:
       generate_footprint(
           sprite_path,
           footprint_path,
           types=stage["types"],
           size_fraction=<computed>,
           blank=stage.get("levitates", False),
       )
   except Exception as exc:
       print(
           f"Warning: footprint generation failed for {stage['name']}: {exc}",
           file=sys.stderr,
       )
   ```
5. Because the front-sprite failure path already `continue`s earlier in the loop,
   a stage whose front sprite failed reaches neither the spritesheet step nor the
   footprint step — no footprint is attempted for it. Leave that as-is; do not add
   an explicit guard.
6. Footprint generation is independent per stage and per view: a failure for one
   stage must not prevent footprints for later stages, and must not skip the
   `export_ini` loop that follows.

## Edge cases
- **Missing `levitates` key**: `stage.get("levitates", False)` must not raise;
  it degrades to `blank=False` (render normally).
- **`levitates` true**: `blank=True` is passed; `generate_footprint` writes an
  all-transparent footprint and never reads the sprite (handled inside the
  footprint module — no extra logic in `main.py`).
- **Single form (`len(stages) == 1`)**: always `size_fraction=0.9` regardless of
  tier. `stage["stage"]` for a single form is not consulted for the mapping.
- **3-stage line**: stages 1/2/3 → 0.6/0.75/0.9. Any unexpected `stage["stage"]`
  value within a 3-stage line defaults to `0.9` via the mapping's default.
- **Front sprite failed**: the existing `continue` skips both spritesheet and
  footprint steps — no footprint file and no footprint warning for that stage.
- **Footprint raises**: caught, warned, run continues to the next stage and to the
  `export_ini` loop.

## Errors
- Any exception from `generate_footprint` (e.g. `ValueError` for a non-palette
  sprite, I/O errors, unexpected internals) is caught by the surrounding
  `except Exception as exc` and reported as
  `Warning: footprint generation failed for {stage['name']}: {exc}` on stderr. The
  process does not exit non-zero and does not stop processing remaining stages.
- No new exit codes or error paths are introduced.

## Constraints & dependencies
- Depends on two already-merged prerequisites:
  - `generate_footprint` in `fakemon_forge/footprint.py` (Pillow-only; must not
    pull torch/diffusers into `sys.modules`).
  - The `levitates` flag emitted by the generator and defaulted via
    `stage.get("levitates", False)`.
- `footprint.py` is Pillow-only; importing `generate_footprint` into `main.py`
  introduces no torch/diffusers dependency, so new tests stay in the regular
  (non-`ml`) suite.
- Must not modify `export_ini` or add `footprint.png` to `stitch_spritesheet`
  (official Gen-3 sheets carry no footprints).
- Follow the existing `main.py` conventions: same `try/except Exception as exc` +
  `print(..., file=sys.stderr)` shape, `str(stage_dir / "...")` path building,
  and placement inside the per-stage loop.

## Tests
Add cases to `tests/test_main.py` following the existing mocking style (patch
`fakemon_forge.main.generate_footprint`; reuse the `ctx` / `ctx_line` fixtures;
no real images or API calls). Note: the `ctx` / `ctx_line` fixtures must be
extended to also patch `fakemon_forge.main.generate_footprint` (and expose the
mock), since `main` will now call it.

- **Called once per stage**: with `ctx` (single stage), `generate_footprint` is
  called exactly once; with `ctx_line` (3 stages), exactly three times.
- **`blank` reflects `levitates`**:
  - A stage with `levitates=True` → the call's `blank` kwarg is `True`.
  - A stage with `levitates` missing or `False` → `blank` is `False`.
  - (Add a stage variant carrying `levitates=True` for the true case; the existing
    `_STAGE_*` fixtures omit the key, exercising the missing/false default.)
- **`size_fraction` mapping**:
  - 3-stage line: the three calls (in order) receive `size_fraction` 0.6, 0.75,
    0.9 for stages 1/2/3.
  - Single form: the single call receives `size_fraction == 0.9`.
- **`types` passthrough**: the call receives `types=stage["types"]`.
- **Output path**: the call's `output_path` (second positional arg) is
  `str(stage_dir / "footprint.png")`; the first positional arg is
  `str(stage_dir / "sprite.png")`.
- **Failure is caught**: setting `generate_footprint.side_effect =
  RuntimeError(...)` must not raise from `main`; stderr contains
  `Warning: footprint generation failed for Flamburr` (mirror
  `test_spritesheet_failure_warns_but_does_not_exit`). A later step of the run
  (the `export_ini` loop / normal completion) still proceeds.
- Keep all these tests in the regular suite (they touch no torch).

## Assumptions
- **[picked default]** Single forms (`len(stages) == 1`) always use
  `size_fraction=0.9` regardless of tier (standard / legendary / mythical). The
  `stage["stage"]` value is not consulted for single forms — matches the spec
  default called out in the task.
- **[picked default]** The `size_fraction` mapping is keyed off `stage["stage"]`
  with a `0.9` default, so an out-of-range `stage["stage"]` within a 3-stage line
  falls back to `0.9` rather than raising.
- **[picked default]** Footprint generation is placed **after** the
  spritesheet-stitching block (last per-stage step), and relies on the existing
  front-sprite-failure `continue` to skip it implicitly rather than adding an
  explicit guard.
- **[picked default]** The `blank` value comes solely from
  `stage.get("levitates", False)`; no additional heuristics (type-based, etc.)
  gate blank footprints in `main.py`.
- **[confirmed by task/code]** The warning message format is
  `Warning: footprint generation failed for {name}: {exc}` and uses
  `file=sys.stderr`, matching the other per-view warnings in `main.py`.
- **[confirmed by task]** No `export_ini` changes and no addition of
  `footprint.png` to `spritesheet.png`.
