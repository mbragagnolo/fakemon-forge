# Spec: Generate `cry.wav` per stage in the main generation loop

## Summary
Wire the existing `fakemon_forge.cries.generate_cry` into the per-stage asset
loop in `fakemon_forge/main.py` so that every evolution stage also receives a
`cry.wav` alongside its sprites, spritesheet, and exported `.ini`.

`generate_cry(line_name, stage, types, output_path)` already exists (pure
stdlib — no torch/diffusers/GPU/network) and synthesizes a deterministic Gen 3
-style creature cry as a mono / 8-bit unsigned PCM / 10512 Hz WAV. This slice
only *calls* it from the main loop; it does not modify `cries.py`.

"Done and correct" means: `main.py` imports `generate_cry`, calls it exactly
once per stage with the correct arguments, does so **before** the sprite block
(so a sprite failure cannot skip it), wraps the call in the project's standard
warn-and-continue pattern, and `tests/test_main.py` covers the call, line-mode
fan-out, sprite-failure independence, and cry-failure isolation. `pytest`
passes from the repo root.

This is slice 2/2 of issue #22 (issue #30). Slice 1 (the `cries.py` module) is
assumed already merged.

## Inputs
No new CLI inputs, flags, or config. The cry call is fed entirely from data
already available inside the loop:

- `stages[0]["name"]` — the line name (stage 1's name), used verbatim as the
  `line_name` seed shared across the whole evolution line.
- `stage["stage"]` — the current stage's integer evolution stage (`>= 1`).
- `stage["types"]` — the current stage's type list (`types[0]` selects the cry
  profile inside `generate_cry`; empty/unknown falls back to a default).
- `stage_dir` — the current stage's output directory (a `pathlib.Path`),
  already created by `write_output`; the WAV is written to
  `str(stage_dir / "cry.wav")`.

## Outputs
- One `cry.wav` file written into each stage's output directory, sibling of
  `sprite.png` (mono, 8-bit unsigned PCM, 10512 Hz — produced by
  `generate_cry`; this slice does not define the audio format).
- On failure of a stage's cry generation: a single line to `stderr` of the
  form `Warning: cry generation failed for {stage['name']}: {exc}` and the loop
  continues.
- No change to stdout's final `Done!` line, to `.ini` contents, or to any other
  asset. `export_ini` is intentionally left untouched — no `.ini` key
  references the cry file (out of scope).

## Behavior
1. Add `from fakemon_forge.cries import generate_cry` to the top-of-file
   imports in `main.py`, alongside the other collaborator imports.
2. Inside the existing `for stage, stage_dir in zip(stages, stage_dirs):`
   loop, add a `generate_cry` block **before** the sprite generation block.
3. The block calls:
   ```
   generate_cry(
       stages[0]["name"],            # line_name — stage 1's name, shared by the line
       stage["stage"],
       stage["types"],
       str(stage_dir / "cry.wav"),
   )
   ```
   wrapped in `try/except Exception as exc:` that prints the warning shown
   under Outputs to `stderr` and does not re-raise.
4. Because the block sits ahead of the sprite block, it runs on every stage
   regardless of whether sprite generation later raises and hits its `continue`.
5. Everything else in the loop (front sprite, back sprite, frame 2, shinies,
   spritesheet) and the subsequent `export_ini` loop is unchanged.

Per-stage, per-line semantics (owned by `cries.py`, restated for the caller's
contract): the whole line shares `line_name = stages[0]["name"]` as its voice
seed; the varying arguments between stages are `stage["stage"]` and
`stage["types"]`. In single-stage mode the loop runs once; in line mode it runs
once per stage, so 1 vs 3 `generate_cry` calls respectively.

## Edge cases
- **Sprite-failure independence (the regression this placement guards):** when
  `generate_sprite` / `generate_sprite_img2img` raises, the sprite block hits
  its `continue` and skips the rest of that stage. Because the cry block is
  placed *before* the sprite block, the cry is still generated for that stage.
  This must be preserved; do not remove or weaken the sprite `continue`.
- **Cry-failure isolation:** if `generate_cry` raises, only a warning is
  printed and the loop proceeds to the sprite block and the remaining assets
  for that stage; `main` never raises from a cry failure.
- **Line name shared across stages:** every stage's cry uses `stages[0]["name"]`
  ("Flamburr" in the test fixtures), not `stage["name"]`, so the entire line
  shares one voice.
- **Empty / unknown types:** delegated to `generate_cry` (falls back to the
  default profile); no special handling in `main.py`.
- **Empty `stages`:** not introduced by this slice — the loop already does not
  execute on an empty list, and `write_output` / `stage_dirs[0]` behavior is
  unchanged.

## Errors
- Any exception from `generate_cry` is caught by the block's
  `except Exception as exc:` and reported as
  `Warning: cry generation failed for {stage['name']}: {exc}` on `stderr`; the
  program continues and exits normally. This mirrors every other asset block in
  the loop (sprite, back sprite, frame 2, shinies, spritesheet).
- No new `sys.exit` paths, no new exception types, and the missing-API-key exit
  path is untouched.

## Constraints & dependencies
- Only `fakemon_forge/main.py` (implementation) and `tests/test_main.py`
  (tests) change. `cries.py` is not modified.
- `generate_cry` is stdlib-only (no torch/diffusers/GPU/network), so the new
  code path and its tests need no ML stack. The new tests belong in
  `tests/test_main.py`, where every collaborator is already mocked via
  `patch("fakemon_forge.main.<name>")`, so they run without torch and are not
  `ml`-marked.
- The warning message wording and the `file=sys.stderr` convention must match
  the existing blocks so failure reporting stays uniform.
- Cry generation is placed strictly before the sprite block so it is never
  gated by the sprite `continue`.

## Tests
Extend `tests/test_main.py` in the existing style (patch collaborators, drive
`main([...])` end to end). Concretely:

- Add `patch("fakemon_forge.main.generate_cry")` to both the `ctx` and
  `ctx_line` fixtures' `with (...)` patch blocks and expose the mock in the
  yielded dict (e.g. `ctx["cry"]`, `ctx_line["cry"]`).
- **Single-stage call:** after a description-only run, assert `generate_cry`
  was called once and its arguments are `line_name == stages[0]["name"]`
  (`"Flamburr"`), `stage == 1` (the stage int), `types == ["Fire"]`, and
  `output_path == str(stage_dir / "cry.wav")`.
- **Line-mode fan-out:** with 3 stages, assert `generate_cry` called 3 times,
  once per stage, each with `line_name == "Flamburr"` (stage 1's name) and the
  correct per-stage `stage` value (1, 2, 3) written into each stage's own
  `stage_dir / "cry.wav"`.
- **Sprite-failure independence:** set `ctx["sprite"].side_effect =
  RuntimeError(...)`; assert `main` does not raise and `generate_cry` is still
  called for that stage (guards the placement-before-sprite regression).
- **Cry-failure isolation:** set the cry mock's `side_effect` to raise; assert
  `main` does not raise, prints a `Warning` on `stderr` mentioning the stage
  name (`"Flamburr"`), and still runs the sprite/other-asset blocks (e.g.
  `generate_sprite` was still called).

## Assumptions
Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** `line_name = stages[0]["name"]` and `stage = stage["stage"]` —
  the whole line shares stage 1's name as its voice seed, matching the
  `generate_cry` docstring contract. (Consistent with the module docs, not an
  independently confirmed product decision.)
- **[picked]** Output filename is `cry.wav` in the stage directory, sibling of
  `sprite.png`. (Matches the sibling-asset convention; no code currently names
  the cry file.)
- **[picked]** The cry block is placed **immediately before** the sprite block,
  so it runs even when sprite generation `continue`s on failure. (Placement
  before the sprite block is mandated by the task; "immediately before" is the
  specific choice.)
- **[picked]** Warning wording is
  `Warning: cry generation failed for {stage['name']}: {exc}` printed to
  `sys.stderr`, matching every other asset block verbatim in structure.
- **[picked]** `export_ini` is intentionally left untouched — no `.ini` key
  references the cry file (out of scope for this slice).
- **[confirmed]** `generate_cry` exists in `fakemon_forge/cries.py` with the
  signature `generate_cry(line_name: str, stage: int, types: list, output_path:
  str) -> None`, is pure stdlib, and writes the WAV to `output_path` verbatim.
- **[confirmed]** Every collaborator in `main` is patched via
  `patch("fakemon_forge.main.<name>")` in `tests/test_main.py`, so patching
  `generate_cry` the same way (and keeping the tests un-`ml`-marked) fits the
  established test pattern.
- **[confirmed]** The sprite block ends with a `continue` on failure, and every
  other asset block uses the identical `except Exception as exc:` warn-to
  -`stderr` pattern — the new block follows that pattern.
