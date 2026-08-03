# Task 30 — Hardening, e2e, and docs

- **Status:** pending
- **Wave:** 3
- **Owns (the only files/dirs this task may create or modify):**
  - `README.md`
  - `tests/test_stages_e2e.py` (new)
- **Depends on:** 20
- **Parallel-safe with:** none (sole task in its wave)
- **Implements spec sections:** Testing (happy-path e2e, the byte-identical guarantee); Edge cases (whole table); Constraints (no species data)

## Goal

Prove the four pieces work together, pin the guarantees that are easy to break
later, and bring the docs in line. `README.md` currently describes `--mode line`
as "a full 3-stage evolutionary line" in several places, which is now wrong.

## Contract (the public interface this task must honor / expose)

Consumes the finished CLI → generator → main path. Adds no new production
surface; this task writes tests and docs only. It must not modify
`generator.py`, `cli.py` or `main.py` — if a real defect turns up, report it and
fix it in the owning task rather than reaching across the boundary here.

## TDD steps (red → green → refactor)

1. **Red — write failing tests first** in `tests/test_stages_e2e.py`:
   - **Happy path, 2 stages.** Drive `main` with `--mode line --stages 2` and a
     mocked client; assert two stage directories, each with a valid `stats.json`
     and `entry.md`, and that both `.ini` exports succeed.
   - **Happy path, 3 stages** — unchanged from today.
   - **The byte-identical guarantee, at full-prompt level.** `--mode line` with no
     `--stages` sends a prompt character-for-character identical to the
     pre-change one. Task 10 pins this at unit level; pin it here through the
     whole CLI → generator path, because this is the regression most likely to be
     broken by a later well-meaning prompt edit.
   - **Both CLI rejections** surface correctly when driven through `main`, not
     just `validate_args`: `--mode single --stages 2` and
     `--tier pseudo --stages 2` each exit 1.
   - **Spec edge case, pinned as intended behavior:** a model returning 3 stages
     when 2 were requested is *accepted*, not retried or truncated. Assert three
     directories are written. This is a deliberate non-goal, and the test exists
     so it is not silently "fixed" later.
   - **No species data leak.** Assert no file under `tests/fixtures/` matches a
     species name, a species ID list, or a ROM offset pattern. Cheap grep-style
     check; guards the `ac5cf2f` / `0ae9a1d` scrub against a future regenerated
     fixture.
2. **Green** — the code should already pass; any failure is a real defect in an
   earlier task. Report it and route the fix to that task's file.
3. **Refactor** with tests green.

## Test requirements

Fully mocked — no real API call, no real image generation. Non-ML: no torch, no
`@pytest.mark.ml`. The whole file must run in the keep sandbox.

## Docs — `README.md`

Update every place that assumes a line is three stages:

| Line | What is wrong now |
|---|---|
| ~21, ~23 | `stage2_/stage3_` annotated "only with `--mode line`" — `stage3_` is now only with `--stages 3` |
| ~90 | Usage synopsis omits `--stages` |
| ~99 | `--mode` described as "`line` — full 3-stage evolutionary line" |
| ~107 | Tier table says pseudo is "only valid with `--mode line`" — now also requires 3 stages |

Also add `--stages` to the flag table with its default, and one worked example
of a 2-stage line alongside the existing examples.

## Acceptance criteria (Definition of Done)

- [ ] 2-stage and 3-stage happy paths both pass end to end through `main`.
- [ ] The byte-identical default prompt is pinned at full-prompt level.
- [ ] Both CLI rejections are exercised through `main`.
- [ ] The "model returns the wrong stage count" non-goal is pinned as accepted.
- [ ] A species-data leak check covers `tests/fixtures/`.
- [ ] `README.md` documents `--stages`, and no longer claims a line is always
      3 stages in any of the listed places.
- [ ] No production file (`generator.py`, `cli.py`, `main.py`) was modified.
- [ ] Full suite green: `pytest` from the repo root.

## Notes / assumptions

- Branched evolution is out of scope and documented as deferred in the spec. Do
  not document it in `README.md` as forthcoming — it depends on injector work
  that has not been scheduled.
- Do not document the ROM, the injector, or placeholder-slot internals in
  `README.md`. This is the public-facing portfolio surface, and prior commits
  deliberately scrubbed exactly that material.
