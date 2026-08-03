# Task 20 — Wire `--stages` through `main.py`

- **Status:** done
- **Wave:** 2
- **Owns (the only files/dirs this task may create or modify):**
  - `fakemon_forge/main.py`
  - `tests/test_main.py`
- **Depends on:** 10, 11
- **Parallel-safe with:** none (sole task in its wave)
- **Implements spec sections:** Inputs (internal threading); Outputs (unchanged shape); Constraints (injector contract)

## Goal

Connect the two halves: the CLI now produces `args.stages`, the generator now
accepts `stages`, and `main.py:54` is the single line that still drops it on the
floor. This is the seam, and it is small by design.

## Contract (the public interface this task must honor / expose)

Consumes both Wave-1 surfaces exactly as frozen:

```python
# fakemon_forge/main.py — currently line 54
stages = generate_fakemon(combined, args.mode, tier=args.tier, client=client)
#   becomes
stages = generate_fakemon(
    combined, args.mode, tier=args.tier, stages=args.stages, client=client
)
```

Beware the name collision already present in `main.py`: the local variable
`stages` holds the returned **list of stage dicts**, while `args.stages` is the
requested **count**. Do not let the two shadow each other — rename the local if
that reads more clearly, but keep the change contained to this file.

## TDD steps (red → green → refactor)

1. **Red — write failing tests first** in `tests/test_main.py`:
   - `main` passes `stages=args.stages` through to `generate_fakemon` — assert on
     the mock's call kwargs for `--stages 2` and for `--stages 3`.
   - With no `--stages`, `generate_fakemon` receives `stages=3`.
   - A 2-stage run writes exactly **two** stage directories, named
     `stage1_<name>` and `stage2_<name>` — the injector's contract, unchanged.
   - A 2-stage run does **not** create a `stage3_*` directory.
   - The per-stage asset loop (sprite / icon / footprint / cry) runs exactly twice
     for a 2-stage line and three times for a 3-stage line.
   - Regression: a default `--mode line` run is unchanged end to end — three stage
     directories, same call counts as before this change.
2. **Green — implement** until those tests pass.
3. **Refactor** with tests green.

## Test requirements

`tests/test_main.py` already mocks the sprite/ML functions — keep it that way so
this stays torch-free and runs in the keep sandbox. No `@pytest.mark.ml`.

## Acceptance criteria (Definition of Done)

- [x] `args.stages` reaches `generate_fakemon` as the `stages` keyword.
- [x] Omitting `--stages` sends `stages=3`.
- [x] A 2-stage run produces exactly two `stageN_<name>` directories, no `stage3_*`.
- [x] The asset loop iterates once per generated stage, no more — asserted for
      sprite, icon, cry and footprint at both 2 and 3 stages.
- [x] `stats.json` keys and the `stageN_<name>` naming are unchanged — the
      injector reads these and must keep working. A test asserts every written
      directory still matches `stage<digits>_`.
- [x] The local-variable / `args.stages` name collision is resolved
      unambiguously — the local is now `forms`.
- [x] Full suite green: `pytest` from the repo root — 648 passed (was 630).

## Notes / assumptions

- The injector consumes 2-stage lines already (its evolution-level fallback
  carries a 2-stage entry, and it pairs forge lines to vanilla lines by
  evolution length), so no injector-side change is needed for this shape. Do not
  add one, and do not alter the directory naming to signal stage count.
- Branched evolution is explicitly out of scope for #59. Do not introduce
  `stage2a_*`-style names — the injector filters directories that don't match
  `stage<digits>_`, so such output would be silently dropped.

### Outcomes worth carrying forward

- **The local is now `forms`, not `stages`.** All four uses were renamed
  (`write_output`, the `zip`, the `forms[0]["name"]` line-name lookup, and the
  footprint branch). This was the collision the task file predicted, and the
  same one that silently broke `generator.py` in task 10.
- **A second, unplanned defect was found and fixed: footprint scaling.**
  `main.py` scaled footprint size by stage position only when
  `len(stages) == 3`, so a 2-stage line fell through to a flat full-size
  footprint and its juvenile printed the same size as its final. Replaced with
  `_FOOTPRINT_FRACTIONS`, keyed line length → stage number: a 2-stage line
  takes `{1: 0.6, 2: 0.9}`, mirroring how task 10 gave a 2-stage final the
  stage-3 height/weight row. The 3-stage and single-form fractions are
  unchanged, and an unrecognised line length still falls through to full size.
  *(User decision, asked during execution — the alternative considered was an
  even split of 0.6 / 0.75, rejected because it contradicts the height/weight
  treatment already shipped.)*
- **`main.py` is now stage-count agnostic.** Nothing left in it branches on a
  line being exactly three stages; everything sizes off `len(forms)`.
