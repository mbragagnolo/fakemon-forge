# Task 11 — `--stages` flag and its two validations

- **Status:** done
- **Wave:** 1
- **Owns (the only files/dirs this task may create or modify):**
  - `fakemon_forge/cli.py`
  - `tests/test_cli.py`
- **Depends on:** 00
- **Parallel-safe with:** 10
- **Implements spec sections:** Inputs (the `--stages` flag); Errors (both new validations); Edge cases (rows 1–4)

## Goal

Add the `--stages {2,3}` flag and the two rejections that keep impossible
shape/tier combinations from reaching the generator. This task owns only
argument parsing and validation — it never calls `generate_fakemon`.

## Contract (the public interface this task must honor / expose)

**Exposes** `args.stages: int`, default `3`, from `parse_args`. Task 20 reads it
to build the `generate_fakemon(..., stages=args.stages)` call.

`--mode` keeps its existing `single | line` choices — do **not** add shape values
to it. Additive `--stages` is what keeps every existing invocation working.

## TDD steps (red → green → refactor)

1. **Red — write failing tests first** in `tests/test_cli.py`:
   - `parse_args([])` yields `stages == 3` (the default that preserves today's
     behavior).
   - `--stages 2` and `--stages 3` parse to the int `2` / `3`, not strings.
   - `--stages 1` and `--stages 4` are rejected by argparse `choices`
     (`SystemExit`). `--stages 1` is deliberately **not** a synonym for
     `--mode single`.
   - `--mode single --stages 2` exits 1 with a message saying `--stages` applies
     only to `--mode line`.
   - `--tier pseudo --mode line --stages 2` exits 1 with a message saying
     pseudo-legendary lines are always 3 stages.
   - `--tier pseudo --mode line --stages 3` is **accepted** — the guard must not
     over-reject.
   - `--tier pseudo --mode line` with no `--stages` is accepted (defaults to 3).
   - Regression: the existing `--tier legendary/mythical --mode line` rejection
     at `cli.py:34` still fires, and its message is unchanged.
   - Errors go to **stderr** with exit code 1 and no traceback, matching the
     existing precedent.
2. **Green — implement** until those tests pass.
3. **Refactor** with tests green.

## Test requirements

Follow the existing `tests/test_cli.py` patterns for asserting exits and stderr
text. Non-ML: no torch, no `@pytest.mark.ml`.

Validation order matters when several rules could fire at once: decide it
deliberately and pin it with a test (e.g. `--mode single --tier pseudo --stages 2`
should produce exactly one message, and always the same one).

## Acceptance criteria (Definition of Done)

- [x] `--stages` accepts only `2` and `3`, defaults to `3`, and yields an int.
- [x] `--stages` with `--mode single` exits 1 with a clear stderr message —
      including an explicit `--stages 3`, since the fault is supplying a flag
      that does not apply to the mode.
- [x] `--tier pseudo --stages 2` exits 1 with a clear stderr message.
- [x] `--tier pseudo` with 3 stages (explicit or default) is accepted.
- [x] `--stages 1` is rejected — not treated as `--mode single`.
- [x] The existing legendary/mythical rejection is unchanged.
- [x] `--mode` choices are unchanged (`single`, `line`).
- [x] Full suite green: `pytest` from the repo root — 630 passed (was 601).

## Notes / assumptions

- **[default, from the spec]** The additive-flag surface was a recommended
  default, not an explicit user choice. The alternative — enumerating shapes in
  `--mode` (`line2` / `line3`) — was rejected because it breaks existing
  `--mode line` invocations. If this proves awkward, raise it rather than
  redesigning the flag mid-task.
- **[open, from the spec]** `--stages 1` as a synonym for `--mode single` is
  assumed **no**: two ways to say one thing invites drift. The test above pins
  that decision.
- Do not touch `fakemon_forge/generator.py` — task 10 owns it, and these two run
  in the same wave.

### Outcomes worth carrying forward

- **`args.stages_given` was added** alongside `args.stages`. `--stages` parses
  with `default=None` so the flag's *presence* can be distinguished from its
  value; `parse_args` records the flag then fills in the default, keeping the
  frozen contract (`args.stages` is always an int, default 3) intact. Without
  it the single-mode rejection could not tell an explicit `--stages 3` from the
  default and would have fired on every single-mode run.
- **`--tier pseudo --mode single` is now rejected** — scope beyond the spec,
  invited by this task's Notes. `README.md:107` and `--tier`'s own help already
  called pseudo a line, but nothing enforced it; it produced a standalone form
  carrying a juvenile's BST, the exact inconsistency #59 removes for the
  standard tier. **`generator._bst_row`'s fallback is now unreachable for
  pseudo but still guards other tiers — do not remove it.**
- **Validation order is pinned by test**, not incidental: unusable run →
  flag-does-not-apply-to-mode → tier/mode → tier/stage-count. Exactly one
  message is emitted when several rules could fire.
- **`--mode`'s help text no longer says "3-stage"**, since a line can now be
  either length. `README.md` still does — that is task 30's.
