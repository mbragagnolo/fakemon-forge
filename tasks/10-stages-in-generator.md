# Task 10 — Thread `stages` through the generator

- **Status:** pending
- **Wave:** 1
- **Owns (the only files/dirs this task may create or modify):**
  - `fakemon_forge/generator.py` — everything **except** the `_BST_TARGETS` constant (owned by task 00)
  - `tests/test_generator.py`
- **Depends on:** 00
- **Parallel-safe with:** 11
- **Implements spec sections:** Inputs (the `stages` parameter); Behavior §4 (prompt), §5 (`_size_defaults`); Edge cases (rows 1, 5, 6); Testing (prompt target counts, progression wording, size mapping, byte-identical default)

## Goal

Make the generator aware of how many stages it is being asked for. Today
`_user_prompt` branches only on `mode == "single"` and otherwise hardcodes three
stages, in both the BST hint and the evolution-progression text; `_size_defaults`
assumes stage 2 is always a middle form.

## Contract (the public interface this task must honor / expose)

**Consumes** `_BST_TARGETS` from task 00, keyed `tier → stage-count → targets`.
Do not reshape it.

**Exposes** the signature tasks 11 and 20 code against — match it exactly:

```python
def generate_fakemon(
    description: str,
    mode: str,
    tier: str = "standard",
    *,
    stages: int = 3,
    client=None,
    api_key: str = None,
) -> list[dict]:
```

`stages` is keyword-only and defaults to `3`. The default is what makes
`--mode line` with no `--stages` behave exactly as today. `stages` is ignored
when `mode == "single"`.

`_user_prompt(description, mode, tier, stages)` and
`_size_defaults(stage, mode, tier, stages)` take the same value; their exact
parameter style is yours, but keep them internal.

## TDD steps (red → green → refactor)

1. **Red — write failing tests first** in `tests/test_generator.py`:
   - **Byte-identical default (highest value test in this task).** The full
     prompt produced by `generate_fakemon(..., "line")` with no `stages` argument
     is character-for-character identical to the prompt produced before this
     change. Capture today's string and assert equality — not a substring check.
   - `--mode line --stages 2` puts exactly **two** BST targets in the prompt;
     `--stages 3` puts three; `--mode single` puts one.
   - The 2-stage BST hint carries 305 and 468; the 3-stage hint carries
     295 / 405 / 518; standard single carries 430.
   - 2-stage progression wording (juvenile → adult, no adolescent middle) appears
     for `--stages 2` and does **not** appear for `--stages 3`; the existing
     three-stage wording appears for `--stages 3` and not for `--stages 2`.
   - `_size_defaults` maps a 2-stage **stage 2** to the **stage-3** size row
     (17 dm / 600 hg), not the stage-2 row — a 2-stage final is a final form.
   - `_size_defaults` for a 3-stage line is unchanged (5/30, 10/150, 17/600).
   - `stages` is ignored in single mode: `generate_fakemon(..., "single", stages=2)`
     produces the same prompt as `stages=3`.
   - Regression: the existing off-spec-stage-number fallback still holds — an
     out-of-range or non-integer `stage` still falls through to the tier table
     rather than raising (shipped in #55).
2. **Green — implement** until those tests pass.
3. **Refactor** with tests green.

## Test requirements

Focused unit tests, each naming the spec behavior it pins. All mocked — no real
API call. Non-ML: no torch, no `@pytest.mark.ml`; these must run in the keep
sandbox.

Do not weaken any existing test in `tests/test_generator.py` to make room. If one
genuinely encodes old behavior that this task changes, update it and call that
out explicitly in the handoff.

## Acceptance criteria (Definition of Done)

- [ ] `generate_fakemon` accepts keyword-only `stages: int = 3`.
- [ ] `--mode line` with no `stages` produces a byte-identical prompt to before.
- [ ] Prompt carries 1 / 2 / 3 BST targets for single / 2-stage / 3-stage.
- [ ] 2-stage progression wording is distinct from the 3-stage wording.
- [ ] A 2-stage final takes the stage-3 size row; 3-stage sizes are unchanged.
- [ ] `stages` has no effect in single mode.
- [ ] `_BST_TARGETS` was not reshaped or re-valued by this task.
- [ ] `stats.json` keys and the `stageN_<name>` layout are unchanged.
- [ ] Full suite green: `pytest` from the repo root.

## Notes / assumptions

- **[assumption, from the spec]** No stage-count validation of the model's
  response. If the model returns three stages when two were requested, that is
  accepted — `_normalize` already tolerates any list length, and the 2-attempt
  retry budget is spent on the name contract. Do not add a retry here.
- If task 00 left a temporary adapter in `_user_prompt` to keep the suite green,
  remove it as part of this task.
