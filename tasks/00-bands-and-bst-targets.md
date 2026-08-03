# Task 00 — Band fixture + `_BST_TARGETS` restructure

- **Status:** pending
- **Wave:** 0
- **Owns (the only files/dirs this task may create or modify):**
  - `tests/fixtures/gen3_bst_bands.json` (new)
  - `fakemon_forge/generator.py` — **only** the `_BST_TARGETS` constant
  - `tests/test_bst_targets.py` (new)
- **Depends on:** none
- **Parallel-safe with:** none (sequential wave)
- **Implements spec sections:** Behavior §2 (the committed fixture), §3 (`_BST_TARGETS`); Testing (band conformance, median equality, the 21-member assertion)

## Goal

Freeze the two contracts every later task codes against: the aggregate band
fixture, and the reshaped `_BST_TARGETS` lookup. The bands were already derived
and validated by the #59 spike — this task only *commits* them and proves the
constants conform. No derivation work happens here.

## Contract (the public interface this task must honor / expose)

**Exposes 1 — `tests/fixtures/gen3_bst_bands.json`.** Aggregate statistics only:

```json
{
  "2": { "0": {"median": 305, "p10": 240, "p90": 360, "n": 84},
         "1": {"median": 468, "p10": 410, "p90": 515, "n": 84} },
  "3": { "0": {"median": 295, "p10": 205, "p90": 314, "n": 37},
         "1": {"median": 405, "p10": 278, "p90": 420, "n": 37},
         "2": {"median": 518, "p10": 450, "p90": 600, "n": 37} },
  "standalone": {"median": 430, "p10": 336, "p90": 500, "n": 54},
  "legendary":  {"median": 580, "p10": 580, "p90": 580, "n": 9},
  "mythical":   {"median": 600, "p10": 600, "p90": 600, "n": 6}
}
```

Outer keys `"2"` / `"3"` are stage counts; inner keys are **0-based** stage
indexes. These values are final — do not recompute them.

**Exposes 2 — `_BST_TARGETS`,** keyed `tier → stage-count → per-stage targets`,
with the standalone value alongside. The exact literal structure is yours to
choose, but it must support these four lookups, and task 10 depends on that:

| lookup | value |
|---|---|
| standard, single | **430** |
| standard, 2-stage | **305 / 468** |
| standard, 3-stage | **295 / 405 / 518** |
| pseudo, 3-stage | **300 / 420 / 600** (unchanged) |
| legendary, single | **580** (unchanged) |
| mythical, single | **600** (unchanged) |

`pseudo` has **no** 2-stage row — that combination is rejected at the CLI
(task 11).

## TDD steps (red → green → refactor)

1. **Red — write failing tests first** in `tests/test_bst_targets.py`:
   - Every `_BST_TARGETS` value equals its band's `median` exactly. This is the
     rule the table follows; it is stricter than the band check and catches a
     value hand-edited to something still in-band.
   - Every `_BST_TARGETS` value lies within its band's `[p10, p90]`.
   - Targets increase monotonically across the stages of every line row.
   - The fixture's `legendary["n"] + mythical["n"]` is **exactly 21**. This is
     the placeholder-filter check — a fixture regenerated without the exclusion
     step reads 46 here and fails loudly instead of silently shifting bands.
   - Standard single (430) is **strictly greater than** standard 3-stage stage 1,
     pinning the #48 consistency fix (a standalone form is not a juvenile).
   - The 2-stage final (468) is strictly greater than the 3-stage *mid*-stage
     (405) — a 2-stage stage 2 is a final form, not a middle one.
2. **Green — implement:** add the fixture, reshape `_BST_TARGETS`, correct the
   values.
3. **Refactor** with tests green.

## Test requirements

Tests load the fixture from disk with `pathlib` relative to the test file — no
new runtime file I/O in `fakemon_forge/`; the package must never read this file.

**Hard constraint:** neither the fixture nor any test may contain a species name,
a species ID, a ROM offset, or a placeholder-slot range. Aggregate numbers only.
This is what keeps the `ac5cf2f` / `0ae9a1d` scrub intact.

## Acceptance criteria (Definition of Done)

- [ ] `tests/fixtures/gen3_bst_bands.json` exists with exactly the values above.
- [ ] `_BST_TARGETS` is keyed by tier → stage count and supports all four lookups.
- [ ] standard single is 430; 3-stage is 295 / 405 / 518; 2-stage is 305 / 468.
- [ ] pseudo (300/420/600), legendary (580) and mythical (600) are unchanged.
- [ ] Every target equals its band median, and lies within `[p10, p90]`.
- [ ] The legendary + mythical band count asserts to exactly 21.
- [ ] No species names, IDs, ROM offsets or slot ranges anywhere in the added files.
- [ ] `fakemon_forge/` gained no file reads and no new dependency.
- [ ] Full suite green: `pytest` from the repo root.

## Notes / assumptions

- Only `_BST_TARGETS` may be touched in `generator.py`. `_user_prompt`,
  `_size_defaults` and `generate_fakemon` belong to task 10 — leaving them alone
  keeps this wave's diff reviewable and avoids a collision.
- Reshaping `_BST_TARGETS` will break `_user_prompt`, which currently reads
  `targets['stage1']`. Task 10 fixes that. If the suite cannot be left green
  here, apply the **minimum** adapter inside `_user_prompt` to keep today's
  output byte-identical, and say so in the handoff — do not implement task 10's
  behavior.
- The bands are already derived and validated (ten values cross-checked against
  the public Serebii listing, all matching). Do not re-derive them; the
  derivation utility belongs in the private injector repo.
