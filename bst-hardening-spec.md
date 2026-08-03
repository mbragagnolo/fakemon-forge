# Spec: Harden BST targets against observed Gen 3 values; support 2-stage lines

> Tracking: #59

## Summary

`_BST_TARGETS` carries hand-picked numbers never checked against real Gen 3 data,
and `mode` is binary (`single` | 3-stage `line`) so 2-stage lines cannot be
generated. This work re-derives every BST target from observed game data, pins
them with a test-only aggregate fixture, fixes the `--mode single` target to
describe a standalone species rather than a juvenile, and adds `--stages {2,3}`.

Branched evolution is explicitly **out of scope** — see "Deferred" below.

## Inputs

CLI surface (`fakemon_forge/cli.py`):

| Flag | Type | Required? | Description |
|---|---|---|---|
| `--mode` | `single` \| `line` | no, default `single` | Unchanged values. `line` no longer implies 3 stages on its own. |
| `--stages` | `2` \| `3` | no, default `3` | **New.** Number of stages when `--mode line`. Rejected with `--mode single` (see Errors). |
| `--tier` | `standard` \| `pseudo` \| `legendary` \| `mythical` | no, default `standard` | Unchanged set — no new tier. |

Internal: `generate_fakemon(description, mode, tier, *, client, api_key)` gains a
`stages: int = 3` keyword parameter. `_user_prompt` and `_size_defaults` take the
same value.

**Default preserved:** `--mode line` with no `--stages` behaves exactly as today
(3 stages), so no existing invocation changes behaviour.

## Outputs

- No change to the shape of `generate_fakemon`'s return value: still a flat
  `list[dict]`, one element per stage, length 1, 2 or 3.
- No change to `stats.json` keys, `write_output`'s `stageN_<name>` directory
  naming, or the `.ini` export.
- The only observable change is the **BST hint text in the prompt**, and hence
  the stat spread the model produces.

## Behavior

### 1. Deriving the observed bands (one-off, offline)

Not part of the shipped package. A throwaway script, run once by the operator:

1. Load a BPEE ROM with the injector's `Rom`.
2. `lines = mapping.derive_vanilla_lines(rom)`.
3. For each **linear** line, for each stage index `i` in `species_ids`, record
   `mapping.bst(rom, species_id)` into a sample bucket keyed `(len(line), i)`.
4. Non-linear lines are skipped entirely — they are the deferred branched case
   and would distort the linear bands.
5. Single-member lines (`len == 1`) form the standalone-species bucket. Split
   that bucket by BST threshold to separate ordinary no-evolution species from
   legendary/mythical ones (the two clusters are widely separated; the exact
   split point is an implementation detail to be reported, not assumed here).
6. Emit `median`, `p10`, `p90` and sample count `n` per bucket.

Cross-check a handful of derived values against
`https://www.serebii.net/pokedex-rs/stat/all.shtml` before committing.

**Only the aggregate output is committed.** No species names, no IDs, no ROM.
This is what keeps the `ac5cf2f` / `0ae9a1d` scrub intact.

### 2. The committed fixture

`tests/fixtures/gen3_bst_bands.json`, consumed **only** by tests:

```
{ "<stage_count>": { "<stage_index>": {"median": int, "p10": int, "p90": int, "n": int} },
  "standalone":    {"median": int, "p10": int, "p90": int, "n": int},
  "legendary":     {...},
  "mythical":      {...} }
```

Runtime code never reads this file. `_BST_TARGETS` stays a plain literal in
`generator.py`, so the package acquires no new file I/O and no new dependency.

### 3. `_BST_TARGETS` restructure

Today the table is keyed `tier → {stage1, stage2, stage3}`, which cannot express
a 2-stage line. It becomes keyed by **tier → stage-count → per-stage targets**,
with the standalone (single) value alongside:

- `standard` gains a 2-stage row and a 3-stage row; its single value becomes the
  standalone-species median, **not** the stage-1 value.
- `pseudo` keeps a 3-stage row only (see Errors — 2 stages is rejected).
- `legendary` / `mythical` are single-form only, as today.

Every number is set to the **median** of its band. Medians rather than means:
the buckets contain outliers in both directions (cocoon mid-stages far below,
pseudo-legendary finals far above) that would drag a mean off the typical value.

### 4. Prompt

`_user_prompt` composes the BST hint from the resolved row:

- `--mode single` → one target.
- `--mode line --stages 2` → two targets, base and final.
- `--mode line --stages 3` → three targets (today's wording).

The stage-count also drives the existing `_EVO_PROGRESSION` text, which currently
hardcodes three stages. A 2-stage line needs its own progression wording
(juvenile → adult, with no adolescent middle).

### 5. `_size_defaults`

`_SIZE_DEFAULTS_BY_LINE_STAGE` is keyed by stage number 1/2/3. For a 2-stage
line, stage 2 is a *final* form, not a middle one, so reusing the stage-2 row
(10 dm / 150 hg) would under-size it. A 2-stage line maps stage 1 → the stage-1
row and stage 2 → the **stage-3** row.

## Edge cases

| Case | Handling |
|---|---|
| `--mode line` with no `--stages` | 3 stages — today's behaviour exactly |
| `--mode single --stages 2` | Rejected (see Errors) |
| `--tier pseudo --stages 2` | Rejected (see Errors) |
| `--tier legendary/mythical --mode line` | Already rejected today; unchanged |
| Model returns 3 stages when 2 were asked for | Out of scope — no stage-count validation or retry is added. `_normalize` already tolerates any list length |
| Model returns a `stage` number outside the requested count | Already handled: `_size_defaults` falls through to the tier table (shipped in #55) |
| Band fixture missing/corrupt | Test-time failure only; runtime is unaffected |
| A derived band has a tiny sample count | Recorded as `n` in the fixture so a reviewer can see which bands are thin; no automatic behaviour |

## Errors

No new runtime failure modes in `generator.py` — the changes are prompt text and
constant lookups, both total.

Two new CLI validations, following the existing `cli.py:34` precedent (print to
stderr, `sys.exit(1)`, no traceback):

- `--stages` given with `--mode single` → "`--stages` applies only to `--mode line`".
- `--tier pseudo --stages 2` → "pseudo-legendary lines are always 3 stages".

## Constraints & dependencies

- Python, no new third-party dependencies; no new runtime file reads.
- **No real species names, IDs or per-species game data in this repo.** The
  committed artifact is aggregate statistics only.
- The derivation script depends on the private `fakemon-rom-injector` and an
  operator-supplied ROM. Neither is a dependency of `fakemon-forge`, and the
  script is not committed here.
- Non-ML throughout: no torch/diffusers, no `@pytest.mark.ml` (per `CLAUDE.md`,
  these tests must run in the keep sandbox).
- `stats.json` and the `stageN_<name>` layout are the injector's contract and
  must not change.

## Testing

- `_BST_TARGETS` values all fall within `[p10, p90]` of their matching band —
  the assertion that makes the constants "hardened" rather than merely edited.
- Targets increase monotonically across the stages of a line.
- A 2-stage line's final target exceeds its own stage-1 target and is
  meaningfully above the 3-stage *mid*-stage target (a 2-stage final is a final).
- `--mode single --tier standard` target is the standalone value, and is
  strictly greater than the 3-stage stage-1 target (the #48 consistency fix).
- `--mode line` without `--stages` produces a byte-identical prompt to today's.
- Prompt contains the right number of BST targets for each `(mode, stages)`.
- 2-stage progression wording appears for `--stages 2` and not for `--stages 3`.
- `_size_defaults` maps a 2-stage final to the stage-3 size row.
- Both new CLI rejections exit 1 with the expected message.

## Deferred

**Branched evolution (Eevee / Wurmple / Clamperl-style).** Cut after inspecting
the injector, which cannot consume it:

- `_write_evolution` writes a *single* method-4 (level-up) edge per non-final
  stage and zeroes records 1–4 — one target per species, never a branch.
- Line pairing matches forge lines to **linear** vanilla paths, grouped by
  evolution length; a branched forge line has nothing to pair against.
- `forge_final_stage_bst` takes the single highest-`N` stage dir as *the* final
  form.
- `_STAGE_DIR_RE` is `^stage(\d+)_`, and non-matching directories are **filtered
  out** rather than rejected — so `stage2a_*` output would be *silently dropped*,
  which is the worst available failure mode.

Groundwork does exist on the injector's read side (`VanillaLine.linear`,
`leaf_ids`, "max leaf BST" for branched lines), so this is a real future
direction. It needs a companion issue on `fakemon-rom-injector` covering the
multi-edge write, non-linear pairing and multi-leaf handling, then a follow-up
here to emit the shape.

Also deferred: a fifth "box legendary" tier for the higher legendary band. The
existing four tier names are kept.

## Open questions / assumptions

- **[default]** CLI surface (`--stages {2,3}` alongside an unchanged `--mode`)
  was a recommended default — no preference was expressed. The alternative
  considered was enumerating shapes in `--mode` (`line2`/`line3`), rejected
  because it breaks existing `--mode line` invocations.
- **[assumption]** Medians (not means) are the target values; rationale above.
- **[assumption]** The legendary/standalone split point in the single-member
  bucket is derived from the data and reported by the derivation step, rather
  than fixed in advance here.
- **[assumption]** No stage-count validation of the model's response. The
  existing 2-attempt retry budget is spent on the name contract, and a
  wrong-length line still normalizes and writes cleanly.
- **[open]** Whether `--stages` should also accept `1` as a synonym for
  `--mode single`. Assumed **no** — two ways to say one thing invites drift.
- **[note]** A stale `spec.md` (the shipped back-sprite palette lock, issue #1
  slice 4/4) sits in the repo root. Left untouched; worth deleting separately.
