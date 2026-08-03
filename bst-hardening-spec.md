# Spec: Harden BST targets against observed Gen 3 values; support 2-stage lines

> Tracking: #59

## Summary

`_BST_TARGETS` carries hand-picked numbers never checked against real Gen 3 data,
and `mode` is binary (`single` | 3-stage `line`) so 2-stage lines cannot be
generated. This work checks every BST target against observed game data and pins
them with a test-only aggregate fixture, then adds `--stages {2,3}`.

The bands have now been derived (see the Changelog). The outcome is mostly
**validation**: pseudo, legendary and mythical are all correct as they stand, and
only two values are actually wrong — `--mode single` (300 → 430, which was
prompting a juvenile's stat budget for a standalone species) and the 3-stage
mid-stage (420 → 405). A 2-stage row is added.

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

**Default preserved:** `--mode line` with no `--stages` still produces a 3-stage
line, so no existing invocation changes *shape*. Its prompt text does change, but
only in the corrected BST numbers — see Testing.

## Outputs

- No change to the shape of `generate_fakemon`'s return value: still a flat
  `list[dict]`, one element per stage, length 1, 2 or 3.
- No change to `stats.json` keys, `write_output`'s `stageN_<name>` directory
  naming, or the `.ini` export.
- The only observable change is the **BST hint text in the prompt**, and hence
  the stat spread the model produces.

## Behavior

### 1. Deriving the observed bands (offline, in the injector repo)

**This step is already done** — the bands below were derived and validated by
the spike on #59. This section records the method so the numbers can be
re-derived and verified, not because it is outstanding work.

The derivation script lives in **`fakemon-rom-injector`** (private), not here:
it depends on that package and on a local-only ROM, and it needs the
placeholder-slot IDs that must not enter this repo. It should be committed
there as a small utility rather than kept as a throwaway.

1. Load a BPEE ROM with the injector's `Rom` (`mapping.bpee_rom(bytes)`).
2. **Exclude the unused placeholder slots and the dummy species before any
   bucketing.** The species table contains 25 unused slots that all carry
   BST 600; left in, they masquerade as legendary-tier single-member lines.
   This step is mandatory — see the corruption table below.
   **Do not filter on "the species name fails to decode":** one genuine
   legendary also fails the strict decode and would be wrongly dropped. Filter
   on the documented unused ID range (which lives in the injector-side script).
3. `lines = mapping.derive_vanilla_lines(rom)`.
4. For each **linear** line, for each stage index `i`, record
   `mapping.bst(rom, species_id)` into a bucket keyed `(len(line), i)`.
   Non-linear lines are skipped entirely — they are the deferred branched case
   and would distort the linear bands.
5. Single-member lines form the standalone bucket, split at **BST 560** into
   ordinary no-evolution species and the legendary/mythical band. Any threshold
   in `(540, 580]` gives the identical partition.
   **Do not use a largest-gap heuristic to find this split.** The largest gap in
   the sorted single-member totals is 600 → 670 (70 points), which falls *inside*
   the legendary cluster. The real boundary is the second-largest gap, 540 → 580.
6. Emit `median`, `p10`, `p90` and sample count `n` per bucket.

#### Correctness assertion

After filtering, the `>= 560` band must contain **exactly 21 members**, clustered
9 / 6 / 2 / 4 across four distinct totals. That is the real count of Gen 3
legendaries and mythicals, and it is what proves the placeholder filter ran.
Without the filter the numbers are badly wrong:

| | unfiltered | filtered |
|---|---|---|
| single-member lines | 100 | 75 |
| standalone median | 502 | **430** |
| high-band members | 46 | **21** |

Ten derived values were cross-checked against
`https://www.serebii.net/pokedex-rs/stat/all.shtml`; all ten matched exactly.

**Only the aggregate output is committed here.** No species names, no IDs, no
ROM, no slot ranges. This is what keeps the `ac5cf2f` / `0ae9a1d` scrub intact.

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

The derived bands:

| bucket | n | p10 | median | p90 |
|---|---|---|---|---|
| 2-stage · stage 1 | 84 | 240 | **305** | 360 |
| 2-stage · stage 2 | 84 | 410 | **468** | 515 |
| 3-stage · stage 1 | 37 | 205 | **295** | 314 |
| 3-stage · stage 2 | 37 | 278 | **405** | 420 |
| 3-stage · stage 3 | 37 | 450 | **518** | 600 |
| standalone | 54 | 336 | **430** | 500 |
| legendary | 9 | — | **580** | — |
| mythical | 6 | — | **600** | — |

### 3. `_BST_TARGETS`: correct two values, add the 2-stage row

The derivation showed the existing table is largely already accurate — this is
mostly a **validation** exercise, not a rewrite. Only two numbers are wrong.

Structurally, the table is keyed `tier → {stage1, stage2, stage3}`, which cannot
express a 2-stage line. It becomes keyed **tier → stage-count → per-stage
targets**, with the standalone (single) value alongside.

| target | current | new | why |
|---|---|---|---|
| single (standard) | 300 | **430** | The real fix. 300 is the *stage-1* value; a single form is a standalone species. Consistent with #48's height/weight decision |
| 3-stage · stage 2 | 420 | **405** | 420 sits exactly at the band's p90 — the top edge, not the typical value |
| 3-stage · stage 1 | 300 | **295** | Band median; in-band already, changed only so every value follows one rule |
| 3-stage · stage 3 | 520 | **518** | Band median; as above |
| 2-stage | *(none)* | **305 / 468** | New row |
| pseudo | 300 / 420 / 600 | **unchanged** | Confirmed correct. The 600 final sits at p90 of the 3-stage final band — genuinely "rivals legendaries" |
| legendary | 580 | **unchanged** | Confirmed correct — the modal value, 9 species |
| mythical | 600 | **unchanged** | Confirmed correct — 6 species, and all four real mythicals are exactly 600 |

**Every value is its band median**, with no exceptions, so any number in the
table can be re-derived and verified without knowing which were hand-picked.
Medians rather than means: the 3-stage buckets have a long low tail (very weak
mid-stage forms) that drags the mean well off the typical value — stage 2's mean
is 371.5 against a median of 405, so a mean-based target would prompt a
mid-stage roughly 34 points weaker than it should be.

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
- Every `_BST_TARGETS` value equals its band's `median` exactly. Stricter than
  the band check and the actual rule the table follows; it catches a value
  hand-edited to something still inside the band.
- The fixture's legendary + mythical band contains **exactly 21 members**. This
  is the placeholder-filter check: a fixture regenerated without the exclusion
  step reads 46 here and fails, rather than silently shifting every band.
- Targets increase monotonically across the stages of a line.
- A 2-stage line's final target exceeds its own stage-1 target and is
  meaningfully above the 3-stage *mid*-stage target (a 2-stage final is a final).
- `--mode single --tier standard` target is the standalone value, and is
  strictly greater than the 3-stage stage-1 target (the #48 consistency fix).
- `--mode line` without `--stages` produces a prompt **identical in structure and
  wording** to today's, differing only in the corrected BST numbers. Assert the
  new prompt equals the old one with exactly those substitutions applied — a
  substring check is too weak to catch reordering or lost lines. A literally
  byte-identical prompt is impossible: the BST values are rendered into that
  string, and correcting them is the point of this issue.
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
- **[confirmed]** Medians (not means) are the target values — resolved by the
  spike. Evidence: 3-stage stage 2 has median 405 against mean 371.5, a 33.5-point
  gap caused by a long low tail. The two agree closely on the 2-stage buckets
  (305 / 305.5, 468 / 465.5), so the choice only matters for 3-stage lines.
- **[confirmed]** The legendary/standalone split point is **560** — resolved by
  the spike. Any threshold in `(540, 580]` gives the identical 54 / 21 partition.
- **[assumption]** No stage-count validation of the model's response. The
  existing 2-attempt retry budget is spent on the name contract, and a
  wrong-length line still normalizes and writes cleanly.
- **[open]** Whether `--stages` should also accept `1` as a synonym for
  `--mode single`. Assumed **no** — two ways to say one thing invites drift.
- **[note]** A stale `spec.md` (the shipped back-sprite palette lock, issue #1
  slice 4/4) sits in the repo root. Left untouched; worth deleting separately.

## Changelog

- 2026-08-03: **Bands derived; the placeholder-slot filter added as a mandatory
  derivation step.** The species table holds 25 unused slots all carrying BST
  600, which masquerade as legendary-tier single-member lines. Replaces: a
  derivation procedure that never mentioned them, and would have produced a
  standalone median of 502 (true value 430) and a legendary band of 46 members
  (true value 21). Adds "high band == 21" as the correctness assertion, and
  warns against a name-decode-based filter, which drops a real legendary.
- 2026-08-03: **Split point resolved to 560.** Replaces: "the exact split point
  is an implementation detail to be reported, not assumed here". Also records
  that a largest-gap heuristic picks the wrong boundary — the largest gap
  (600 → 670) sits inside the legendary cluster; the real split is the
  second-largest (540 → 580).
- 2026-08-03: **§3 reframed from a full re-derivation to "correct two values and
  add the 2-stage row", with real numbers filled in.** Replaces: the assumption
  that every tier's numbers needed re-deriving. Only single (300 → 430) and
  3-stage stage 2 (420 → 405) were actually wrong; pseudo, legendary and mythical
  are all confirmed correct as they stand. Stage 1 (300 → 295) and stage 3
  (520 → 518) move to their medians so the table follows one uniform rule
  — recommended default, no preference expressed.
- 2026-08-03: **Medians confirmed over means**, with the 33.5-point 3-stage
  stage-2 gap as evidence. Replaces: an unverified `[assumption]`.
- 2026-08-03: **"Byte-identical prompt" restated as "structure-identical, numbers
  may change"** — surfaced during task 00. Replaces: "`--mode line` without
  `--stages` produces a byte-identical prompt to today's", which was
  unsatisfiable. The BST values are rendered directly into that prompt string
  (`BST targets: stage 1 ~300, stage 2 ~420, stage 3 ~520.`), so correcting them
  necessarily changes it. The guarantee now pins structure and wording, asserted
  as old-prompt-with-substitutions rather than a substring check.
- 2026-08-03: **Derivation script relocated to the private injector repo.**
  Replaces: "a throwaway script, run once by the operator". It depends on that
  package and on the placeholder-slot IDs, which must not enter this repo, so it
  belongs there as a committed utility.
