# Spec: Emit and persist height_dm / weight_hg with stage/tier defaults and clamping

## Summary

`generate_fakemon` (`fakemon_forge/generator.py`) asks the Mistral model for
a JSON array of stage dicts and post-processes them through `_normalize`
before returning. `_normalize` currently only enforces the Gen 3 name
contract (`fakemon_forge/generator.py:130`). `writer.py` then persists a
fixed subset of stage keys (`_STATS_KEYS`) to each stage's `stats.json`,
writing `levitates` defensively via `.get()` so hand-built/partial stage
dicts still work.

Gen 3 Pokédex data records height in decimetres and weight in hectograms;
`fakemon-forge` emits neither today, and `export_ini.py` hardcodes
`Hght=5`/`Wght=30`. This slice adds two new integer stage fields —
`height_dm` and `weight_hg` — sourced from the model, defaulted per
stage/tier when the model omits them, and clamped to safe ranges when
present but out of bounds. It does **not** touch `export_ini.py`; wiring the
hardcoded `Hght`/`Wght` lines to the new fields is left to a later slice.

"Done and correct" means: every stage dict returned by `generate_fakemon`
(and by `_normalize` directly) carries an integer `height_dm` in `[1, 999]`
and an integer `weight_hg` in `[1, 9999]`, regardless of what the model
returned or omitted; `writer.py` persists both into `stats.json`, defaulting
to `5`/`30` when a stage dict lacks the keys entirely; and the system prompt
documents both fields with calibration anchors so the model's own values (the
common case) are reasonable before any clamping applies.

## Inputs

### `generate_fakemon(description, mode, tier="standard", *, client=None, api_key=None)`

Signature is unchanged. The LLM response for each stage dict may now
optionally include:

- `height_dm` — int, height in decimetres, model's best guess.
- `weight_hg` — int, weight in hectograms, model's best guess.

Both are optional from the model's perspective (older/malformed responses
omitting them must not crash `_normalize`); `mode` (`"single"`/`"line"`) and
`tier` (`"standard"`/`"pseudo"`/`"legendary"`/`"mythical"`) — both already
parameters of `generate_fakemon` and already threaded into `_normalize(stages,
mode, tier)` — select the default when a field is missing.

### `_normalize(stages, mode, tier)`

Unchanged signature. Behavior extended per stage (see Behavior).

### `writer._write_stats(stage, stage_dir)` / `write_output(stages, base_dir="output")`

Unchanged signatures. `stage` dicts passed in may or may not carry
`height_dm`/`weight_hg` (e.g. hand-built dicts in tests, or future callers
that skip `_normalize`).

## Outputs

- Every stage dict returned from `generate_fakemon` (and from `_normalize`
  called directly) has `stage["height_dm"]` as an `int` in `[1, 999]` and
  `stage["weight_hg"]` as an `int` in `[1, 9999]`.
- `stats.json` written by `writer.write_output` includes `"height_dm"` and
  `"weight_hg"` integer keys alongside the existing persisted fields.
- `_SYSTEM_PROMPT` contains the `height_dm` / `weight_hg` field
  documentation with the scale anchors from the issue, so prompt-content
  tests (`_SYSTEM_PROMPT` string, `messages` sent to the client) can assert
  on it the same way existing tests assert on e.g. `levitates`.

## Behavior

### `_normalize` — defaulting

For each stage, if `"height_dm"` is absent (key not in the dict — not merely
falsy, since `0` must never be emitted, see Edge cases) or `"weight_hg"` is
absent, fill it in using this table, keyed by `mode` and, for `mode="single"`,
`tier`, and for `mode="line"`, `stage["stage"]`:

| Case | height_dm | weight_hg |
|---|---|---|
| `mode="line"`, `stage["stage"] == 1` | 5 | 30 |
| `mode="line"`, `stage["stage"] == 2` | 10 | 150 |
| `mode="line"`, `stage["stage"] == 3` | 17 | 600 |
| `mode="single"`, `tier == "standard"` | 10 | 150 |
| `mode="single"`, `tier in {"pseudo", "legendary", "mythical"}` | 17 | 600 |

This mirrors the existing per-stage loop in `_normalize` — no new looping
construct, just two more assignments (guarded by presence-check) alongside
the existing `stage["name"] = _repair_name(stage["name"])` line.

### `_normalize` — clamping

Independent of defaulting, once each stage has a `height_dm`/`weight_hg`
value (whether model-supplied or just defaulted), clamp:

- `height_dm` to `[1, 999]`
- `weight_hg` to `[1, 9999]`

using standard min/max clamping (`max(1, min(999, value))` and
`max(1, min(9999, value))`). Values already in range pass through unchanged.
Defaulted values (5/10/17 and 30/150/600) are all inside both ranges, so
clamping never alters a just-applied default — the two steps compose
without special-casing.

### `_SYSTEM_PROMPT` — field documentation

Add to the per-stage field list in `_SYSTEM_PROMPT`
(`fakemon_forge/generator.py:25`), following the existing `field – description`
style used for `name`, `stage`, `types`, etc.:

```
height_dm – height in decimetres (integer).
weight_hg – weight in hectograms (integer).
  For scale: a small rodent is ~3 dm / 35 hg, a mid-size quadruped
  ~10 dm / 300 hg, a large final-stage dragon ~20 dm / 2100 hg.
  Values must grow across an evolutionary line.
```

This is prompt guidance only — it does not change what `_normalize` accepts
or how it defaults/clamps; the model is free to (and expected to) return
values the anchors merely calibrate.

### `writer.py` — persistence

Add `"height_dm"` and `"weight_hg"` to `_STATS_KEYS`
(`fakemon_forge/writer.py:4`). Extend `_write_stats` following the existing
`levitates` precedent — excluded from the generic dict comprehension (since
that comprehension does a required `stage[k]` lookup that would `KeyError`
on an absent key) and set individually via `.get()` with a flat default:

```python
data["height_dm"] = stage.get("height_dm", 5)
data["weight_hg"] = stage.get("weight_hg", 30)
```

The writer's defaults are flat (`5`/`30`), not stage/tier-scaled — see
Assumptions.

## Edge cases

- **Model omits both fields**: `_normalize` fills both from the table; result
  is already in-range, so clamping is a no-op.
- **Model returns `height_dm: 0`**: `0` is present (not absent), so no
  default applies; clamping raises it to `1`. Same for `weight_hg: 0`. This
  is the mechanism that guarantees the `[1, ...]` floor is never violated by
  a model-supplied zero — per the issue, downstream treats non-positive as a
  hard error, so defaulting-on-falsy (which would also catch `0`) is
  explicitly *not* used; presence, not truthiness, is what's checked.
- **Model returns a huge value** (e.g. `height_dm: 50000`): present, so no
  default; clamped down to `999` (`weight_hg` to `9999`).
- **Model returns a negative value**: present; clamped up to `1` by the same
  `max(1, ...)` floor that handles `0`.
- **`mode="line"` but a stage dict's `"stage"` key is missing or outside
  `{1, 2, 3}`**: out of scope — `_normalize` (and the rest of the pipeline)
  already assumes `stage["stage"] ∈ {1, 2, 3}` for `mode="line"` and has no
  existing fallback for a malformed `stage` number; this slice does not add
  one for height/weight defaulting either.
- **`_normalize` called directly (as existing tests do) with hand-built
  stage dicts lacking `height_dm`/`weight_hg`**: defaults apply exactly as
  in the full `generate_fakemon` path, since `_normalize` is the sole place
  the table lives.
- **Writer called with a stage dict that has no `height_dm`/`weight_hg` at
  all** (e.g. a hand-built dict in a writer test, bypassing `_normalize`):
  `stats.json` gets the flat `5`/`30` defaults, matching `levitates`' existing
  `False` fallback pattern.
- **Writer called with a stage dict that already has valid `height_dm`/
  `weight_hg`** (the normal `generate_fakemon` → `write_output` path):
  those exact values are persisted; the writer does not re-clamp (clamping
  is `_normalize`'s responsibility only, so a stage dict that skipped
  `_normalize` and carries an out-of-range value would be persisted as-is —
  consistent with the writer trusting its input the way it already does for
  every other field).

## Errors

No new error paths. `_normalize` still cannot fail the way the JSON-parse /
name-violation retry loop can — it remains pure post-processing with no API
calls (`test_normalize_makes_no_api_call` already pins this down and
continues to hold, since defaulting/clamping touch no client). A stage dict
missing required *existing* fields (e.g. `name`) is already unhandled
upstream of this change and stays that way.

## Constraints & dependencies

- No new dependencies. Pure Python arithmetic in `generator.py`; `writer.py`
  change is the same `.get()`-with-default shape already used for
  `levitates`.
- Downstream (a future slice, not this one) treats `height_dm`/`weight_hg` as
  2-byte unsigned values written into `.ini` `Hght`/`Wght` fields and as
  hard-error on non-positive — this spec's ranges (`[1, 999]`, `[1, 9999]`)
  are sized to that constraint but the `.ini` write itself is out of scope
  here.
- `export_ini.py`'s hardcoded `Hght=5` / `Wght=30` lines are intentionally
  untouched by this slice (per the issue's file list) — reading the new
  `stats.json` keys into the `.ini` export is left to a later slice in #48.

## Assumptions

- **[picked]** Writer's absent-key defaults are flat `5`/`30`, not
  stage/tier-scaled, per the issue's explicit instruction — the writer has
  no `tier`/`mode` parameter and the scaled table is generator-only.
- **[picked]** Defaulting is presence-based (`"height_dm" not in stage`), not
  truthiness-based (`not stage.get("height_dm")`), so that a model-supplied
  `0` is clamped to `1` rather than silently replaced by the stage/tier
  default. This is the only reading consistent with the issue's "clamping
  (not defaulting) is what guarantees the `[1, ...]` floor" statement.
- **[picked]** Clamping applies uniformly to every stage regardless of
  whether the value came from the model or from the default table; this is
  simpler than special-casing "just-defaulted" values and is behaviorally
  identical since all default values already lie inside both ranges.
- **[picked]** The `mode="line"` default table is keyed off each stage
  dict's own `stage["stage"]` field (already guaranteed present and
  validated as one of the required fields by existing tests/behavior), not
  off list position — these coincide in practice but keying off the
  explicit field is more robust and matches how `writer.py` already keys
  stage directory names off `stage["stage"]`.
- **[picked]** No new CLI surface, no change to `export_ini.py`, no change
  to `main.py` — this slice is generator + writer only, matching the issue's
  file list exactly.
- **[picked]** `_SYSTEM_PROMPT` field documentation is added as a literal
  block appended after the existing `levitates` line, preserving the
  field's declared order in the docstring-style listing; exact prose is
  taken verbatim from the issue since it was already reviewed/approved
  there.
- Scope is limited to this one slice (2/5 of #48) as instructed; broader
  concerns (`.ini` export wiring, CLI flags to override height/weight,
  UI/tests around `main.py`) are explicitly deferred to later slices and not
  addressed here.
