# Spec: `abilities_gen3` schema field

## Summary

Add a new `abilities_gen3` field to generated Fakemon stage data: a list of
1–2 real Gen 3 ability names, distinct from the existing free-text `ability`
flavor field. The pool of valid names is derived at runtime from
`resources/gen3_abilities.json` (excluding index `0` "None" and index `76`
"Cacophony"), embedded in the LLM system prompt, and enforced by a
post-parse validation/canonicalization step in `generator._normalize`. The
field is persisted to `stats.json` by `writer.py` using the same
defensive-default pattern already used for `levitates`.

This closes the gap where ~60% of free-text `ability` values aren't real Gen
3 abilities and silently resolve to ability index `0` in the exported
`.ini`. `abilities_gen3` gives `export_ini.py` (a later slice) a
machine-trustworthy source to resolve instead.

## Inputs

- `resources/gen3_abilities.json` — existing file, `{index_str: name}`,
  currently 78 entries (`"0"`–`"77"`). Read at runtime by `generator.py`,
  the same way `export_ini._resolve_ability` already reads it. Never a
  hardcoded count or copied list — the count has drifted before and must
  keep working if it drifts again.
- LLM JSON response per stage: a new `abilities_gen3` key, expected to be a
  list of 1–2 strings (model-authored spelling/casing, may be invalid, may
  be duplicated, may exceed 2 entries — `_normalize` must not assume the
  model followed instructions).
- Existing `_normalize(stages, mode, tier)` inputs are unchanged in shape.

## Outputs

- Every stage dict returned by `generate_fakemon` (via `_normalize`) carries
  `abilities_gen3`: a list of **0–2** strings, each an exact canonical
  spelling taken from `resources/gen3_abilities.json` values (e.g.
  `"Compound Eyes"`), never the model's raw casing/spacing, never `"None"`
  or `"Cacophony"`.
- `stats.json` (via `writer.py`) gets an `abilities_gen3` key: the stage's
  list if present, else `[]`.
- The free-text `ability` field and its persistence are unchanged.

## Behavior

### Pool construction (`generator.py`)

- At module load (mirroring how `_SYSTEM_PROMPT` is already a module-level
  constant string built once), read `resources/gen3_abilities.json`
  relative to the repo (same `Path(__file__).parent / "resources"`
  resolution style as `export_ini._RESOURCES`), and build the usable pool:
  all values except the one keyed `"0"` and the one keyed `"76"` — exclude
  by index key so the pool tracks the file regardless of whether the
  *names* at those indexes ever change.
- Build a lookup from normalized name → canonical name, using the
  normalization `"".join(name.split()).lower()`, for use in `_normalize`.
- Embed the pool (canonical names) in `_SYSTEM_PROMPT`, listing it as the
  closed set the model must choose from for `abilities_gen3`, and document
  the field's contract inline:
  - `abilities_gen3` — list of 1 or 2 **distinct** ability names, chosen
    only from the provided Gen 3 ability list.
  - Guidance: prefer two abilities (authentic Gen 3 species are roughly
    half dual-ability, and variety is preferred); one is acceptable for a
    single signature ability.
  - Guidance (line mode only, not validated): all stages in a line should
    share the same `abilities_gen3`; the final stage may add one more.
  - Guidance: the free-text `ability` should express the same concept as
    the chosen `abilities_gen3` entries (flavor/mechanic tie), though
    `ability` itself stays free text.

### Validation (`_normalize`)

For each stage, independently (no cross-stage checks):

1. Read `stage.get("abilities_gen3", [])`; treat a missing/absent key the
   same as an empty list (no `KeyError`).
2. For each entry, normalize with `"".join(name.split()).lower()` and look
   it up in the pool built above.
   - No match (including anything normalizing to `"none"` or
     `"cacophony"`, since those aren't in the pool) → drop the entry.
   - Match → replace it with the pool's canonical spelling.
3. Collapse duplicates by comparing normalized form (so
   `"Blaze"`/`"blaze"`/`"BLAZE"` collapse to one entry), keeping first
   occurrence order.
4. Cap the result at 2 entries — dedup first, then cap (see Assumptions for
   why this ordering matters).
5. Assign the resulting list (possibly empty) back to
   `stage["abilities_gen3"]`.

This mirrors the existing shape of `_normalize`: a per-stage loop that
mutates and returns `stages`, no exceptions raised, no API calls.

### Persistence (`writer.py`)

- Add `"abilities_gen3": []` to `_STATS_DEFAULTS` (the same dict that
  already holds `levitates`/`height_dm`/`weight_hg`), which — via the
  existing `_write_stats` loop `data[key] = stage.get(key, fallback)` —
  persists the stage's `abilities_gen3` if present, else `[]`.
  `_STATS_KEYS` already spreads `*_STATS_DEFAULTS`, so no separate change
  is needed there.
- No changes to `_write_entry`, directory logic, or any other writer
  behavior.

## Edge cases

- Model omits `abilities_gen3` entirely → `[]` after `_normalize`; writer
  persists `[]`.
- Model returns an empty list → stays `[]`.
- Model returns entries with mixed valid/invalid: `["Blaze", "Solar
  Power"]` (Solar Power is Gen 4) → `["Blaze"]`.
- Model returns only invalid/invented names: `["Molten Core", "Ashwalk"]`
  → `[]`.
- Model returns `"None"` or `"Cacophony"` (literal table entries but
  outside the usable pool) → dropped, since the pool excludes indexes `0`
  and `76` by construction.
- Model returns the same ability twice with different casing/spacing:
  `["Compound Eyes", "compoundeyes"]` → `["Compound Eyes"]` (single entry,
  canonical spelling).
- Model returns 3+ valid distinct abilities → capped to the first 2, after
  dedup.
- Model returns non-list, non-string, or otherwise malformed entries for
  `abilities_gen3` — out of scope for this slice; only list-of-strings
  input is handled, matching how `_normalize` already trusts the parsed
  JSON shape for other fields (e.g. `base_stats`, `types`), with the
  2-attempt JSON-parse retry in `generate_fakemon` as the only existing
  guard against malformed LLM output.
- `resources/gen3_abilities.json` pool size or contents change in the
  future → pool and prompt text adapt automatically since both are derived
  from the file at runtime, not hardcoded.

## Errors

- No new error paths. `resources/gen3_abilities.json` is a committed repo
  resource read the same way `export_ini.py` already reads it (no
  try/except there today); this slice follows the same assumption that the
  file exists and is valid JSON.
- `_normalize` does not raise on missing/malformed `abilities_gen3` input —
  it degrades to `[]`, consistent with "an empty list is treated identically
  to an absent field by consumers."

## Constraints & dependencies

- Must not hardcode the ability count or a copied ability list anywhere in
  `generator.py` — always derive from `resources/gen3_abilities.json` at
  runtime.
- The normalization function `"".join(name.split()).lower()` must match
  exactly what downstream code (a later slice, per the issue) will use —
  this slice only needs internal consistency between the prompt-pool
  lookup and `_normalize`, but the exact formula is specified and must not
  be approximated (e.g. `.replace(" ", "")` alone would differ on
  tabs/newlines/repeated spaces).
- No new third-party dependencies; `json`/`pathlib` only, matching
  `export_ini.py`'s existing pattern.
- No ML code involved; nothing in this slice touches `sprites.py` or
  `@pytest.mark.ml`.
- Test suite must pass fully mocked, no real Mistral API calls, following
  existing `tests/test_generator.py` conventions (mock `client`, call
  `_normalize` directly for normalization-only tests).

## Assumptions

- **Pool exclusion is by index key, not by matching the string values
  `"None"`/`"Cacophony"`.** The issue describes the outcome ("excluding
  index 0 and index 76") but not the filter mechanism; filtering by key
  (`"0"`, `"76"`) is more robust than filtering by name, since it keeps
  working even if those entries' text ever changes, and is the default
  picked here.
- **`_SYSTEM_PROMPT` stays a module-level string constant**, computed once
  at import time by reading the JSON file, rather than becoming a
  function. This matches the existing test pattern
  (`from fakemon_forge.generator import ... _SYSTEM_PROMPT`), which
  imports it as a plain string, and still satisfies "read at runtime" (as
  opposed to a value hardcoded in source) since it's computed from the
  file rather than copy-pasted.
- **Writer implementation goes through `_STATS_DEFAULTS`**, not a literal
  `data["abilities_gen3"] = stage.get("abilities_gen3", [])` line as shown
  in the issue text — the existing `_write_stats` loop already generalizes
  that exact pattern for `levitates`/`height_dm`/`weight_hg`, and reusing
  it avoids a one-off line duplicating existing logic. Net behavior is
  identical.
- **No cross-stage validation or repair**, confirmed explicit in the
  issue's own "Assumptions picked" section: the line convention (shared
  abilities across stages, final-stage bonus) is prompt guidance only.
- **Malformed non-list `abilities_gen3` from the LLM (e.g. a string, a
  dict) is out of scope.** The issue's validation rules (drop invalid,
  cap, dedup) presume list-of-strings input, matching how other
  `_normalize` fields (e.g. `base_stats`) already trust the LLM's JSON
  shape without defensive type-checking. If this needs hardening, it's a
  separate concern from ability validity.
- **Dedup happens before cap, not after.** Cap-then-dedup could yield
  fewer than 2 valid distinct entries when duplicates appear early (e.g.
  `["Blaze", "blaze", "Flash Fire"]` → dedup-then-cap keeps `["Blaze",
  "Flash Fire"]`; cap-then-dedup would incorrectly yield `["Blaze"]`). Not
  stated explicitly in the issue; picked as the more sensible reading of
  "collapse duplicates... cap at 2."
