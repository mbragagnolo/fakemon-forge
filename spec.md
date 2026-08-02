# Spec: Surface height/weight, ability2, and category in the .ini export

## Summary

`fakemon_forge/export_ini.py` reads `stats.json` + `entry.md` from a stage
directory and writes a Gen 3-style `.ini`. Today it hardcodes `Hght=5`,
`Wght=30`, and a zero `ability2` byte, and derives `PokedexType` from the
primary type word plus a literal `" POKEMON"` suffix. `stats.json` may
(independently of this slice) start carrying four new optional keys —
`height_dm`, `weight_hg`, `abilities_gen3`, `category` — written defensively
by `writer.py` and defaulted when absent. This slice makes `export_ini` read
those four keys when present and fall back to today's exact behavior when
they are not, except for one intentional, called-out change: the
`PokedexType` field drops its `" POKEMON"` suffix unconditionally, for both
new-format and legacy directories.

"Done and correct" means: `export_ini()` produces a valid `.ini` for both
new-format and legacy `stats.json` files, the four new fields are correctly
encoded when present, legacy `Hght`/`Wght`/ability2 output is byte-for-byte
unchanged, `PokedexType` never carries the `" POKEMON"` suffix going forward,
and `tests/test_export_ini.py` (new) plus the full `pytest` suite pass with
no ML involvement.

## Inputs

- `stage_dir: Path` — a stage directory containing `stats.json` and
  `entry.md`, same as today's `export_ini(stage_dir)` signature (unchanged).
- `stats.json` fields relevant to this slice, all optional/defensive:
  - `height_dm: int` — height in decimetres (Gen 3 native unit for `Hght`).
  - `weight_hg: int` — weight in hectograms (Gen 3 native unit for `Wght`).
  - `abilities_gen3: list[str]` — 0, 1, or 2 real Gen 3 ability names, e.g.
    `["Blaze", "Solar Power"]`.
  - `category: str` — a Pokédex category label, e.g. `"Flame"` (the word
    that would traditionally precede "POKEMON" in Gen 3 dex screens, but
    stored and emitted here without that suffix — see Behavior).
- Existing fields this slice continues to depend on: `data["ability"]`
  (free-text or real ability name, legacy path), `data["types"]`,
  `data["base_stats"]`, `data["name"]`.
- `resources/gen3_abilities.json` — unchanged, 78 entries, `"index": "Name"`,
  used via `_resolve_ability` for both `data["ability"]` and, newly, each
  name in `abilities_gen3`.

## Outputs

- Same as today: `export_ini` writes `{stage_dir}/{data['name']}.ini` and
  returns its `Path`.
- Changed `.ini` lines:
  - `Hght={height_dm}` when `stats.json` has `height_dm`; else `Hght=5`
    (today's literal).
  - `Wght={weight_hg}` when `stats.json` has `weight_hg`; else `Wght=30`
    (today's literal).
  - `PokedexType={category}` when `category` is present, non-empty, and a
    string; else `PokedexType={data['types'][0].upper()}` — in both branches,
    **no trailing `" POKEMON"`** (removed unconditionally, including for
    legacy directories with no `category` key — see Behavior for why this is
    the one intentional non-byte-identical change).
- `BaseStats` hex blob: byte offset 22 (`ability1`) unchanged in meaning;
  byte offset 23 (`ability2`, previously always `0x00`) now carries a real
  index when `abilities_gen3` supplies a second name. Blob length (28 bytes /
  56 hex chars) is unchanged.

## Behavior

1. **Height/weight.**
   - Read `data.get("height_dm")` and `data.get("weight_hg")` with a
     presence check (`"height_dm" in data`, not truthiness — see Edge cases
     for why `0` must round-trip).
   - If present, emit them verbatim as `Hght=`/`Wght=` (they are already
     clamped to valid 2-byte-field ranges upstream at generation time — this
     module does not re-validate or re-clamp).
   - If absent (legacy `stats.json`, or any writer that omits them), emit
     the flat literals `Hght=5` / `Wght=30`, matching today's output
     exactly. This is a per-field fallback: `height_dm` and `weight_hg` are
     considered independently, so a file with one but not the other emits
     the real value for the one present and the `5`/`30` literal for the
     one absent.

2. **Ability bytes.**
   - `_encode_base_stats(data, ability1_idx, ability2_idx)` gains a second
     required parameter and packs both into bytes 22–23 (previously
     `ability_idx & 0xFF, 0x00`), as `ability1_idx & 0xFF, ability2_idx &
     0xFF`.
   - Index resolution, done by the caller (`export_ini`) before building the
     stats blob:
     - `abilities_gen3` present and non-empty (a list with ≥1 entry):
       - `ability1_idx = _resolve_ability(abilities_gen3[0])`
       - `ability2_idx = _resolve_ability(abilities_gen3[1])` if
         `len(abilities_gen3) >= 2`, else `0x00`.
     - `abilities_gen3` absent, empty, or not a non-empty list — treat as
       not present:
       - `ability1_idx = _resolve_ability(data.get("ability", ""))` (today's
         call, unchanged).
       - `ability2_idx = 0x00`.
   - `_resolve_ability` itself is unchanged and is reused for both the
     legacy free-text `ability` lookup and each `abilities_gen3` entry: exact
     case-insensitive name match against `resources/gen3_abilities.json`
     values, else `_ABILITY_FALLBACK.get(lower, 0)`.
   - `_ABILITY_MOVES` (movepool injection) stays keyed on `data["ability"]`
     only. It is **not** consulted for `abilities_gen3` entries and is not
     extended to cover real Gen 3 ability names — out of scope per the task.
   - `_ABILITY_FALLBACK` / the custom-name path through `_resolve_ability`
     remains the only route for legacy/custom ability strings (including
     `data["ability"]` and, incidentally, any `abilities_gen3` entry that
     happens not to match a canonical name — same fallback-to-0 behavior as
     today, no new error path).

3. **PokedexType from category.**
   - `category = data.get("category")`.
   - If `category` is a non-empty `str`, emit `PokedexType={category}`
     (used as-is; not upper-cased — it is assumed to already be in display
     form, e.g. `"Flame"`, since it is free text from the generator/writer
     rather than a fixed type-name lookup).
   - Otherwise (`category` absent, `None`, empty string, or non-`str`),
     fall back to `PokedexType={data['types'][0].upper()}` — the existing
     type-word derivation, just without the suffix.
   - The literal `" POKEMON"` suffix is removed from both branches. This is
     a deliberate behavior change to every existing `.ini` output,
     independent of whether `stats.json` is new- or legacy-format: the
     suffix duplicated a label the editor tooling renders separately and
     pushed the field past an 11-character budget. Flagged here so the PR
     description carries it forward as the one place legacy output is not
     byte-identical to today.

## Edge cases

- `stats.json` has none of the four new keys (fully legacy): `Hght=5`,
  `Wght=30`, ability2 byte `0x00`, `PokedexType` = upper-cased primary type
  with no suffix. This is the only diff from current output (the suffix
  removal).
- `abilities_gen3` present but `[]` (empty list): treated identically to
  absent — falls back to `data["ability"]`.
- `abilities_gen3` has exactly 1 entry: `ability1` from that entry,
  `ability2 = 0x00`.
- `abilities_gen3` has an entry with no match in `gen3_abilities.json` and no
  `_ABILITY_FALLBACK` entry: resolves to index `0`, same as any unresolved
  ability today — no exception raised.
- `category` present but empty string `""`, or present but non-`str` (e.g.
  a stray number/list from a malformed file): falls back to the type-word
  derivation, per "non-empty and a string" gate in the task description.
- `data["ability"]` missing entirely and `abilities_gen3` also
  absent/empty: `_resolve_ability("")` is called exactly as it would be
  today (`data.get("ability", "")` already guards this) — resolves via
  fallback dict miss to `0`.
- `height_dm` or `weight_hg` present as `0`: `0` is a valid falsy-but-present
  int; must be distinguished from "absent" via a presence check (`"height_dm"
  in data`), not truthiness, so `Hght=0` is emitted rather than silently
  falling back to `5`.

## Errors

- No new error paths are introduced. Malformed/missing keys degrade to the
  documented fallbacks rather than raising.
- Existing error behavior is unchanged: `export_ini` still lets
  `FileNotFoundError` propagate if `stats.json`/`entry.md` are missing, and
  `KeyError` propagate if required legacy keys (`name`, `types`,
  `base_stats`) are missing — this slice does not add defensiveness around
  those, only around the four new optional keys.

## Constraints & dependencies

- No new third-party dependencies; stays within `json`, `hashlib`, `textwrap`,
  `pathlib`, matching the existing module.
- `_encode_base_stats`'s signature change (`ability_idx` → two params) is an
  internal, non-public-API change within this module; its one call site
  (`export_ini`) is updated in the same change.
- Byte layout of the 28-byte `BaseStats` blob is otherwise unchanged — same
  length, same field order, only the value written at offset 23 changes from
  a hardcoded `0x00` to a computed index.
- Must not touch `sprites.py`, `generator.py`, or `writer.py` — this slice is
  explicitly independent of the generator/writer changes (per task) and
  reads whatever `stats.json` already contains. `writer.py`'s current
  `_STATS_KEYS` set does not include the four new fields; this spec does not
  change that, since writer changes are out of scope here.
- No ML code, no `@pytest.mark.ml`, no `import torch` — this module has none
  today and none is introduced.
- New test file `tests/test_export_ini.py` follows the existing test
  conventions in this repo (see `tests/test_writer.py`): plain `tmp_path`
  fixtures (write `stats.json`/`entry.md` directly, since there's no writer
  output carrying the new fields yet), no mocking framework needed since
  `export_ini` only does file I/O and pure computation. Minimum coverage per
  the task:
  - New-format round-trip: all four fields present, `.ini` reflects them.
  - Legacy fallback: none of the four present → `Hght=5`, `Wght=30`,
    ability2 byte `0x00`, and `PokedexType` with no `" POKEMON"` suffix.
  - Ability byte positions: assert `ability1`/`ability2` land at the correct
    offsets within the `BaseStats` hex string (byte offsets 22/23, i.e. hex
    chars `[44:46]`/`[46:48]` of the 56-char blob).
  - `PokedexType` from `category` and from the type-word fallback, both
    asserting the absence of `" POKEMON"`.

## Assumptions

- **`category` is emitted verbatim, not upper-cased.** The task doesn't
  specify casing for the `category` branch; since it's free text (unlike the
  fixed `data['types'][0]` lookup, which the task explicitly says to
  upper-case), assumed to already be in the caller's intended display form
  and passed through unchanged. *(Assumption — not confirmed by existing
  code/tests, since no prior code path touches `category`.)* If wrong, it's
  a one-line fix isolated to the `category` branch.
- **Presence check for `height_dm`/`weight_hg` uses key presence
  (`"key" in data`), not truthiness.** Chosen so `height_dm: 0` (a
  legitimate, if unusual, value) is distinguishable from "key absent."
  *(Assumption, since the task doesn't explicitly address a zero-value edge
  case; follows the same "present" framing the task uses for `category`.)*
- **`abilities_gen3` entries are matched via the exact same
  `_resolve_ability` function used for `data["ability"]`**, rather than a
  separate stricter lookup that requires an exact canonical match. The task
  says "reuse `_resolve_ability` or an index map" — reusing it directly is
  simpler and keeps one code path, at the cost of allowing
  `_ABILITY_FALLBACK` custom-name hits (e.g. `"steam engine"`) to leak into
  the `abilities_gen3` path even though that field is documented as "real
  Gen 3 ability names." Treated as harmless since it only ever produces a
  best-effort index, never an error. *(Assumption/design choice — task
  allows either approach explicitly.)*
- **No re-clamping or range validation of `height_dm`/`weight_hg`** in this
  module, per the task's explicit statement that "values are already
  clamped at generation time." Out-of-range values (if a future writer bug
  produced one) would be written as-is into a 2-byte field with no
  truncation guard here. *(Confirmed by task text, not an independent
  assumption — noted for completeness.)*
- **Scope stays exactly as sliced.** This is a small, single-file
  (plus new test file) change; no further scoping-down was needed — the
  task is already an appropriately-sized slice.
