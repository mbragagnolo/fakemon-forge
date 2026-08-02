# Spec: `category` schema field (Pokédex category noun)

## Summary

Add a `category` field to generated Fakemon stages — the Pokédex category
noun (e.g. `"SEED"`, `"MOUSE"`, `"TINY TURTLE"`) that describes what a
creature *is*, distinct from its type. Today no such field exists, so
downstream consumers (e.g. `export_ini.py`'s `PokedexType` line) fall back to
the bare primary-type word, and every species ends up categorised `"WATER"`
or `"FIRE"`. This slice adds the field to the LLM prompt, enforces its
contract in `generator._normalize`, and persists it (with an empty-string
default) in `writer.py`. Wiring `category` into any consumer (e.g.
`export_ini.py`) is out of scope — this slice only produces and persists the
field.

Done: `generate_fakemon(...)` returns stages carrying an uppercase `category`
of ≤ 11 valid Gen 3 chars (or the uppercased primary-type word when the LLM
omits/mangles it), `write_output(...)` persists it to `stats.json` with a
`""` default, and `pytest` passes with no new `ml`-marked tests.

## Inputs

- LLM JSON response per stage: an optional `category` key, expected to be a
  short string (e.g. `"Seed"`, `"tiny turtle"`), but may be absent, empty,
  non-string, over-length, or contain illegal characters — same failure
  modes as `name` today.
- `stage["types"]`: list of 1–2 type strings, already validated/normalized
  upstream of `_normalize`'s `category` handling (unchanged by this slice).

## Outputs

- Each stage dict returned by `generate_fakemon` / `_normalize` gains a
  `category` key: an uppercase string, ≤ 11 characters, drawn from the Gen 3
  charset, with no trailing `" POKEMON"`.
- `stats.json` written by `write_output` gains a `"category"` key: the
  stage's `category` if present, else `""`.

## Behavior

### `generator.py`

1. **System prompt.** Add a `category` line to `_SYSTEM_PROMPT`'s field list,
   directly under (or near) the existing `types` line, per the wording given
   in the issue:
   ```
   category – Pokédex category noun in caps, max 11 characters, e.g. "SEED",
     "MOUSE", "TINY TURTLE". Describes what the creature *is*, not its type —
     never "FIRE" or "WATER". No trailing "POKEMON".
   ```

2. **New constant.** `_MAX_CATEGORY_LEN = 11`, kept separate from
   `_MAX_NAME_LEN = 10` — same numeric family, different budget, not
   collapsed into one constant.

3. **Charset.** Reuse the existing `_ALLOWED_NAME_CHARS` set for `category`
   (the issue specifies the identical Gen 3 text set already used for
   names — no new charset constant needed).

4. **`_normalize` additions**, per stage, evaluated independently of the
   existing name-repair logic (order relative to the existing `name` /
   `abilities_gen3` lines in `_normalize` does not matter functionally):
   - Read `raw = stage.get("category")`.
   - **Fallback:** if `raw` is missing, `""` (or empty after cleaning), or
     not a `str`, set `stage["category"] = stage["types"][0].upper()`.
   - **Otherwise:**
     a. Strip illegal characters: keep only characters in
        `_ALLOWED_NAME_CHARS`.
     b. Strip a trailing `" POKEMON"` (case-insensitive) if present, so an
        LLM that ignores the instruction still produces a bare noun. Compare
        against the cleaned string from step (a).
     c. Uppercase the result.
     d. Truncate to `_MAX_CATEGORY_LEN` characters.
     e. If the result is empty after cleaning/truncation, fall back to
        `stage["types"][0].upper()` (empty-after-cleaning is equivalent to
        "no usable category supplied").
   - This is a pure, single-pass transform — no retry, no interaction with
     `_name_violations` / `_corrective_message` / the two-attempt API loop.
     `category` is fixed up unconditionally inside `_normalize`, which
     already runs only after the retry loop has produced a name-valid
     `stages` list.

   Suggested helper (naming only — implementation deferred):
   `_normalize_category(raw, types) -> str`, called from `_normalize` as
   `stage["category"] = _normalize_category(stage.get("category"), stage["types"])`.

5. **No change to the retry loop.** `_name_violations`, `_corrective_message`,
   `_repair_name`, and the `generate_fakemon` retry logic are untouched.
   `category` never triggers a second API call.

### `writer.py`

- Add `"category"` to `_STATS_DEFAULTS` with default `""`:
  ```python
  _STATS_DEFAULTS = {"levitates": False, "height_dm": 5, "weight_hg": 30,
                      "abilities_gen3": [], "category": ""}
  ```
  This follows the existing `levitates` precedent exactly: `_STATS_KEYS`
  already derives from `_STATS_DEFAULTS` via `*_STATS_DEFAULTS`, and
  `_write_stats` already does `stage.get(key, fallback)` for every key in
  `_STATS_DEFAULTS`, so no other code in `writer.py` changes.
- The writer does **not** compute the `types[0].upper()` fallback itself —
  that fallback already runs inside `_normalize`, so any stage dict that
  went through `generate_fakemon` already carries a non-empty `category` by
  the time it reaches the writer. The `""` default in `writer.py` exists
  only for hand-built/partial stage dicts (mirroring how `levitates`,
  `height_dm`, etc. are defaulted for the same reason), not as a second
  place implementing the fallback rule.

## Edge cases

- `category` absent entirely → `types[0].upper()`.
- `category == ""` → `types[0].upper()`.
- `category` is `None`, an int, a list, etc. → non-`str`, `types[0].upper()`.
- `category` is only illegal characters (e.g. all emoji) → cleans to `""` →
  `types[0].upper()`.
- `category` is exactly 11 characters → passes through unchanged (after
  case/charset normalization).
- `category` is 12+ characters → truncated to 11, no retry, no API call
  (e.g. `"TINY TURTLE"` → `"TINY TURTL"`).
- `category` given in lowercase or mixed case → uppercased.
- `category` given as `"Seed Pokemon"` → trailing `" POKEMON"` stripped
  (case-insensitive) → `"SEED"`.
- `category` given as `"Seedmon"` or any string that merely *contains*
  "pokemon" mid-word (not as a trailing `" POKEMON"` token) → left as-is
  aside from case/charset/length handling; only an exact trailing
  `" POKEMON"` suffix is stripped.
- `category` contains illegal characters mixed with legal ones (e.g. a
  newline or emoji embedded in an otherwise valid noun) → illegal chars
  stripped, remainder kept (mirrors `_repair_name`'s behavior for names).
- Truncation and trailing-`" POKEMON"`-stripping can interact: stripping
  happens before truncation, so a >11-char noun that happens to end in
  `" POKEMON"` is shortened by the strip first, then truncated only if
  still too long.
- `types` list is always non-empty by the time `_normalize` runs (existing
  invariant from the LLM contract / upstream validation) — `types[0]` is
  safe to index without a further empty-list guard, consistent with how
  `footprint.py` and `cries.py` already treat `types[0]` as available
  (they only guard against an empty list defensively; `_normalize` itself
  has no such guard for any existing field and this slice doesn't add one).
- `write_output` called with a hand-built stage dict lacking `category`
  entirely (e.g. in tests or the `levitates`-precedent-style manual dict) →
  `stats.json["category"] == ""`.

## Errors

No new error paths. `category` handling never raises, never calls `sys.exit`,
and never extends the API retry loop — malformed `category` values are
silently repaired in `_normalize`, same as the existing `abilities_gen3`
filtering (which also silently drops/repairs rather than erroring).

## Constraints & dependencies

- No new dependencies.
- `_MAX_CATEGORY_LEN` must stay a distinct constant from `_MAX_NAME_LEN`
  even though both are currently small integers — the issue explicitly
  calls out not collapsing them, since the two limits are allowed to
  diverge independently in the future.
- Must not touch `sprites.py`/`vision.py`/ML code paths — this is a pure
  text-schema change; no `@pytest.mark.ml` tests.
- `writer.py`'s `_STATS_KEYS`/`_STATS_DEFAULTS` mechanism is reused as-is;
  no structural change to `_write_stats`.

## Assumptions

- **Trailing-`" POKEMON"` stripping is case-insensitive and matches only an
  exact trailing `" POKEMON"` token** (i.e. `str.upper().removesuffix("
  POKEMON")` semantics after uppercasing), not a broader regex/substring
  removal. The issue's example (`"Seed Pokemon"` → `"SEED"`) supports this;
  more aggressive stripping (e.g. removing "POKEMON" anywhere in the string)
  is not specified and risks mangling legitimate nouns, so the conservative
  suffix-only interpretation is picked.
- **Order of operations for the non-fallback branch** is: strip illegal
  chars → strip trailing `" POKEMON"` → uppercase → truncate → re-check for
  emptiness → fallback if empty. The issue doesn't pin down ordering
  precisely; this order ensures the `" POKEMON"` check isn't defeated by
  stray illegal characters immediately preceding it, and ensures truncation
  is the last content-affecting step (matching `_repair_name`'s
  strip-then-truncate order for names, per its existing
  `test_last_resort_repair_strips_then_truncates` test).
- **The Gen 3 charset for `category` is identical to `_ALLOWED_NAME_CHARS`**
  and reuses that constant directly rather than introducing a duplicate
  `_ALLOWED_CATEGORY_CHARS`. The issue states the charset is "same table as
  names," so no new set literal is needed.
- **Empty-after-cleaning triggers the same fallback as missing/absent**,
  rather than persisting an empty string from `_normalize`. The issue's
  fallback rule is phrasing "absent, empty, or non-string" as the trigger
  conditions; treating "became empty after stripping illegal chars" as
  equivalent to "empty" keeps the invariant that `_normalize`'s output
  `category` is never blank, which existing code (and `writer.py`'s `""`
  default meaning "absent") relies on to distinguish "never went through
  `_normalize`" from "went through `_normalize`."
- **This slice does not modify `export_ini.py`** (or any other consumer) to
  read the new `category` field for `PokedexType`. The issue's file list
  scopes this slice to `generator.py` + `writer.py` (+ their tests) only;
  wiring consumers to prefer `category` over `types[0]` is left to a later
  slice, consistent with the issue being "slice 4/5."
- **No helper function name is prescribed** by the issue; `_normalize_category`
  is suggested as a natural counterpart to `_normalize_abilities_gen3` for
  implementation, but this is a naming suggestion, not a requirement — the
  implementer may inline the logic in `_normalize` instead.
- **Scope is appropriately sized for one slice** — it touches two source
  files with a small, self-contained normalization rule and mirrors an
  existing precedent (`levitates`) end to end, so no further scoping-down
  was needed.
