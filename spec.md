# Spec: `levitates` flag in generator schema and `stats.json`

## Summary
Introduce a new stage-level semantic boolean, `levitates`, that the Mistral
generator is asked to produce and that the writer persists into each stage's
`stats.json`. `levitates` is `true` only when the creature never touches the
ground (levitating, bodiless, gaseous, amorphous — floating orbs, ghosts,
cloud/gas creatures); otherwise `false`.

This slice only *produces and persists* the flag. Nothing consumes it yet — a
downstream footprint-generation feature (later slice of #20) will read it. The
field is optional output data: a model that omits it must never cause a crash,
and every consumer defaults it to `False` via `stage.get("levitates", False)`.

"Done and correct" means:
- `fakemon_forge/generator.py`'s `_SYSTEM_PROMPT` documents `levitates` as an
  expected boolean field with the "never touches the ground" definition, and
  the existing "Return ONLY the JSON array… no extra keys" instruction stays
  consistent (i.e. `levitates` is now one of the allowed/expected keys).
- `fakemon_forge/writer.py` persists `levitates` into `stats.json`: a stage that
  provides it writes that value; a stage that omits it writes `False` with no
  exception raised.
- The other required `_STATS_KEYS` remain strictly required (still raise
  `KeyError` if absent).
- `pytest` passes from the repo root (the ~21 `ml`-marked tests skip in the
  slim sandbox — expected per `CLAUDE.md`; this slice adds no `ml` tests).

## Inputs
- **Generator side:** unchanged public signature of
  `generate_fakemon(description, mode, tier="standard", *, client=None,
  api_key=None)`. The only change is the text of `_SYSTEM_PROMPT`. The LLM
  response (a JSON array of stage dicts) may now include a `levitates` boolean
  per stage; it may also omit it.
- **Writer side:** `write_output(stages, base_dir="output")` receives a list of
  stage dicts. Each stage dict may or may not contain a `levitates` key. When
  present, `levitates` is expected to be a JSON boolean (`true`/`false`).

## Outputs
- **`_SYSTEM_PROMPT`** gains one documented field line for `levitates` in the
  "must have exactly these fields" list, with the never-touches-the-ground
  definition.
- **`stats.json`** for every written stage gains a `levitates` key alongside the
  existing whitelisted keys (`name`, `stage`, `types`, `ability`,
  `base_stats`). Its value is the stage's provided boolean, or `False` when the
  stage dict omits the key.
- `entry.md` output is unchanged.
- No new files, directories, or CLI flags.

## Behavior
### `fakemon_forge/generator.py`
- Add exactly one field to the schema block inside `_SYSTEM_PROMPT`, matching
  the existing `name          – …` / `stage         – …` formatting convention
  (aligned en-dash descriptions), e.g. a line of the form:
  `levitates     – boolean; true only if the creature levitates, is
  bodiless/gaseous/amorphous, or otherwise never touches the ground (e.g.
  floating orbs, ghosts, cloud/gas creatures); otherwise false`.
- Do **not** change parsing, retry, fence-stripping, `_user_prompt`,
  `_BST_TARGETS`, `_EVO_PROGRESSION`, or `_TIER_NOTES`. The "Return ONLY the
  JSON array. No markdown fences, no explanation, no extra keys." trailer stays
  verbatim; `levitates` is now an expected key so it is not an "extra key".
- `generate_fakemon` returns whatever the model produced, unchanged — including
  or excluding `levitates`. No normalization or defaulting happens here.

### `fakemon_forge/writer.py`
- Add `"levitates"` to the `_STATS_KEYS` set.
- Change `_write_stats` so a **missing `levitates` key writes `False`** instead
  of raising `KeyError`, while all other keys stay required. Recommended:
  special-case `levitates` with `stage.get("levitates", False)` and build the
  rest with the existing strict `stage[k]` lookup — `levitates` is the only key
  with a fallback. The written value must be the stage's actual boolean when
  present. Example shape (illustrative, not prescriptive):
  `data = {k: stage[k] for k in _STATS_KEYS if k != "levitates"}` then
  `data["levitates"] = stage.get("levitates", False)`.
- `_resolve_dir`, `_write_entry`, and `write_output`'s control flow are
  unchanged. Stage directory naming, collision suffixing, and `entry.md` writing
  are untouched.
- Key ordering inside `stats.json` is not significant (JSON object); tests must
  assert on parsed values, not on serialized field order.

## Edge cases
- **Stage omits `levitates`:** `stats.json` contains `"levitates": false`; no
  exception. This is the primary new behavior.
- **Stage provides `levitates: true`:** persisted as `true`.
- **Stage provides `levitates: false`:** persisted as `false` (indistinguishable
  in output from the omitted case, which is intended).
- **Another required key missing (e.g. `ability`):** still raises `KeyError` as
  today — strictness preserved for all non-`levitates` keys.
- **Model returns a non-boolean `levitates`** (e.g. a string): out of scope —
  the value is written through as-is (no validation/coercion) in this slice.
- **Multi-stage line where some stages have `levitates` and some don't:** each
  stage is written independently; per-stage value or `False` fallback applies
  individually.

## Errors
- No new error paths, exit codes, or messages are introduced.
- The generator's existing malformed-JSON retry (2 attempts, then
  `sys.exit(1)`) and API-exception handling are unchanged; adding `levitates`
  to the prompt does not affect them.
- The writer must not raise for a missing `levitates` key (the whole point of
  the `.get` fallback). It continues to raise `KeyError` for any other missing
  required key — existing, intended strictness, not a regression.

## Constraints & dependencies
- `keep-depends-on: none` — independent of every earlier slice; slice 1/3 of #20.
- Pure-Python changes to `generator.py` and `writer.py` only; no new imports, no
  new dependencies, no ML/torch code touched (adds no `ml`-marked tests).
- Must not alter `stats.json`'s exclusion of LLM-only fields: `pokedex_entry`
  and `sprite_prompt` remain out of `stats.json` (guarded by
  `test_stats_json_excludes_llm_only_fields`, which checks `_LLM_ONLY &
  set(data.keys())` is empty). `levitates` is a stats-level field, not an
  LLM-only field.
- All existing tests must continue to pass unchanged, in particular:
  - `test_stats_json_has_required_fields` (`_STATS_KEYS <= set(data.keys())`,
    so adding a key is safe).
  - `test_stage_has_all_required_fields` (checks a fixed field tuple that does
    not include `levitates`, so unaffected).
  - The existing `test_writer.py` fixtures (`_STAGE_1` … `_STAGE_3`) do **not**
    define `levitates`; after this change they must still write successfully,
    persisting `levitates: false`.

## Testing (expected new tests)
Follow existing mocked-Mistral patterns in `tests/test_generator.py` and the
`tmp_path` patterns in `tests/test_writer.py`. No real API calls or image
generation. Suggested coverage:
- **generator — prompt mentions `levitates`:** assert the word `levitates`
  appears in `_SYSTEM_PROMPT` (import it directly, or inspect the assembled
  `client.chat.complete` messages as existing prompt tests do via
  `_get_prompt_text`). Optionally assert a definition keyword (e.g. "ground")
  is present.
- **writer — provided value:** a stage dict with `"levitates": True` writes
  `stats.json` whose parsed `data["levitates"] is True`; likewise a stage with
  `"levitates": False` writes `False`.
- **writer — omitted key:** a stage dict lacking `levitates` (e.g. the existing
  `_STAGE_1` fixture) writes `stats.json` with `data["levitates"] is False` and
  raises no exception.
- Optionally update a local `_STATS_KEYS` expectation to include `levitates`,
  taking care not to break `test_stats_json_excludes_llm_only_fields`.

## Assumptions
- **[picked default]** Persisting `levitates` into `stats.json` via `_STATS_KEYS`
  is the correct home for the flag — it matches how every other stage-level
  datum is stored. (Stated as the spec's recommended default; not independently
  confirmed by a consumer, since none exists yet.)
- **[picked default]** Only `levitates` receives a missing-key fallback; all
  other `_STATS_KEYS` entries remain required and raise if absent, preserving
  current strictness.
- **[picked default]** The missing-key default value is boolean `False` (not
  `None`, not the string `"false"`), so `stats.json` always carries a JSON
  boolean and downstream `stage.get("levitates", False)` reads uniformly.
- **[picked default]** No validation/coercion of the `levitates` value type is
  performed in this slice; a malformed non-boolean from the model is written
  through as-is. Type-hardening is deferred (out of scope).
- **[picked default]** The prompt keeps the wording "must have exactly these
  fields" and simply lists `levitates` among them, rather than marking it
  "optional". Reconciling "exactly these fields" with the writer's tolerance for
  omission is intentional: the prompt asks for it; the writer tolerates absence
  for robustness.
- **[from code]** `stats.json` continues to exclude `pokedex_entry` and
  `sprite_prompt`; `levitates` is added to the stats whitelist (`_STATS_KEYS`),
  not the LLM-only set — confirmed by `_LLM_ONLY` / `_STATS_KEYS` in
  `tests/test_writer.py`.
- **[from code]** `_write_stats`'s current `data = {k: stage[k] for k in
  _STATS_KEYS}` is the exact line to change; it currently raises `KeyError` on
  any missing whitelisted key, which is why `levitates` needs the `.get`
  fallback.
