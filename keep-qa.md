# Manual QA — `levitates` flag

- [ ] Generate a floating/ghost-type Fakemon and confirm each stage's `stats.json` contains `"levitates": true`.
- [ ] Generate a normal ground-dwelling Fakemon and confirm `stats.json` shows `"levitates": false`.
- [ ] Simulate a model response that omits `levitates` (or use a stage dict without it) and confirm `stats.json` still writes `"levitates": false` with no crash.
- [ ] Confirm `stats.json` still excludes `pokedex_entry` and `sprite_prompt`, and still includes `name`, `stage`, `types`, `ability`, `base_stats`.
- [ ] Confirm `entry.md` output is unchanged (no `levitates` leakage into the flavour text).
- [ ] Run `pytest` from the repo root and confirm green (ml tests skip in the slim sandbox — expected).
