# Manual QA — footprint generation in main.py

- [ ] Run a single-form generation and confirm a `footprint.png` (16×16 RGBA) is written in the stage dir.
- [ ] Run `--mode line` (3-stage) and confirm each stage dir gets a `footprint.png` scaled by tier (stage 1 smallest, stage 3 largest ≈0.9).
- [ ] Generate a Fakemon with `levitates: true` and confirm its footprint is all-transparent (blank).
- [ ] Confirm `spritesheet.png` still contains no footprint tile and the `.ini` export is unchanged.
- [ ] Force a footprint failure and confirm a `Warning: footprint generation failed for <name>` prints to stderr while the run finishes normally (exit 0, remaining stages + export still produced).
- [ ] Confirm a stage whose front sprite failed produces no `footprint.png` and no footprint warning.
