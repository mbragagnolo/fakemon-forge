# Manual QA — per-stage `sprite_small.png` icon

- [ ] Run the pipeline for a single Fakemon and confirm `sprite_small.png` (32x64, two stacked 32x32 frames, teal background) appears in the stage dir alongside `sprite.png`.
- [ ] Run with `--mode line` and confirm each of the 3 stage dirs gets its own `sprite_small.png`.
- [ ] Confirm `spritesheet.png` is unchanged — `sprite_small.png` is NOT stitched into it.
- [ ] Force an icon failure (e.g. feed a non-`P`-mode `sprite.png`) and confirm a single `Warning: icon generation failed for <name>: ...` prints to stderr, the run does not abort, and the other views (back/frame2/shiny/spritesheet) are still produced for that stage.
