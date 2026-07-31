# Manual QA — chibi img2img pass for `sprite_small.png` (#28)

- [ ] Run the pipeline on a text description with a real GPU/LoRA and confirm each stage dir now contains `sprite_chibi.png` (768×768 `P`-mode) alongside `sprite.png`.
- [ ] Confirm `sprite_small.png` is now derived from the chibi render and reads as a chibi caricature (big head / small body), not a literal miniature of the battle sprite.
- [ ] Force the chibi img2img pass to fail (e.g. OOM / bad pipeline) and confirm `sprite_small.png` still exists as the plain downscale of `sprite.png`, with **no** "icon generation failed" warning printed.
- [ ] Confirm `sprite_chibi.png` does **not** appear in the stitched `spritesheet.png` (still the 6 shipped views).
- [ ] Run in line mode (3 stages) and confirm each stage independently produces its own `sprite_chibi.png` / `sprite_small.png` and falls back per-stage.
- [ ] Eyeball whether `_CHIBI_TAGS` / default `strength=0.8` actually yield caricature proportions from img2img (the flagged GPU-spike tunable).
