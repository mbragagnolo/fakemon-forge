# Manual QA — Pin key/white/black in shiny + reference-lock background→index 0

- [ ] Generate a full stage and open all six views (front frame 1/2, back, and each shiny); confirm every background is the uniform key `(200,200,168)` at palette index 0 — no near-white haze in the backdrop.
- [ ] Inspect frame 2 and the back sprite: their noisy near-white backgrounds are flattened to the key and lock to frame 1's palette (background reads as index 0, not the reserved white slot), so transparency keying works in-game.
- [ ] Check a shiny sprite's palette: index 0 is still exactly `(200,200,168)` (the chromatic key was NOT hue-rotated), while creature colours are recoloured.
- [ ] Confirm `(255,255,255)` and `(0,0,0)` entries are identical between a normal sprite and its shiny (white/black pinned), and that the three shinies of one stage share one identical rotated palette.
- [ ] Load a generated sprite in a Gen-3 tool/emulator and verify the background is treated as transparent (keyed off index 0) across all six views.
