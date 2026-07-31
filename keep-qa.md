# Manual QA — back sprite shared palette + cross-view shiny consistency

- [ ] Generate a Fakemon from a text description and confirm `sprite_back.png` uses the **exact same 16-colour palette** as `sprite.png` (frame 1).
- [ ] Confirm `sprite_back_shiny.png`, `sprite_shiny.png`, and `sprite_frame2_shiny.png` all share **one** consistent rotated shiny palette.
- [ ] Run the img2img path (`--image drawing.png`) and confirm the back sprite still seeds from the user's drawing but locks its palette to `sprite.png` (frame 1), not the drawing.
- [ ] In line mode (3 stages), confirm each stage's back sprite locks to its own stage's `sprite.png` and its three shinies stay mutually consistent.
- [ ] Confirm the front `sprite.png` itself is unchanged (it defines the palette, never reference-locked) and back-sprite colours landing far from the palette posterize gracefully rather than erroring.
- [ ] Confirm the exported `.ini` is still valid (unchanged — references no sprite files).
