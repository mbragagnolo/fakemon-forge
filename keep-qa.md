# Approach chosen

**Approach B — vision-description-only txt2img** (the fallback recorded in the
parent issue and in `spec.md`). Approach A (doubling the drawing onto both
halves of a 1536x768 img2img init canvas) was not adopted: it cannot be
validated in this sandbox (no torch/diffusers/GPU per `CLAUDE.md`), and the
repo's own prior GPU findings already record that img2img seeded with a
front-facing view does not rotate the subject
(`research-sprite-generation.md` §1, finding 3: "Back sprites aren't back
views"). `--image` now feeds `describe_image` only; the sprite pair comes from
the same single `generate_sprite_pair` txt2img call text-only mode uses.

# Manual QA

- [ ] `fakemon-forge --image drawing.png --description "fire lizard"` writes
      both `sprite.png` and `sprite_back.png` in the stage directory (the
      regression this slice fixes).
- [ ] The back sprite is a genuine rear view of the same creature and shares
      `sprite.png`'s exact 16-colour palette (front/back must not drift).
- [ ] The generated creature still visibly resembles the input drawing —
      the drawing now reaches the renderer only through the vision
      description, so check colours/features survive that hop.
- [ ] `fakemon-forge --image drawing.png` (no `--description`) works and
      produces the same pair.
- [ ] `fakemon-forge --image drawing.png --mode line` writes a front+back pair
      per stage, plus `sprite_back_shiny.png` and a `spritesheet.png` whose
      back cell is populated.
- [ ] Confirm on the GPU host that `--image` runs load only the txt2img
      pipeline (no second SDXL pipeline load / no VRAM regression).
- [ ] Optional, if a GPU spike is cheap: sanity-check Approach A once
      (doubled init canvas at strength 0.8) to confirm the recorded fallback
      was the right call before this closes out #61.
