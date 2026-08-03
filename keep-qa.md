# Manual QA — squash-init frame 2 (issue #67)

Run on the host, where the real torch/diffusers/SDXL stack is available (the
sandbox auto-skips the `ml` tests).

- [ ] Run `pytest tests/test_sprites_ml.py -k frame2` on the host and confirm
      `test_frame2_pipeline_init_image_is_squashed_frame1_not_raw_front` and
      `test_frame2_pipeline_called_with_low_strength` both pass.
- [ ] Generate a creature end-to-end and eyeball `sprite.png` vs
      `sprite_frame2.png`: frame 2 should read as a real breathing/pose change
      (body compressed, feet planted), not a recolour of frame 1.
- [ ] Confirm frame 2 is *not* just the bare `procedural_squash` output on most
      creatures — i.e. the img2img candidate is being accepted, not silently
      falling back. Compare against `procedural_squash(frame1)` directly.
- [ ] Check frame 2 still shares frame 1's exact 16-colour palette and size,
      and that the stitched sheet + `sprite_frame2_shiny.png` are unaffected.
- [ ] Sanity-check the back sprite path (`generate_sprite_img2img`) still works
      — it now goes through the extracted `_run_img2img_on_image` helper.
- [ ] Pass a non-palette-mode PNG as `front_sprite_path` and confirm
      `generate_frame2` still fails gracefully (main.py catches and warns); the
      `ValueError` now fires before the pipeline call instead of after.

## Known pre-existing failure (not from this change)

- [ ] `test_frame2_falls_back_to_squash_on_garbage_candidate` fails on the host
      and has since at least commit `027e604` (four slices back). Its premise is
      wrong, not the code: `quantize_to_reference` applies
      `Color(1.1)`/`Contrast(1.1)`, so a byte-identical candidate round-trips at
      `difference_ratio ≈ 0.171`, which is *inside* the `[0.02, 0.30]` band and
      gets accepted. The test therefore never exercises the fallback it names.
      Decide separately whether the fixture or the band's `low` bound is wrong —
      out of scope for this slice, which was told to leave the band untouched.
