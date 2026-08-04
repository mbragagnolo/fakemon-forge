# Bugfix Verification: te2 LoRA silently not applied (transformers 5 conversion mapping)

Fixed in `32fb6ce` on `fix/te2-lora-conversion-mapping`.

## Original repro (now fixed)

- [ ] Run a single generation (any description) → no
      `[transformers] CLIPTextModelWithProjection LOAD REPORT` table in the
      output; no `UNEXPECTED`/`MISSING` rows for `lora_A`/`lora_B` keys.
- [ ] The generated sprite renders and still reads as Gen-3 pixel art — the
      LoRA now applies to the unet, te1 **and** te2.

## Adjacent behavior

- [ ] Front+back pair generation works (same fused pipeline, wide canvas
      split still behaves).
- [ ] Frame-2 generation works (img2img pipeline is derived from the fused
      components, so it inherits the now-complete LoRA).
- [ ] Eyeball a prompt you know well from before the fix → style/trigger
      adherence is the same or better, not degraded (te2 is SDXL's dominant
      text encoder, so some visual shift is expected and fine).

## Fix boundary cases

- [ ] Run a batch of at least 2 generations in one process → the second
      pipeline load does not raise
      `Conversion mapping ... already exists` (the `is None` guard skips
      re-registration).
- [ ] te1 still loads its LoRA half: no LOAD REPORT for `CLIPTextModel`
      either.
