# Prototype findings: NoobAI-XL + Back&Front LoRA vs current stack (+ k-centroid A/B)

## Question

Does NoobAI-XL 1.1 + the Pokemon Sprite XL PixelArt LoRA (back&front variant)
beat the current SD1.5 stack (dreamshaper-8 + pksp768) at final 64×64 Gen-3
quality — **including genuine back views** — on the 8GB Quadro RTX 4000?
By-product: does k-centroid beat NEAREST as the 768→64 downscaler?

## What was tried

- Same 5 handcrafted creature descriptions (knight, fox, bird, serpent, sprout —
  body plans that stress the known weak spots) through both stacks, seed 42.
- Current stack: production code paths (`generate_sprite` +
  `generate_sprite_img2img` backside chain, exactly as `main.py` wires them).
- NoobAI stack: `StableDiffusionXLPipeline` + `pkspbf_nb_v1.safetensors`,
  prompt `"gen3, {desc}, white background"`, neg standard quality tags,
  28 steps, CFG 5.5, Euler a, `enable_model_cpu_offload` + VAE tiling.
- Comparison grid: every output → production Gen-3 quantize → 64×64 via both
  NEAREST and k-centroid (`out/comparison_grid.png`).

Three integration snags were solved on the way (all captured in `gen_noobai.py`):

1. **LoRA loading**: kohya SDXL LoRAs crash diffusers 0.38's
   `load_lora_weights` (`IndexError` in rank detection). Fix = the production
   `_apply_lora` trick extended per-encoder — **te1 (`CLIPTextModel`) needs the
   `text_model.` level stripped, te2 (`CLIPTextModelWithProjection`) must keep
   it** — plus passing `unet_config=pipe.unet.config` to `lora_state_dict` so
   kohya SGM block names get remapped.
2. **Canvas orientation**: the model page says "768x1536" but the pair is
   **side-by-side — width=1536, height=768** (verified from the Civitai sample
   images; portrait canvases produce one vertically-stretched creature).
3. NoobAI backgrounds come out slightly vignetted, tripping
   `_flatten_background_to_key`'s gradient warning (fallback path handled it;
   the author's img2img-from-white-canvas trick is the known fix).

## Result

**The NoobAI stack wins decisively on 5/5 fronts and 4/5 backs.**

- Fronts: coherent anatomy, clean 1px outlines, real Gen-3-style tone ramps,
  grid-aligned pixels. The current stack's mushy silhouettes (knight, serpent)
  become readable creatures. The serpent — an unreadable loop-blob on the
  current stack — becomes a proper horned sea-dragon.
- Backs: **genuine back views with held identity** for knight, fox, bird
  (fox back is textbook Gen 3 back-sprite composition). The current stack
  produced zero real back views (all five are noisy front variants, one with
  the face still visible).
- Failure modes found: 1/5 back view clipped at the canvas seam (serpent coil
  crossed the midline — needs margin/centering control or a bbox check +
  reroll), 1/5 moderate front↔back identity drift (sprout gained pink buds on
  the back view only).
- Feasibility: **25–31s per 1536×768 pair, peak VRAM 5.41 GiB** with cpu
  offload — comfortably inside 8GB, and one call replaces the current
  front (≈11s) + backside img2img (≈8s) two-step while eliminating its
  identity-drift failure entirely.
- k-centroid vs NEAREST: k-centroid is visibly cleaner on every current-stack
  sprite (orphan-pixel speckle collapses into solid fields, outlines survive).
  On NoobAI output the delta is smaller (its pixels are already near
  grid-aligned) but still positive. No regression observed anywhere.

## Recommendation

There is no spec for this feature yet — the research doc
(`research-sprite-generation.md`) played that role. Fold these findings in and
write the real spec:

1. **Adopt the NoobAI-XL + back&front LoRA as the sprite backend** (research
   doc Tier 1 confirmed). Spec must cover: the three integration snags above
   (LoRA loader shim, landscape pair canvas + split, white-background
   handling), seam-clipping mitigation (margin prompt/bbox check + reroll),
   and what replaces the backside img2img path (delete vs fallback).
2. **Adopt k-centroid for the 768→64 downscale** (Tier 0 confirmed) —
   independent, zero-risk, also improves any remaining SD1.5 outputs.
3. Open question for the spec: frame-2 animation with the new backend
   (low-strength img2img on the NoobAI front half? unchanged procedural
   squash fallback?) — untested in this spike.
4. License note for the public README: NoobAI base carries a
   no-commercialisation clause (fine for the portfolio; SDXL-variant LoRA is
   the documented fallback).
