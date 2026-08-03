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

## Follow-up spike: frame-2 on the NoobAI backend

Question: does the production frame-2 recipe (low-strength img2img, "open
mouth" tag, `build_frame2` acceptance band) transfer to the new backend?

**Yes — 5/5 ACCEPTED.** `StableDiffusionXLImg2ImgPipeline` + same LoRA on the
NoobAI front half at production strength 0.35: difference ratios 0.079–0.215
(band is [0.02, 0.30]), ~8s per frame (23s first incl. warmup), peak 5.42 GiB.
Visually (`out/frame2_strip.png`, `out/*_anim.gif`): identity held, motion
reads as subtle pose/shading shifts — the intended breathing-frame feel. The
procedural-squash fallback was never needed. The XL img2img call takes plain
`prompt=` (no compel/`prompt_embeds`), so the SD1.5 `_encode_prompt` path
does not carry over.

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
3. Frame-2: keep the production recipe unchanged on the new backend (verified
   working, see follow-up spike) — only the pipeline class and prompt encoding
   change (XL img2img, plain `prompt=`, no compel).
4. License note for the public README: NoobAI base carries a
   no-commercialisation clause (fine for the portfolio; SDXL-variant LoRA is
   the documented fallback).
