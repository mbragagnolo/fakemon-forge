# Sprite generation — upgrade research (discovery, 2026-08-03)

Pure discovery, no spec/code. Current stack: `Lykon/dreamshaper-8` (SD1.5-class) +
`pksp768_V2-1` LoRA @768px, DPM++ Karras, in-house Gen-3 post (16-colour contract,
key flatten, NEAREST ÷12 → 64px cells). GPU: Quadro RTX 4000 — Turing, 8GB VRAM,
fp16 OK, **no bf16**.

## 1. Diagnosis (verified on actual output)

Inspected `output/Corlance`, `output/Mossit`, `output/Dracobit` at native 768 and at
final 64px (NEAREST ÷12, mirroring `stitch_spritesheet`):

1. **Misaligned pseudo-pixel grid.** The LoRA paints "pixel-art style" with irregular
   ~8–14px pseudo-pixels that don't align to the ÷12 sampling grid. NEAREST picks one
   arbitrary point per cell → orphan pixels, broken outlines, mush at 64px.
2. **SD1.5 composition ceiling.** Incoherent anatomy under detail load (Corlance's
   arms, Dracobit's disjointed limbs). Confirms "model, not prompting" hypothesis.
3. **Back sprites aren't back views.** Corlance's back sprite is another ¾-front view.
   img2img + `backside` tag doesn't rotate the subject.
4. **No Gen-3 shading discipline.** Real Gen 3 sprites: 1px dark outline + 2–3 tone
   ramps per hue. Current output: noisy banded shading that quantisation can't fix.

## 2. Convergent headline finding

**The author of pksp768 (titansteng on Civitai / sWizad on HF) shipped a direct
successor in Jan 2025**: [Pokemon Sprite XL PixelArt LoRA](https://civitai.com/models/378602/pokemon-sprite-xl-pixelart-lora)
— base **NoobAI-XL Epsilon v1.1** (Illustrious/SDXL family). 39.7K downloads, the
de-facto standard now. Same recipe/tag vocabulary as pksp768 (gen1–gen5, type tags),
same 768-generate → ÷8 workflow.

- **Back&Front variant**: generates front + back **in one 768×1536 canvas**
  (÷8 → 96×192, split top/bottom) — one denoising pass, shared latent →
  **identity consistency by construction**. Replaces the entire img2img-backside
  step (failure mode #3).
- Author also recommends img2img from a plain white canvas at denoise 1.0 instead of
  txt2img.
- An SDXL-base v1.0 variant exists on the same page (fallback if NoobAI's licence
  addendum is unwanted).
- Two of three research agents independently ranked this #1.

## 3. Local model landscape (8GB Turing verdicts)

| Option | Verdict |
|---|---|
| **NoobAI-XL / Illustrious-XL (SDXL arch)** | Fits with fp16 + `enable_model_cpu_offload` + tiled VAE; ~30–60s/768px image. Where the sprite-LoRA ecosystem lives now. **Primary candidate.** |
| SDXL base | Same fit; OpenRAIL++; thinner sprite ecosystem than NoobAI. |
| Flux.1-dev GGUF Q4 | Fits (~6.8GB, diffusers native GGUF, `compute_dtype=fp16`), but ~2–5 min/image on Turing + non-commercial licence. Quality ceiling (Retro Diffusion chose Flux arch for its flagship). Optional `--hq` backend at most. [Pokemon Sprite-Generator Flux LoRA](https://civitai.com/models/854918/pokemon-sprite-generator-v1) exists. |
| Flux.1-schnell | Apache 2.0 but sprite LoRAs all target dev. |
| SD 3.5 Medium | Runs fine, zero sprite-LoRA ecosystem. Dead end. |
| Sana / PixArt-Sigma / Lumina 2 | Ruled out: licence (Sana NC), bf16-hostile on Turing, and/or no ecosystem. |
| Pony V6 XL | Viable, strictly dominated by NoobAI for this niche. |

Licences: NoobAI = FAIPL-1.0-SD **+ no-commercialisation clause** ([note](https://x.com/satos73/status/1899426295492309103)) — OK for a non-monetised public portfolio; SDXL-variant fallback if that itches. Illustrious later versions re-released under OpenRAIL ([Civitai article](https://civitai.com/articles/18619/what-the-license)).

Stackable pixel LoRAs: [Pixel Art XL (NeriJS)](https://civitai.com/models/120096/pixel-art-xl),
[Illustrious Pixel Art](https://civitai.com/models/43820/illustrious-pixel-art-xl-and-15-by-creativehotia),
[Elin sprite style](https://civitai.com/models/1084875/pixel-art-sprite-elin-style-illustriousxl-noob-lora).

## 4. Hosted API landscape

| Service | Fit | Cost/creature |
|---|---|---|
| **Gemini Nano Banana 2** (`gemini-3.1`-family image models) | Best identity-consistency per dollar. Editing mode solves front→back ("same creature from behind") and frame 2 ("subtle breathing pose"). Community-proven sprite workflows ([Robotic Ape lessons](https://roboticape.com/2026/03/07/generating-game-sprites-with-gemini-image-generation-nano-banana-pro-lessons-learned/): #00FF00 chroma-key bg, ban gradients). Output is fake-pixel-grid/high-colour — existing quantiser absorbs that. $0.034–0.067/img. | ~$0.10–0.20 |
| **Retro Diffusion API** ([retrodiffusion.ai](https://retrodiffusion.ai/)) | Best native pixel-art authenticity: true grid, `input_palette` (16-colour constraint at gen time!), `remove_bg`, `battle_sprites` style, and **idle/`subtle_motion` animation endpoint = turnkey frame 2**. Back sprite unproven (`character_turnaround`, RD Pro reference images — needs spike). $0.015–0.18/img, anims $0.07–0.25. Outputs commercially yours. | ~$0.20–0.50 |
| **PixelLab** ([pixellab.ai](https://www.pixellab.ai/pixellab-api)) | Only API with a dedicated **rotate endpoint (8 views incl. back) at native 64×64** + forced palettes + official Python SDK. Risk: tuned for humanoid side-scroller chars; monster battle poses unproven. ~40 free gens → zero-cost spike. | ~$0.05–0.30 |
| GPT Image 2 | Fine NB alternative ($0.05/medium edit) but NB2 cheaper + stronger community sprite workflow. | — |
| FLUX API (Kontext / FLUX.2) | Cheap editing ($0.015–0.045) with identity preservation; thin pixel-art evidence. | — |
| Scenario | Ruled out: subscription + would need training on Pokémon sprites (IP landmine for public repo). | — |
| Recraft / Ideogram / Leonardo | Ruled out: no identity mechanism → back-sprite problem unsolved. | — |

Hybrid worth noting: **NB2 designs the consistent front/back/frame trio → RD img2img
(low strength + `input_palette`) pixel-art-ifies each** — ~$0.35/creature, best of both.

## 5. Custom LoRA training (breaks the ceiling durably)

- **Data**: [PokeAPI/sprites](https://github.com/PokeAPI/sprites) — Gen 3–5
  front/back/shiny/back-shiny per game ≈ 3–6k images after dedup. (Nintendo IP —
  private-pipeline concern only; note pksp768 was itself trained on the same sprites,
  so status quo is no cleaner.) [veekun](https://veekun.com/dex/downloads) has pre-packed
  back-sprite sheets. msikma/pokesprite = menu icons only, useless here.
- **Recipe (proven by pksp768 + successors)**: 96px sprites nearest ×8 → 768,
  BLIP captions + controlled tags. Improvement available: **deterministic captions from
  PokeAPI metadata** (types, colour, body shape, `frontside`/`backside`, `shiny`,
  `genN`) → reliable view conditioning. Danbooru-style tags on NoobAI bases.
- **Paired-canvas retrain**: concatenate front+back into one 96×192 canvas per sample
  (the NoobAI LoRA's trick). Nothing about it needs SDXL — an SD1.5 paired-canvas LoRA
  could even train locally on the 8GB card.
- **Cost**: SDXL LoRA on 8GB Turing = no (60+ hrs). Cloud: 4090 @ $0.29–0.69/hr →
  **$1–5/run**; [Civitai on-site trainer](https://civitai.com/articles/14024/sdxl-loras-training-guide-civitai-trainer)
  ≈ **$0.50–1/run** (supports Illustrious/NoobAI bases). Cheap enough to iterate weekly.

## 6. Pipeline techniques (compound with everything)

- **k-centroid downscale** (Astropulse): per-tile k-means instead of NEAREST's single
  arbitrary sample — directly fixes failure mode #1. MIT Python impl:
  [pixeldetector](https://github.com/Astropulse/pixeldetector). Drop-in for the
  768→64 step, keep existing quantise after. ~1 hour, zero risk, cheapest real win.
- Adjacent: [sd-webui-pixelart](https://github.com/mrreplicart/sd-webui-pixelart)
  (recommended by pksp768's author), [sd-palettize](https://github.com/Astropulse/sd-palettize),
  [sd-pixel](https://github.com/Leodotpy/sd-pixel) (fat-pixel fixer).
- **Reference-only ControlNet** (SD1.5, free, 8GB-fine): commonly recommended for view
  changes — cheap A/B vs current img2img backside if staying on SD1.5.
- Scribble/lineart ControlNet with mirrored front silhouette: locks proportions for
  back sprite; unnecessary if paired-canvas generation is adopted.
- IP-Adapter for back views: evidence is face/human-centric; known to drop
  attire/surface detail — low expectations for creatures.
- [WAN Pokemon sprite animation video LoRA](https://civitai.com/models/1595383/pokemon-sprite-animation-video-lora)
  exists for the animation angle (overkill for 2-frame, but noted).

## 7. Recommendation (tiered, compoundable)

1. **Tier 0 — k-centroid swap** (~1 hr, free): replace NEAREST ÷12 with k-centroid in
   `stitch_spritesheet`/`postprocess`. Improves *current* output immediately and
   benefits every later tier.
2. **Tier 1 — NoobAI-XL + Pokemon Sprite XL LoRA (Back&Front variant)** — the main
   move. Same author, same tag vocabulary, minimal pipeline reshape (SDXL pipeline +
   768×1536 canvas + split). Fixes composition ceiling *and* back-sprite identity in
   one step. ~30–60s/image on the 8GB card with offload. Licence: non-commercial
   clause acceptable for portfolio; SDXL variant as fallback.
3. **Tier 2 — optional API backend** (`--cloud` flag): NB2 for design quality +
   view-consistent edits, and/or Retro Diffusion for authentic pixel grid + idle-anim
   frame 2. ~$0.10–0.50/creature. Never the default for a local-first CLI.
4. **Tier 3 — custom LoRA retrain** (later, if Tier 1 still underwhelms):
   paired-canvas + deterministic PokeAPI-metadata captions on Illustrious/NoobAI via
   Civitai trainer or rented 4090, ~$1–5/iteration.

## 8. Open questions → spike candidates

- Does the Back&Front variant hold quality for *novel* fakemon (vs known species)?
  → prototype: same 5 prompts, current stack vs NoobAI stack, compare at 64px.
- k-centroid vs NEAREST A/B on existing native-768 outputs (no regen needed —
  `output/Corlance` etc. are valid inputs; legacy 96px outputs are not).
- 768×1536 canvas on 8GB: does it fit with cpu-offload + tiled VAE, and how slow?
- Retro Diffusion back-sprite reliability (`character_turnaround` / RD Pro reference
  images) — read their ToS page manually first (JS SPA, agent couldn't render).
- PixelLab rotate-to-back on a monster sprite — free trial, zero-cost spike.
