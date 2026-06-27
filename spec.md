# Spec: fakemon-forge

## Summary
A CLI tool that takes a child's drawing (scan/photo) and/or a text description and generates a complete Pokémon-like creature (Fakemon) — including a GBA-style pixel art sprite, base stats, typing, ability, and Pokédex entry. Optionally generates a full 3-stage evolutionary line.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `--image` | file path (jpg/png) | No* | Scan or photo of a hand-drawn creature |
| `--description` | string | No* | Free-text description of the creature ("breathes fire, three tails") |
| `--mode` | `single` \| `line` | No | Whether to generate one form or a 3-stage evolutionary line. Defaults to `single`. |

*At least one of `--image` or `--description` must be provided.

## Outputs

A folder tree under `output/` named after the generated Fakemon:

```
output/
  <Name>/
    stage1_<Name>/
      sprite.png        # 96x96 GBA-style pixel art
      stats.json        # base stats, type, ability
      entry.md          # Pokédex entry
    stage2_<Name2>/     # only if --mode line
      sprite.png
      stats.json
      entry.md
    stage3_<Name3>/     # only if --mode line
      sprite.png
      stats.json
      entry.md
```

`stats.json` shape:
```json
{
  "name": "Flamburr",
  "stage": 1,
  "types": ["Fire"],
  "ability": "Blaze",
  "base_stats": {
    "hp": 45,
    "attack": 52,
    "defense": 43,
    "sp_atk": 60,
    "sp_def": 50,
    "speed": 65
  }
}
```

## Behavior

### Pipeline (per run)

1. **Vision step** (only if `--image` provided): send the drawing to Mistral's vision model to extract a plain-English description of the creature's appearance, colors, and notable features.

2. **LLM generation step**: send the combined vision description + user `--description` to `mistral-large`. The prompt instructs it to produce, for each stage required by `--mode`:
   - A portmanteau-style name
   - Type(s) (one or two)
   - One ability
   - Six base stats (HP, Atk, Def, Sp.Atk, Sp.Def, Speed) balanced to a total BST appropriate for the stage
   - A short Pokédex entry (2–3 sentences)
   - A detailed visual prompt for sprite generation (creature appearance, colors, pose, style keywords)

3. **Image generation step** (per stage):
   - Load `lambdalabs/sd-pokemon-diffusers` via `diffusers`
   - If `--image` provided: run **img2img** using the drawing as conditioning image + the stage's visual prompt
   - If no image provided: run **txt2img** using the stage's visual prompt
   - Generate at 512×512
   - Post-process: downsample to 96×96 + palette quantization to 16 colors (GBA approximation)
   - Save as `sprite.png`

4. **Write outputs**: serialize `stats.json` and `entry.md` for each stage, create folder structure.

### Evolutionary line logic (`--mode line`)
- Stage 1: the base form as described by the user/drawing
- Stage 2 & 3: LLM invents evolved forms — larger/more powerful/more complex visually, with escalating BST
- All three stage names and visual prompts are generated in a single LLM call to ensure thematic consistency

## Edge cases

| Case | Handling |
|------|----------|
| Neither `--image` nor `--description` provided | Exit with a clear error message before any API calls |
| `--image` path does not exist or is not an image | Validate file exists and has an image extension; exit with error |
| Mistral API returns malformed JSON for stats | Retry once; if it fails again, exit with error and print raw LLM response |
| Image generation produces a blank/corrupt output | Save it anyway and warn the user; do not block the rest of the output |
| `--mode single` with a name collision in `output/` | Append a numeric suffix (e.g. `Flamburr_2/`) rather than overwriting |

## Errors

| Failure mode | Surface |
|---|---|
| Missing required input | `sys.exit(1)` with human-readable message |
| Mistral API auth failure | `sys.exit(1)` with message pointing to env var |
| Diffusers model load failure (OOM, missing weights) | `sys.exit(1)` with model name and exception |
| Unexpected LLM output structure | Retry once, then `sys.exit(1)` and print raw response for debugging |

## Constraints & dependencies

- **Language**: Python 3.10+
- **LLM**: Mistral API (`mistral-large`; vision step uses a Mistral vision-capable model e.g. `pixtral-large` or `mistral-small` with vision). API key read from `MISTRAL_API_KEY` environment variable.
- **Image generation**: `diffusers` library, model `lambdalabs/sd-pokemon-diffusers`
- **Image post-processing**: `Pillow` for downsampling and palette quantization
- **No separate server required** — all image generation runs in-process
- **Style target**: GBA Gen 3 Pokémon sprite aesthetic (96×96, limited palette, white or transparent background)
- **Output format**: PNG sprites, JSON stats, Markdown Pokédex entries

## Open questions / assumptions

- [ASSUMED] Mistral vision step uses `pixtral-large` or equivalent vision model available on the Mistral API. If the user's Mistral plan does not include vision, a fallback to text-only mode should be added.
- [ASSUMED] The 16-color palette quantization uses Pillow's `Image.quantize()` with no specific GBA palette — a real GBA palette constraint could be added later.
- [ASSUMED] Sprite background is white (not transparent) for simplicity; transparency can be added as a post-processing flag later.
- [ASSUMED] No animated sprites — static PNG only.
- [ASSUMED] BST targets per stage: ~300 (stage 1), ~420 (stage 2), ~520 (stage 3), loosely following Gen 3 conventions.
- [OPEN] A signature move was out of scope for v1 but was noted as a potential addition.
- [OPEN] Catch rate, egg group, and gender ratio were out of scope for v1.
