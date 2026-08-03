# fakemon-forge

A CLI tool that turns a child's drawing and/or a text description into a complete Fakemon — a Pokémon-like creature with a GBA-style pixel-art sprite, base stats, typing, ability, and Pokédex entry. Optionally generates a 2- or 3-stage evolutionary line.

## How it works

1. **Vision** (if `--image` provided) — the drawing is sent to a Mistral vision model, which extracts a plain-English description of the creature's appearance, colours, and features.
2. **LLM generation** — Mistral Large invents the Fakemon: name, type(s), ability, base stats, Pokédex entry, and a visual prompt for each stage.
3. **Sprite generation** — NoobAI-XL 1.1, with the Pokemon Sprite XL PixelArt back&front LoRA fused in at scale 0.7, renders a single 1536×768 canvas in one `txt2img` call: the front view on the left half, the back view on the right. The canvas is split content-aware into its two halves (rerolling once, then falling back to a naive midline cut, if no clean split column is found), and each half is palette-quantised to 16 colours under a Gen-3 palette contract to approximate a GBA sprite. The halves are kept at their native 768×768 render size; the drop to GBA pixel scale happens where a small image is actually needed — the 64px spritesheet cells and the 32px party icon — and uses k-centroid downscaling, which gives each output pixel its source tile's dominant colour rather than blending new colours into the palette.
4. **Output** — stats, entry text, and sprite are written to an `output/` folder tree.

## Outputs

```
output/
  <Name>/
    stage1_<Name>/
      sprite.png      # 768×768 front view, 16-colour GBA-style pixel art
      sprite_back.png # 768×768 back view, same palette as the front
      spritesheet.png # all six views stitched into 64px GBA-scale cells
      ...             # plus shiny, frame-2, party-icon and footprint views
      stats.json      # types, ability, base stats
      entry.md        # Pokédex flavour text
    stage2_<Name2>/   # only with --mode line
      ...
    stage3_<Name3>/   # only with --mode line --stages 3
      ...
```

### stats.json shape

```json
{
  "name": "Frostile",
  "stage": 1,
  "types": ["Ice", "Dragon"],
  "ability": "Frostbite",
  "base_stats": {
    "hp": 50,
    "attack": 55,
    "defense": 45,
    "sp_atk": 60,
    "sp_def": 50,
    "speed": 40
  }
}
```

### entry.md shape

Plain Markdown, one or two sentences of flavour text:

```
Frostile's crystal wings shimmer in the cold air, refracting light into tiny ice shards.
It breathes out a chilly mist that can freeze small puddles in seconds.
```

## Installation

**Requirements:** Python 3.10+, a [Mistral API key](https://console.mistral.ai/), and a CUDA-capable GPU (strongly recommended — CPU inference is very slow).

```bash
# 1. Clone the repo
git clone https://github.com/mbragagnolo/fakemon-forge.git
cd fakemon-forge

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package and its dependencies
pip install -e .

# 4. Set your Mistral API key
# Windows
set MISTRAL_API_KEY=your_key_here
# macOS / Linux
export MISTRAL_API_KEY=your_key_here
```

The first run will download the `Laxhar/noobai-XL-1.1` base model weights from Hugging Face (an SDXL checkpoint, about 6-7 GB). They are cached locally afterwards.

### LoRA weights

Sprite generation also requires the Pokemon Sprite XL PixelArt **back&front** LoRA. Unlike the base model, it is not auto-downloaded — it must be fetched manually:

1. Download it from [Civitai model 378602](https://civitai.com/models/378602) ("back&front Noob v1"). A Civitai login is required.
2. Place the file at `models/loras/pkspbf_nb_v1.safetensors` (relative to the repo root). `models/` is gitignored, so this file is never committed and each clone needs its own copy.

Running the tool without the LoRA file in place fails when the sprite pipeline loads — it tries to read LoRA weights from a path that doesn't exist, so the run stops with `Error: failed to load model: ...` before any sprite is generated.

### A note on the model weights' license

This note is about the *model weights* only; fakemon-forge's own license is unchanged (see [LICENSE](LICENSE)).

NoobAI-XL is distributed under a Fair-AI license that includes a no-commercialisation clause. That's fine here — fakemon-forge is a non-monetised public portfolio project. If that clause is ever a problem for your use case, the same Civitai model page (378602) also publishes a variant of this LoRA trained against SDXL base, so stock SDXL can stand in for NoobAI-XL. That swap is a documented escape hatch rather than a supported flag — it would need a code change to `_BASE_MODEL_ID` and `_LORA_PATH`.

### GPU vs CPU

On a CUDA GPU the image generation step takes roughly 10–30 seconds per stage. On CPU it can take several minutes. The tool auto-detects CUDA and uses `float16` precision when available.

## Usage

```
fakemon-forge [--image PATH] [--description TEXT] [--mode {single,line}] [--stages {2,3}] [--tier {standard,pseudo,legendary,mythical}]
```

At least one of `--image` or `--description` must be provided.

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | — | Path to a JPG or PNG drawing of the creature |
| `--description TEXT` | — | Free-text description ("breathes fire, three tails") |
| `--mode` | `single` | `single` — one form; `line` — an evolutionary line, whose length `--stages` sets |
| `--stages` | `3` | Number of stages in an evolutionary line. Only valid with `--mode line`; a single form is one stage by definition |
| `--tier` | `standard` | Power tier controlling BST targets and lore tone (see below) |

### Power tiers

Base-stat-total targets are the medians of the observed Gen 3 distributions for
lines of that length, so a 2-stage final form is built as a *final* form rather
than as a middle stage that stopped early.

| Tier | Final BST | Notes |
|------|------------|-------|
| `standard` | ~518 (3-stage), ~468 (2-stage), ~430 (single) | Typical fully-evolved Pokémon |
| `pseudo` | ~600 | Pseudo-legendary feel; only valid with `--mode line` at `--stages 3` |
| `legendary` | ~580 | Single form only; awe-inspiring, lore-significant |
| `mythical` | ~600 | Single form only; mysterious, tied to ancient legend |

### Examples

```bash
# Text only, single form
fakemon-forge --description "a small ghost made of old clockwork gears"

# Drawing + description, full 3-stage evolutionary line
fakemon-forge --image my_drawing.png --description "fire lizard with three tails" --mode line

# Two-stage line from text
fakemon-forge --description "a mossy stone golem" --mode line --stages 2

# Legendary from a drawing
fakemon-forge --image titan_sketch.png --tier legendary

# Pseudo-legendary line from text
fakemon-forge --description "deep-sea serpent" --mode line --tier pseudo
```

## Running the tests

```bash
pip install pytest
pytest
```

The test suite mocks all external API and model calls, so no API key or GPU is needed to run them.

## Dependencies

| Package | Purpose |
|---------|---------|
| `mistralai` | LLM generation and image vision |
| `diffusers` | SDXL sprite generation (NoobAI-XL + LoRA) |
| `transformers` | Model loading support |
| `accelerate` | Device placement / mixed precision |
| `Pillow` | Downsampling and palette quantisation |
| `torch` | CUDA inference |

## License

See [LICENSE](LICENSE).
