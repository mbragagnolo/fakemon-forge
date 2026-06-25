# fakemon-forge

A CLI tool that turns a child's drawing and/or a text description into a complete Fakemon — a Pokémon-like creature with a GBA-style pixel-art sprite, base stats, typing, ability, and Pokédex entry. Optionally generates a full 3-stage evolutionary line.

## How it works

1. **Vision** (if `--image` provided) — the drawing is sent to a Mistral vision model, which extracts a plain-English description of the creature's appearance, colours, and features.
2. **LLM generation** — Mistral Large invents the Fakemon: name, type(s), ability, base stats, Pokédex entry, and a visual prompt for each stage.
3. **Sprite generation** — `lambdalabs/sd-pokemon-diffusers` renders a 512×512 image, which is then downsampled to 96×96 and palette-quantised to 16 colours to approximate a GBA sprite.
4. **Output** — stats, entry text, and sprite are written to an `output/` folder tree.

## Outputs

```
output/
  <Name>/
    stage1_<Name>/
      sprite.png      # 96×96 GBA-style pixel art
      stats.json      # types, ability, base stats
      entry.md        # Pokédex flavour text
    stage2_<Name2>/   # only with --mode line
      ...
    stage3_<Name3>/   # only with --mode line
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

The first run will download the `lambdalabs/sd-pokemon-diffusers` model weights from Hugging Face (~1.7 GB). They are cached locally afterwards.

### GPU vs CPU

On a CUDA GPU the image generation step takes roughly 10–30 seconds per stage. On CPU it can take several minutes. The tool auto-detects CUDA and uses `float16` precision when available.

## Usage

```
fakemon-forge [--image PATH] [--description TEXT] [--mode {single,line}] [--tier {standard,pseudo,legendary,mythical}]
```

At least one of `--image` or `--description` must be provided.

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | — | Path to a JPG or PNG drawing of the creature |
| `--description TEXT` | — | Free-text description ("breathes fire, three tails") |
| `--mode` | `single` | `single` — one form; `line` — full 3-stage evolutionary line |
| `--tier` | `standard` | Power tier controlling BST targets and lore tone (see below) |

### Power tiers

| Tier | Stage 3 BST | Notes |
|------|------------|-------|
| `standard` | ~520 | Typical fully-evolved Pokémon |
| `pseudo` | ~600 | Pseudo-legendary feel; only valid with `--mode line` |
| `legendary` | ~580 | Single form only; awe-inspiring, lore-significant |
| `mythical` | ~600 | Single form only; mysterious, tied to ancient legend |

### Examples

```bash
# Text only, single form
fakemon-forge --description "a small ghost made of old clockwork gears"

# Drawing + description, full evolutionary line
fakemon-forge --image my_drawing.png --description "fire lizard with three tails" --mode line

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

The test suite (113 tests) mocks all external API and model calls, so no API key or GPU is needed to run them.

## Dependencies

| Package | Purpose |
|---------|---------|
| `mistralai` | LLM generation and image vision |
| `diffusers` | Stable Diffusion sprite generation |
| `transformers` | Model loading support |
| `accelerate` | Device placement / mixed precision |
| `Pillow` | Downsampling and palette quantisation |
| `torch` | CUDA inference |

## License

See [LICENSE](LICENSE).
