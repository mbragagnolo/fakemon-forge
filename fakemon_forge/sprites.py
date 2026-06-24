import sys
from PIL import Image

_MODEL_ID = "lambdalabs/sd-pokemon-diffusers"
_GEN_SIZE = 512
_SPRITE_SIZE = 96
_PALETTE_COLORS = 16


def postprocess(image: Image.Image) -> Image.Image:
    image = image.resize((_SPRITE_SIZE, _SPRITE_SIZE), Image.LANCZOS)
    return image.quantize(colors=_PALETTE_COLORS)


def generate_sprite(prompt: str, output_path: str, *, pipeline) -> None:
    result = pipeline(prompt=prompt, width=_GEN_SIZE, height=_GEN_SIZE)
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def load_txt2img_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        pipe = StableDiffusionPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=torch.float32
        )
        return pipe
    except Exception as exc:
        print(
            f"Error: failed to load model '{_MODEL_ID}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
