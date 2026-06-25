import sys
from PIL import Image, ImageEnhance

_MODEL_ID = "lambdalabs/sd-pokemon-diffusers"
_GEN_SIZE = 512
_SPRITE_SIZE = 96
_PALETTE_COLORS = 16


_CLIP_MAX_WORDS = 50  # CLIP tokenises to ~77 tokens; 50 words stays safely under


def _clip_prompt(prompt: str) -> str:
    words = prompt.split()
    return " ".join(words[:_CLIP_MAX_WORDS]) if len(words) > _CLIP_MAX_WORDS else prompt


def postprocess(image: Image.Image) -> Image.Image:
    image = image.resize((_SPRITE_SIZE, _SPRITE_SIZE), Image.NEAREST)
    image = ImageEnhance.Color(image).enhance(1.8)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    return image.quantize(colors=_PALETTE_COLORS)


def generate_sprite(prompt: str, output_path: str, *, pipeline) -> None:
    result = pipeline(prompt=_clip_prompt(prompt), width=_GEN_SIZE, height=_GEN_SIZE)
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def generate_sprite_img2img(
    prompt: str, image_path: str, output_path: str, *, pipeline
) -> None:
    init = Image.open(image_path).convert("RGB").resize((_GEN_SIZE, _GEN_SIZE), Image.LANCZOS)
    result = pipeline(prompt=_clip_prompt(prompt), image=init)
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def _device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_txt2img_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        device, dtype = _device_and_dtype()
        pipe = StableDiffusionPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=dtype, safety_checker=None
        )
        return pipe.to(device)
    except Exception as exc:
        print(
            f"Error: failed to load model '{_MODEL_ID}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_img2img_pipeline():
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        device, dtype = _device_and_dtype()
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=dtype, safety_checker=None
        )
        return pipe.to(device)
    except Exception as exc:
        print(
            f"Error: failed to load model '{_MODEL_ID}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
