import colorsys
import hashlib
import sys
from pathlib import Path
from PIL import Image, ImageEnhance

_BASE_MODEL_ID = "Lykon/dreamshaper-8"
_LORA_PATH = Path(__file__).parent.parent / "models" / "loras" / "pksp768_V2-1.safetensors"
_LORA_SCALE = 0.7
_GEN_SIZE = 768
_NUM_STEPS = 30
_CFG_SCALE = 7
_SPRITE_SIZE = 96
_PALETTE_COLORS = 16

_TYPE_TAGS = {
    "Normal": "normaltype", "Fire": "firetype", "Water": "watertype",
    "Electric": "electrictype", "Grass": "grasstype", "Ice": "icetype",
    "Fighting": "fightingtype", "Poison": "poisontype", "Ground": "groundtype",
    "Flying": "flyingtype", "Psychic": "psychictype", "Bug": "bugtype",
    "Rock": "rocktype", "Ghost": "ghosttype", "Dragon": "dragontype",
    "Dark": "darktype", "Steel": "steeltype", "Fairy": "fairytype",
}

_GEN_STYLE = "gen3"


def build_prompt(sprite_prompt: str, types: list[str], extra_tags: list[str] | None = None) -> str:
    type_tags = [_TYPE_TAGS[t] for t in types if t in _TYPE_TAGS]
    all_tags = type_tags + [_GEN_STYLE] + (extra_tags or [])
    tags = " ".join(all_tags)
    return f"{tags} {sprite_prompt}".strip() if tags else sprite_prompt


def _encode_prompt(prompt: str, pipeline):
    from compel import Compel
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    return compel(prompt)


def postprocess(image: Image.Image) -> Image.Image:
    image = image.resize((_SPRITE_SIZE, _SPRITE_SIZE), Image.NEAREST)
    image = ImageEnhance.Color(image).enhance(1.1)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    return image.quantize(colors=_PALETTE_COLORS)


def _is_achromatic(r: int, g: int, b: int) -> bool:
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return lum < 40 or lum > 215


def generate_shiny(sprite_path: str, name: str, output_path: str) -> None:
    """Derive a shiny palette from an existing sprite by hue-rotating mid-tone colors.

    Black (lum < 40) and white (lum > 215) entries are preserved exactly.
    The hue shift is seeded from the Pokémon's name so each one is unique.
    """
    img = Image.open(sprite_path)
    if img.mode != "P":
        raise ValueError(f"Expected palette-mode image, got {img.mode}")

    hue_shift = (int(hashlib.md5(name.encode()).hexdigest(), 16) % 300 + 30) / 360.0

    flat = img.getpalette()  # [R, G, B, R, G, B, ...] × 256
    new_palette = []
    for i in range(0, len(flat), 3):
        r, g, b = flat[i], flat[i + 1], flat[i + 2]
        if _is_achromatic(r, g, b):
            new_palette.extend([r, g, b])
        else:
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h = (h + hue_shift) % 1.0
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            new_palette.extend([round(nr * 255), round(ng * 255), round(nb * 255)])

    shiny = img.copy()
    shiny.putpalette(new_palette)
    shiny.save(output_path)


def _make_generator(seed: int | None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    return g


def generate_sprite(
    prompt: str, types: list[str], output_path: str, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None,
) -> None:
    conditioning = _encode_prompt(build_prompt(prompt, types, extra_tags), pipeline)
    result = pipeline(
        prompt_embeds=conditioning,
        width=_GEN_SIZE,
        height=_GEN_SIZE,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
    )
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def generate_back_sprite(
    prompt: str, types: list[str], output_path: str, *, pipeline, seed: int | None = None
) -> None:
    generate_sprite(prompt, types, output_path, pipeline=pipeline, extra_tags=["backside"], seed=seed)



def generate_sprite_img2img(
    prompt: str, types: list[str], image_path: str, output_path: str, *, pipeline,
    extra_tags: list[str] | None = None, seed: int | None = None, strength: float = 0.8,
) -> None:
    init = Image.open(image_path).convert("RGB").resize((_GEN_SIZE, _GEN_SIZE), Image.LANCZOS)
    conditioning = _encode_prompt(build_prompt(prompt, types, extra_tags), pipeline)
    result = pipeline(
        prompt_embeds=conditioning,
        image=init,
        num_inference_steps=_NUM_STEPS,
        guidance_scale=_CFG_SCALE,
        generator=_make_generator(seed),
        strength=strength,
    )
    sprite = postprocess(result.images[0])
    sprite.save(output_path)


def make_img2img_pipeline(txt2img_pipe):
    from diffusers import StableDiffusionImg2ImgPipeline
    return StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)


def _device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _apply_lora(pipe) -> None:
    from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin

    path = str(_LORA_PATH)
    state_dict, network_alphas, metadata = StableDiffusionLoraLoaderMixin.lora_state_dict(
        path, return_lora_metadata=True
    )

    pipe.load_lora_into_unet(
        state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet,
        metadata=metadata,
        _pipeline=pipe,
    )

    # diffusers converts kohya TE keys to "text_encoder.text_model.encoder.*" but
    # the actual text encoder modules are named "encoder.*" (no text_model. wrapper),
    # so rank detection fails.  Strip the extra level before handing off.
    def _drop_text_model(d):
        if not d:
            return d
        old = "text_encoder.text_model."
        new = "text_encoder."
        return {new + k[len(old):] if k.startswith(old) else k: v for k, v in d.items()}

    pipe.load_lora_into_text_encoder(
        _drop_text_model(state_dict),
        network_alphas=_drop_text_model(network_alphas),
        text_encoder=pipe.text_encoder,
        lora_scale=pipe.lora_scale,
        metadata=metadata,
        _pipeline=pipe,
    )
    pipe.fuse_lora(lora_scale=_LORA_SCALE)


def _set_dpmpp_karras(pipe) -> None:
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )


def _load_base_pipeline(pipe_cls):
    device, dtype = _device_and_dtype()
    pipe = pipe_cls.from_pretrained(_BASE_MODEL_ID, torch_dtype=dtype, safety_checker=None)
    _apply_lora(pipe)
    _set_dpmpp_karras(pipe)
    return pipe.to(device)


def load_txt2img_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        return _load_base_pipeline(StableDiffusionPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)


def load_img2img_pipeline():
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        return _load_base_pipeline(StableDiffusionImg2ImgPipeline)
    except Exception as exc:
        print(f"Error: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)
