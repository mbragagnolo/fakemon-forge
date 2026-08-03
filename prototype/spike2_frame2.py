# Spike 2b: can frame 2 show an ACTUAL POSE CHANGE (not just color jitter)?
# Variants per creature:
#   ladder50  - strength 0.50, "open mouth" (current tag, more freedom)
#   posetags  - strength 0.40, explicit pose tags
#   squashinit- init = procedural_squash(frame1), strength 0.30 (pose change
#               guaranteed by construction; model just cleans it up)
# Metrics: difference_ratio (production band [0.02,0.30]) + mask_shift =
# fraction of pixels whose background/creature classification flips — color
# jitter scores ~0 on mask_shift, real pose change scores > 0.
import sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from fakemon_forge.sprites import (
    postprocess, quantize_to_reference, recenter_to_anchor, difference_ratio,
    procedural_squash, _background_index,
)
from spike_prompts import CREATURES, SEED
from xl_lora import apply_lora_xl, NEG

HERE = Path(__file__).parent
NB = HERE / "out" / "noobai"
OUT = HERE / "out" / "frame2b"
OUT.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", torch_dtype=torch.float16
)
apply_lora_xl(pipe, str(HERE / "models" / "pkspbf_nb_v1.safetensors"))
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


def mask_shift(a: Image.Image, b: Image.Image) -> float:
    """Fraction of pixels whose creature/background classification differs."""
    bg_a, bg_b = _background_index(a), _background_index(b)
    ma = a.point(lambda p: 255 if p != bg_a else 0).convert("1")
    mb = b.point(lambda p: 255 if p != bg_b else 0).convert("1")
    da, db = ma.get_flattened_data(), mb.get_flattened_data()
    return sum(1 for x, y in zip(da, db) if x != y) / len(da)


def run(name, desc, tag, init, strength, prompt_extra):
    t0 = time.time()
    candidate = pipe(
        prompt=f"gen3, {desc}, {prompt_extra}, white background",
        negative_prompt=NEG, image=init, strength=strength,
        num_inference_steps=28, guidance_scale=5.5,
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    locked = quantize_to_reference(candidate, frame1)
    recentred = recenter_to_anchor(locked, frame1)
    ratio = difference_ratio(recentred, frame1)
    shift = mask_shift(recentred, frame1)
    verdict = "IN-BAND" if 0.02 <= ratio <= 0.30 else "out-of-band"
    recentred.save(OUT / f"{name}_{tag}.png")
    print(f"{name} {tag}: {time.time()-t0:.0f}s ratio {ratio:.3f} "
          f"mask_shift {shift:.4f} {verdict}", flush=True)


TARGETS = [c for c in CREATURES if c[0] in ("knightcoral", "emberfox", "murkfin")]

for name, types, desc in TARGETS:
    canvas = Image.open(NB / f"{name}_canvas.png")
    w, h = canvas.size
    front_raw = canvas.crop((0, 0, w // 2, h))
    frame1 = postprocess(front_raw, size=768)
    frame1.save(OUT / f"{name}_frame1.png")
    # Baseline for the metric: round-1 result (strength .35, open mouth)
    run(name, desc, "baseline35", front_raw.convert("RGB"), 0.35, "open mouth")
    run(name, desc, "ladder50", front_raw.convert("RGB"), 0.50, "open mouth")
    run(name, desc, "posetags", front_raw.convert("RGB"), 0.40,
        "crouching, head lowered, mouth wide open")
    squash = procedural_squash(frame1)
    run(name, desc, "squashinit", squash.convert("RGB"), 0.30, "open mouth")

print("frame2 pose spike done")
