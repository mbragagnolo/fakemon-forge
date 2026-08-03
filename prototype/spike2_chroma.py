# Spike 2a: does the back&front LoRA honor a non-white (chroma) background?
# If yes, keying on green kills the whites-eaten-by-key-filler bug outright.
# Test the creature with white detail (knightcoral) + a control (bloomling).
import sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

sys.path.insert(0, str(Path(__file__).parent))
from spike_prompts import CREATURES, SEED
from xl_lora import apply_lora_xl, NEG

HERE = Path(__file__).parent
OUT = HERE / "out" / "chroma"
OUT.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", torch_dtype=torch.float16
)
apply_lora_xl(pipe, str(HERE / "models" / "pkspbf_nb_v1.safetensors"))
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

TARGETS = [c for c in CREATURES if c[0] in ("knightcoral", "bloomling")]
BGS = ["green screen background", "simple magenta background"]

for name, types, desc in TARGETS:
    for bg in BGS:
        tag = bg.split()[1] if bg.startswith("simple") else "green"
        t0 = time.time()
        image = pipe(
            prompt=f"gen3, {desc}, {bg}",
            negative_prompt=NEG,
            width=1536, height=768,
            num_inference_steps=28, guidance_scale=5.5,
            generator=torch.Generator("cuda").manual_seed(SEED),
        ).images[0]
        image.save(OUT / f"{name}_{tag}.png")
        print(f"{name} {tag}: {time.time()-t0:.0f}s", flush=True)

print("chroma spike done")
