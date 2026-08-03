# Spike 2c: does the SDXL trainer-sprite LoRA (pk_trainer_xl_v1, trained on
# SDXL-base) hold up on the NoobAI base? 5-minute compat test, 2 trainers.
import sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from spike_prompts import SEED
from xl_lora import apply_lora_xl, NEG

HERE = Path(__file__).parent
OUT = HERE / "out" / "trainer"
OUT.mkdir(parents=True, exist_ok=True)

lora_path = hf_hub_download("sWizad/pokemon-trainer-sprite-pixelart", "pk_trainer_xl_v1.safetensors")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", torch_dtype=torch.float16
)
apply_lora_xl(pipe, lora_path)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

TRAINERS = [
    ("hiker", "a young hiker with a red jacket and a big backpack, simple background"),
    ("scientist", "a female scientist in a white lab coat holding a small ball, simple background"),
]

for name, prompt in TRAINERS:
    t0 = time.time()
    image = pipe(
        prompt=prompt, negative_prompt=NEG,
        width=768, height=768,
        num_inference_steps=28, guidance_scale=5.5,
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    image.save(OUT / f"{name}.png")
    print(f"{name}: {time.time()-t0:.0f}s", flush=True)

print("trainer spike done")
