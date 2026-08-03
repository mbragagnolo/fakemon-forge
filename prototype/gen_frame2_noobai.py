# Spike: frame-2 animation on the NoobAI backend.
# Question: does low-strength img2img on the NoobAI front half produce a
# candidate that the production build_frame2 ACCEPTS (difference ratio in
# [0.02, 0.30] after palette-lock + recenter), or does it always fall back to
# the procedural squash?
# Mirrors production generate_frame2: strength 0.35, "open mouth" tag, same seed.
import sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from fakemon_forge.sprites import (
    postprocess, quantize_to_reference, recenter_to_anchor, difference_ratio,
    build_frame2,
)
from spike_prompts import CREATURES, SEED

HERE = Path(__file__).parent
NB = HERE / "out" / "noobai"
OUT = HERE / "out" / "frame2"
OUT.mkdir(parents=True, exist_ok=True)
LORA = HERE / "models" / "pkspbf_nb_v1.safetensors"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", torch_dtype=torch.float16
)


# Same loader shim as gen_noobai.py (kohya SDXL LoRA vs diffusers 0.38).
def _apply_lora_xl(pipe, path: str) -> None:
    from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

    state_dict, network_alphas, metadata = StableDiffusionXLLoraLoaderMixin.lora_state_dict(
        path, return_lora_metadata=True, unet_config=pipe.unet.config
    )
    pipe.load_lora_into_unet(
        state_dict, network_alphas=network_alphas, unet=pipe.unet,
        metadata=metadata, _pipeline=pipe,
    )

    def _drop_text_model(d, prefix):
        if not d:
            return d
        old = f"{prefix}.text_model."
        new = f"{prefix}."
        return {new + k[len(old):] if k.startswith(old) else k: v for k, v in d.items()}

    for encoder, prefix, fix in ((pipe.text_encoder, "text_encoder", True),
                                 (pipe.text_encoder_2, "text_encoder_2", False)):
        sd = _drop_text_model(state_dict, prefix) if fix else state_dict
        al = _drop_text_model(network_alphas, prefix) if fix else network_alphas
        pipe.load_lora_into_text_encoder(
            sd, network_alphas=al, text_encoder=encoder, prefix=prefix,
            lora_scale=pipe.lora_scale, metadata=metadata, _pipeline=pipe,
        )
    pipe.fuse_lora(lora_scale=1.0)


_apply_lora_xl(pipe, str(LORA))
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

NEG = "worst quality, low quality, blurry, watermark, signature, text, jpeg artifacts"
STRENGTH = 0.35  # production generate_frame2 default

for name, types, desc in CREATURES:
    canvas = Image.open(NB / f"{name}_canvas.png")
    w, h = canvas.size
    front_raw = canvas.crop((0, 0, w // 2, h))          # 768x768 front half
    frame1 = postprocess(front_raw, size=768)           # production P-mode sprite
    frame1.save(OUT / f"{name}_frame1.png")

    init = front_raw.convert("RGB")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    candidate = pipe(
        prompt=f"gen3, {desc}, open mouth, white background",
        negative_prompt=NEG, image=init, strength=STRENGTH,
        num_inference_steps=28, guidance_scale=5.5,
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    dt = time.time() - t0

    # Recompute the acceptance inputs so the spike can report the ratio
    # (build_frame2 decides silently).
    locked = quantize_to_reference(candidate, frame1)
    recentred = recenter_to_anchor(locked, frame1)
    ratio = difference_ratio(recentred, frame1)
    accepted = 0.02 <= ratio <= 0.30

    frame2 = build_frame2(frame1, candidate)
    frame2.save(OUT / f"{name}_frame2.png")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name}: {dt:.0f}s, ratio {ratio:.3f}, "
          f"{'ACCEPTED' if accepted else 'fallback->squash'}, peak {peak:.2f} GiB",
          flush=True)

print("frame2 spike done")
