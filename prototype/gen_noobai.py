# Spike: generate the 5 test creatures with NoobAI-XL 1.1 + Pokemon Sprite XL
# PixelArt LoRA (back&front variant): one 768x1536 canvas = front + back sprite.
# Output: prototype/out/noobai/<name>_canvas.png (raw RGB 768x1536)
# Records wall time + peak VRAM — 8GB feasibility is part of the spike question.
import sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

sys.path.insert(0, str(Path(__file__).parent))
from spike_prompts import CREATURES, SEED

OUT = Path(__file__).parent / "out" / "noobai"
OUT.mkdir(parents=True, exist_ok=True)
LORA = Path(__file__).parent / "models" / "pkspbf_nb_v1.safetensors"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Laxhar/noobai-XL-1.1", torch_dtype=torch.float16
)


# Same bug + fix as fakemon_forge.sprites._apply_lora, extended to SDXL's two
# text encoders: diffusers converts kohya TE keys to
# "text_encoder{,_2}.text_model.encoder.*" but the module names lack the
# "text_model." level, so rank detection finds nothing and load_lora_weights
# dies with IndexError. Strip the extra level per encoder before handing off.
def _apply_lora_xl(pipe, path: str) -> None:
    from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin

    # unet_config is what triggers the kohya SGM->diffusers block remapping;
    # load_lora_weights passes it internally, a direct call must too.
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

    # te1 (CLIPTextModel) names modules WITHOUT the text_model. wrapper — needs
    # the drop; te2 (CLIPTextModelWithProjection) names them WITH it — keys must
    # stay untouched. Verified empirically against named_modules() of each.
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
pipe.enable_vae_tiling()

NEG = "worst quality, low quality, blurry, watermark, signature, text, jpeg artifacts"

for name, types, desc in CREATURES:
    prompt = f"gen3, {desc}, white background"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    image = pipe(
        prompt=prompt, negative_prompt=NEG,
        width=1536, height=768,  # side-by-side pair — Civitai samples are landscape
        num_inference_steps=28, guidance_scale=5.5,
        generator=torch.Generator("cuda").manual_seed(SEED),
    ).images[0]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    image.save(OUT / f"{name}_canvas.png")
    print(f"{name}: {dt:.0f}s, peak VRAM {peak:.2f} GiB", flush=True)

print("noobai stack done")
