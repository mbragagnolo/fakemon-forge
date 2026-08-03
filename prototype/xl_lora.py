# Shared kohya-SDXL-LoRA loader shim for diffusers 0.38 (see FINDINGS.md):
# unet_config triggers SGM block remapping; te1 needs the text_model. level
# stripped, te2 must keep it.
def apply_lora_xl(pipe, path: str, scale: float = 1.0) -> None:
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
    pipe.fuse_lora(lora_scale=scale)


NEG = "worst quality, low quality, blurry, watermark, signature, text, jpeg artifacts"
