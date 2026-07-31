"""Sprite tests that run without the real ML stack installed.

Everything here either exercises pure-PIL/string code (build_prompt,
postprocess) or fakes torch/diffusers wholesale via sys.modules injection
(the load_* tests) — so this file runs in environments with only pytest,
Pillow, and mistralai, e.g. the keep sandbox. Tests that trigger real
`import torch` calls live in test_sprites_ml.py.
"""

import random
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from fakemon_forge.sprites import (
    postprocess,
    build_prompt,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    _BASE_MODEL_ID,
    _LORA_SCALE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(w=512, h=512, color=(200, 100, 50)):
    return Image.new("RGB", (w, h), color=color)


def _noisy_image(w=512, h=512):
    img = Image.new("RGB", (w, h))
    rng = random.Random(42)
    img.putdata([
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        for _ in range(w * h)
    ])
    return img


# ---------------------------------------------------------------------------
# build_prompt()
# ---------------------------------------------------------------------------

def test_build_prompt_no_types_prepends_only_style_tag():
    assert build_prompt("spiky wolf", []) == "gen3 spiky wolf"


def test_build_prompt_single_type_prepends_tag():
    result = build_prompt("fire lizard", ["Fire"])
    assert result.startswith("firetype")
    assert "fire lizard" in result


def test_build_prompt_two_types_prepends_both_tags():
    result = build_prompt("rock crab", ["Rock", "Water"])
    assert "rocktype" in result
    assert "watertype" in result
    assert "rock crab" in result


def test_build_prompt_unknown_type_is_skipped():
    result = build_prompt("mystery blob", ["Shadow"])
    assert "shadowtype" not in result
    assert result == "gen3 mystery blob"


# ---------------------------------------------------------------------------
# postprocess()
# ---------------------------------------------------------------------------

def test_postprocess_resizes_to_96x96():
    assert postprocess(_rgb_image()).size == (96, 96)


def test_postprocess_output_is_palette_mode():
    assert postprocess(_rgb_image()).mode == "P"


def test_postprocess_at_most_16_colors():
    assert len(set(postprocess(_noisy_image()).get_flattened_data())) <= 16


def test_postprocess_does_not_mutate_input():
    img = _rgb_image()
    original_size = img.size
    postprocess(img)
    assert img.size == original_size


# ---------------------------------------------------------------------------
# load_txt2img_pipeline()
# ---------------------------------------------------------------------------

def _make_lora_pipeline_mock():
    mock_mixin = MagicMock()
    mock_mixin.lora_state_dict.return_value = ({}, {}, None)
    mock_mod = MagicMock()
    mock_mod.StableDiffusionLoraLoaderMixin = mock_mixin
    return mock_mod


def _mock_modules(pipe_side_effect=None, cuda=False):
    mock_pipe_cls = MagicMock()
    if pipe_side_effect:
        mock_pipe_cls.from_pretrained.side_effect = pipe_side_effect
    else:
        mock_pipe_cls.from_pretrained.return_value = MagicMock()

    mock_diffusers = MagicMock()
    mock_diffusers.StableDiffusionPipeline = mock_pipe_cls

    mock_torch = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.cuda.is_available.return_value = cuda

    return {
        "diffusers": mock_diffusers,
        "torch": mock_torch,
        "diffusers.loaders": MagicMock(),
        "diffusers.loaders.lora_pipeline": _make_lora_pipeline_mock(),
    }, mock_pipe_cls


def test_load_returns_pipeline():
    mods, _ = _mock_modules()
    with patch.dict("sys.modules", mods):
        pipe = load_txt2img_pipeline()
    assert pipe is not None


def test_load_calls_from_pretrained_with_model_id():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    mock_pipe_cls.from_pretrained.assert_called_once()
    assert mock_pipe_cls.from_pretrained.call_args.args[0] == _BASE_MODEL_ID


def test_load_applies_lora_weights():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.load_lora_into_unet.assert_called_once()
    pipe.load_lora_into_text_encoder.assert_called_once()
    pipe.fuse_lora.assert_called_once_with(lora_scale=_LORA_SCALE)


def test_load_exits_on_oom(capsys):
    mods, _ = _mock_modules(pipe_side_effect=RuntimeError("CUDA out of memory"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit) as exc:
            load_txt2img_pipeline()
    assert exc.value.code == 1


def test_load_error_mentions_exception(capsys):
    mods, _ = _mock_modules(pipe_side_effect=RuntimeError("missing weights"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit):
            load_txt2img_pipeline()
    assert "missing weights" in capsys.readouterr().err


def test_load_uses_float16_when_cuda_available():
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    assert mock_pipe_cls.from_pretrained.call_args.kwargs["torch_dtype"] == "float16"


def test_load_uses_float32_when_no_cuda():
    mods, mock_pipe_cls = _mock_modules(cuda=False)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    assert mock_pipe_cls.from_pretrained.call_args.kwargs["torch_dtype"] == "float32"


def test_load_moves_pipeline_to_cuda_when_available():
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    mock_pipe_cls.from_pretrained.return_value.to.assert_called_once_with("cuda")


def test_load_moves_pipeline_to_cpu_when_no_cuda():
    mods, mock_pipe_cls = _mock_modules(cuda=False)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    mock_pipe_cls.from_pretrained.return_value.to.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# load_img2img_pipeline()
# ---------------------------------------------------------------------------

def _mock_img2img_modules(pipe_side_effect=None, cuda=False):
    mock_pipe_cls = MagicMock()
    if pipe_side_effect:
        mock_pipe_cls.from_pretrained.side_effect = pipe_side_effect
    else:
        mock_pipe_cls.from_pretrained.return_value = MagicMock()

    mock_diffusers = MagicMock()
    mock_diffusers.StableDiffusionImg2ImgPipeline = mock_pipe_cls

    mock_torch = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.cuda.is_available.return_value = cuda

    return {
        "diffusers": mock_diffusers,
        "torch": mock_torch,
        "diffusers.loaders": MagicMock(),
        "diffusers.loaders.lora_pipeline": _make_lora_pipeline_mock(),
    }, mock_pipe_cls


def test_load_img2img_returns_pipeline():
    mods, _ = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        pipe = load_img2img_pipeline()
    assert pipe is not None


def test_load_img2img_uses_correct_model_id():
    mods, mock_pipe_cls = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        load_img2img_pipeline()
    assert mock_pipe_cls.from_pretrained.call_args.args[0] == _BASE_MODEL_ID


def test_load_img2img_applies_lora_weights():
    mods, mock_pipe_cls = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        load_img2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.load_lora_into_unet.assert_called_once()
    pipe.load_lora_into_text_encoder.assert_called_once()
    pipe.fuse_lora.assert_called_once_with(lora_scale=_LORA_SCALE)


def test_load_img2img_exits_on_failure(capsys):
    mods, _ = _mock_img2img_modules(pipe_side_effect=RuntimeError("OOM"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit) as exc:
            load_img2img_pipeline()
    assert exc.value.code == 1


def test_load_img2img_error_mentions_exception(capsys):
    mods, _ = _mock_img2img_modules(pipe_side_effect=RuntimeError("missing weights"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit):
            load_img2img_pipeline()
    assert "missing weights" in capsys.readouterr().err
