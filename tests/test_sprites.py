import random
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from fakemon_forge.sprites import (
    generate_sprite,
    generate_sprite_img2img,
    postprocess,
    build_prompt,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    _encode_prompt,
    _BASE_MODEL_ID,
    _LORA_PATH,
    _NUM_STEPS,
    _CFG_SCALE,
    _LORA_SCALE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_encode_prompt():
    """Patch _encode_prompt for all sprite tests so compel isn't required."""
    with patch("fakemon_forge.sprites._encode_prompt", return_value=MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pipeline(image: Image.Image):
    result = MagicMock()
    result.images = [image]
    pipe = MagicMock()
    pipe.return_value = result
    return pipe


def _fake_img2img_pipeline(image: Image.Image):
    result = MagicMock()
    result.images = [image]
    pipe = MagicMock()
    pipe.return_value = result
    return pipe


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

def test_build_prompt_no_types_returns_prompt_unchanged():
    assert build_prompt("spiky wolf", []) == "spiky wolf"


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
    assert result == "mystery blob"


# ---------------------------------------------------------------------------
# _encode_prompt() / prompt_embeds passthrough
# ---------------------------------------------------------------------------

def test_encode_prompt_called_with_built_prompt_and_pipeline(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds) as mock_enc:
        generate_sprite("spiky ice wolf", [], str(out), pipeline=pipe)
    mock_enc.assert_called_once_with("spiky ice wolf", pipe)


def test_encode_prompt_result_passed_as_prompt_embeds(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds):
        generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt_embeds"] is fake_embeds
    assert "prompt" not in pipe.call_args.kwargs


def test_type_tags_included_in_encoded_prompt(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    with patch("fakemon_forge.sprites._encode_prompt", return_value=MagicMock()) as mock_enc:
        generate_sprite("fire lizard", ["Fire", "Flying"], str(out), pipeline=pipe)
    encoded_prompt = mock_enc.call_args.args[0]
    assert "firetype" in encoded_prompt
    assert "flyingtype" in encoded_prompt


def test_img2img_encode_prompt_called(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds) as mock_enc:
        generate_sprite_img2img("spiky ice wolf", [], str(init_img), str(out), pipeline=pipe)
    mock_enc.assert_called_once_with("spiky ice wolf", pipe)


def test_img2img_encode_prompt_result_passed_as_prompt_embeds(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds):
        generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt_embeds"] is fake_embeds
    assert "prompt" not in pipe.call_args.kwargs


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
# generate_sprite()
# ---------------------------------------------------------------------------

def test_generate_sprite_creates_file(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert out.exists()


def test_saved_sprite_is_96x96(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert Image.open(out).size == (96, 96)


def test_saved_sprite_is_png(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert Image.open(out).format == "PNG"


def test_saved_sprite_has_palette_mode(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert Image.open(out).mode == "P"


def test_pipeline_called_with_768x768(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    kwargs = pipe.call_args.kwargs
    assert kwargs["width"] == 768
    assert kwargs["height"] == 768


def test_pipeline_called_with_num_inference_steps(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["num_inference_steps"] == _NUM_STEPS


def test_pipeline_called_with_guidance_scale(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["guidance_scale"] == _CFG_SCALE


def test_pipeline_called_exactly_once(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert pipe.call_count == 1


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
# generate_sprite_img2img()
# ---------------------------------------------------------------------------

def test_img2img_creates_file(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert out.exists()


def test_img2img_saved_sprite_is_96x96(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).size == (96, 96)


def test_img2img_saved_sprite_is_png(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).format == "PNG"


def test_img2img_saved_sprite_has_palette_mode(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).mode == "P"


def test_img2img_conditioning_image_passed_to_pipeline(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert "image" in pipe.call_args.kwargs


def test_img2img_conditioning_image_is_768x768(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["image"].size == (768, 768)


def test_img2img_conditioning_image_is_rgb(tmp_path):
    init_img = tmp_path / "drawing.png"
    Image.new("RGBA", (100, 100)).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["image"].mode == "RGB"


def test_img2img_pipeline_called_exactly_once(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_count == 1


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
