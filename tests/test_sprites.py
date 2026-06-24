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
    load_txt2img_pipeline,
    load_img2img_pipeline,
    _clip_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pipeline(image: Image.Image):
    """Return a callable mock that behaves like a diffusers pipeline."""
    result = MagicMock()
    result.images = [image]
    pipe = MagicMock()
    pipe.return_value = result
    return pipe


def _rgb_image(w=512, h=512, color=(200, 100, 50)):
    return Image.new("RGB", (w, h), color=color)


def _noisy_image(w=512, h=512):
    """512x512 image with many random colors to stress-test quantization."""
    img = Image.new("RGB", (w, h))
    rng = random.Random(42)
    img.putdata([
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        for _ in range(w * h)
    ])
    return img


# ---------------------------------------------------------------------------
# _clip_prompt()
# ---------------------------------------------------------------------------

def test_clip_prompt_short_prompt_unchanged():
    short = "a fire lizard with three tails"
    assert _clip_prompt(short) == short


def test_clip_prompt_truncates_long_prompt():
    long = " ".join(["word"] * 80)
    result = _clip_prompt(long)
    assert len(result.split()) == 50


def test_clip_prompt_exactly_50_words_unchanged():
    exact = " ".join(["word"] * 50)
    assert _clip_prompt(exact) == exact


def test_generate_sprite_prompt_is_clipped(tmp_path):
    long_prompt = " ".join(["fire"] * 80)
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite(long_prompt, str(out), pipeline=pipe)
    sent_prompt = pipe.call_args.kwargs["prompt"]
    assert len(sent_prompt.split()) <= 60


# ---------------------------------------------------------------------------
# postprocess()
# ---------------------------------------------------------------------------

def test_postprocess_resizes_to_96x96():
    result = postprocess(_rgb_image())
    assert result.size == (96, 96)


def test_postprocess_output_is_palette_mode():
    result = postprocess(_rgb_image())
    assert result.mode == "P"


def test_postprocess_at_most_16_colors():
    result = postprocess(_noisy_image())
    assert len(set(result.get_flattened_data())) <= 16


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
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    assert out.exists()


def test_saved_sprite_is_96x96(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    assert Image.open(out).size == (96, 96)


def test_saved_sprite_is_png(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    assert Image.open(out).format == "PNG"


def test_saved_sprite_has_palette_mode(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    assert Image.open(out).mode == "P"


def test_pipeline_called_with_prompt(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("spiky ice wolf", str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt"] == "spiky ice wolf"


def test_pipeline_called_with_512x512(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    kwargs = pipe.call_args.kwargs
    assert kwargs["width"] == 512
    assert kwargs["height"] == 512


def test_pipeline_called_exactly_once(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", str(out), pipeline=pipe)
    assert pipe.call_count == 1


# ---------------------------------------------------------------------------
# load_txt2img_pipeline()
# ---------------------------------------------------------------------------

def _mock_modules(pipe_side_effect=None, cuda=False):
    """Return patched sys.modules with fake diffusers + torch."""
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

    return {"diffusers": mock_diffusers, "torch": mock_torch}, mock_pipe_cls


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
    assert mock_pipe_cls.from_pretrained.call_args.args[0] == "lambdalabs/sd-pokemon-diffusers"


def test_load_exits_on_oom(capsys):
    mods, _ = _mock_modules(pipe_side_effect=RuntimeError("CUDA out of memory"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit) as exc:
            load_txt2img_pipeline()
    assert exc.value.code == 1


def test_load_error_mentions_model_name(capsys):
    mods, _ = _mock_modules(pipe_side_effect=RuntimeError("OOM"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit):
            load_txt2img_pipeline()
    assert "lambdalabs/sd-pokemon-diffusers" in capsys.readouterr().err


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

def _fake_img2img_pipeline(image: Image.Image):
    result = MagicMock()
    result.images = [image]
    pipe = MagicMock()
    pipe.return_value = result
    return pipe


def test_img2img_creates_file(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    assert out.exists()


def test_img2img_saved_sprite_is_96x96(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).size == (96, 96)


def test_img2img_saved_sprite_is_png(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).format == "PNG"


def test_img2img_saved_sprite_has_palette_mode(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).mode == "P"


def test_img2img_pipeline_called_with_prompt(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("spiky ice wolf", str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt"] == "spiky ice wolf"


def test_img2img_conditioning_image_passed_to_pipeline(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    assert "image" in pipe.call_args.kwargs


def test_img2img_conditioning_image_is_512x512(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))  # deliberately not 512x512
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    passed_image = pipe.call_args.kwargs["image"]
    assert passed_image.size == (512, 512)


def test_img2img_conditioning_image_is_rgb(tmp_path):
    init_img = tmp_path / "drawing.png"
    Image.new("RGBA", (100, 100)).save(str(init_img))  # RGBA input
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
    passed_image = pipe.call_args.kwargs["image"]
    assert passed_image.mode == "RGB"


def test_img2img_pipeline_called_exactly_once(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", str(init_img), str(out), pipeline=pipe)
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

    return {"diffusers": mock_diffusers, "torch": mock_torch}, mock_pipe_cls


def test_load_img2img_returns_pipeline():
    mods, _ = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        pipe = load_img2img_pipeline()
    assert pipe is not None


def test_load_img2img_uses_correct_model_id():
    mods, mock_pipe_cls = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        load_img2img_pipeline()
    assert mock_pipe_cls.from_pretrained.call_args.args[0] == "lambdalabs/sd-pokemon-diffusers"


def test_load_img2img_exits_on_failure(capsys):
    mods, _ = _mock_img2img_modules(pipe_side_effect=RuntimeError("OOM"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit) as exc:
            load_img2img_pipeline()
    assert exc.value.code == 1


def test_load_img2img_error_mentions_model_name(capsys):
    mods, _ = _mock_img2img_modules(pipe_side_effect=RuntimeError("OOM"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit):
            load_img2img_pipeline()
    assert "lambdalabs/sd-pokemon-diffusers" in capsys.readouterr().err


def test_load_img2img_error_mentions_exception(capsys):
    mods, _ = _mock_img2img_modules(pipe_side_effect=RuntimeError("missing weights"))
    with patch.dict("sys.modules", mods):
        with pytest.raises(SystemExit):
            load_img2img_pipeline()
    assert "missing weights" in capsys.readouterr().err
