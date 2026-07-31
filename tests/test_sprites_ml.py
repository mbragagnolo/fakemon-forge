"""Sprite tests that need the real ML stack importable.

generate_sprite() / generate_sprite_img2img() call _make_generator(), which
does a real `import torch` even when the pipeline itself is a mock — so these
tests require torch to be installed. They are marked `ml` and auto-skipped
(see conftest.py) in environments without torch, e.g. the keep sandbox.
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from fakemon_forge.sprites import (
    build_prompt,
    generate_sprite,
    generate_sprite_img2img,
    _NUM_STEPS,
    _CFG_SCALE,
)

pytestmark = pytest.mark.ml

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


# ---------------------------------------------------------------------------
# _encode_prompt() / prompt_embeds passthrough
# ---------------------------------------------------------------------------

def test_encode_prompt_called_with_built_prompt_and_pipeline(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds) as mock_enc:
        generate_sprite("spiky ice wolf", [], str(out), pipeline=pipe)
    mock_enc.assert_called_once_with(build_prompt("spiky ice wolf", []), pipe)


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
    mock_enc.assert_called_once_with(build_prompt("spiky ice wolf", []), pipe)


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
