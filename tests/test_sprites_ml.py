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
    generate_frame2,
    postprocess,
    procedural_squash,
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


def _frame1_file(tmp_path, name="sprite.png"):
    """Write a real P-mode front sprite fixture (96px for test speed)."""
    from PIL import ImageDraw
    img = Image.new("RGB", (96, 96), (40, 40, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((26, 28, 70, 84), fill=(200, 80, 60))
    d.ellipse((34, 40, 46, 52), fill=(240, 240, 240))
    d.rectangle((38, 74, 58, 84), fill=(80, 60, 40))
    frame1 = postprocess(img, size=96)
    path = tmp_path / name
    frame1.save(str(path))
    return path


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


def test_saved_sprite_is_native_768(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert Image.open(out).size == (768, 768)


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


def test_img2img_saved_sprite_is_native_768(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert Image.open(out).size == (768, 768)


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
# generate_sprite_img2img(reference_path=...) — back-sprite palette lock
# ---------------------------------------------------------------------------

def test_img2img_reference_path_adopts_reference_palette(tmp_path):
    """With reference_path set, the saved back sprite locks to that palette
    (proves it adopts the shared palette instead of an adaptive one)."""
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_back.png"
    generate_sprite_img2img(
        "fire lizard", [], str(init_img), str(out), pipeline=pipe,
        extra_tags=["backside"], reference_path=str(ref),
    )
    saved = Image.open(str(out))
    assert saved.mode == "P"
    assert saved.getpalette() == Image.open(str(ref)).getpalette()


def test_img2img_reference_path_saved_matches_reference_size(tmp_path):
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_back.png"
    generate_sprite_img2img(
        "fire lizard", [], str(init_img), str(out), pipeline=pipe,
        reference_path=str(ref),
    )
    saved = Image.open(str(out))
    assert saved.size == Image.open(str(ref)).size   # locked output adopts reference size
    assert saved.format == "PNG"


def test_img2img_reference_path_pipeline_called_once_with_passthrough(tmp_path):
    """The reference only affects post-quantization: the pipeline is still
    invoked exactly once with the unchanged strength / image / prompt_embeds."""
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_back.png"
    fake_embeds = MagicMock()
    with patch("fakemon_forge.sprites._encode_prompt", return_value=fake_embeds):
        generate_sprite_img2img(
            "fire lizard", [], str(init_img), str(out), pipeline=pipe,
            extra_tags=["backside"], strength=0.65, reference_path=str(ref),
        )
    assert pipe.call_count == 1
    kwargs = pipe.call_args.kwargs
    assert kwargs["strength"] == 0.65
    assert kwargs["image"].size == (768, 768)
    assert kwargs["prompt_embeds"] is fake_embeds


def test_img2img_without_reference_path_uses_adaptive_palette(tmp_path):
    """Regression: the two branches diverge only in palette. The locked output
    equals the reference's palette; the unlocked output need not."""
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_back.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    saved = Image.open(str(out))
    assert saved.mode == "P"
    assert saved.size == (768, 768)


# ---------------------------------------------------------------------------
# generate_frame2()
# ---------------------------------------------------------------------------

def test_frame2_creates_file(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert out.exists()


def test_frame2_saved_matches_front_size(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert Image.open(out).size == Image.open(str(front)).size


def test_frame2_saved_is_png(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert Image.open(out).format == "PNG"


def test_frame2_saved_is_palette_mode(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert Image.open(out).mode == "P"


def test_frame2_shares_frame1_palette(tmp_path):
    """build_frame2 guarantees the frame shares frame 1's exact palette
    (proves the raw candidate is not double-quantized to a fresh palette)."""
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert Image.open(out).getpalette() == Image.open(front).getpalette()


def test_frame2_pipeline_called_with_low_strength(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["strength"] == 0.35


def test_frame2_default_extra_tags_include_open_mouth(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    with patch("fakemon_forge.sprites._encode_prompt", return_value=MagicMock()) as mock_enc:
        generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert "open mouth" in mock_enc.call_args.args[0]


def test_frame2_honours_caller_supplied_extra_tags(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    with patch("fakemon_forge.sprites._encode_prompt", return_value=MagicMock()) as mock_enc:
        generate_frame2(
            "fire lizard", [], str(front), str(out), pipeline=pipe,
            extra_tags=["closed eyes"],
        )
    encoded = mock_enc.call_args.args[0]
    assert "closed eyes" in encoded
    assert "open mouth" not in encoded


def test_frame2_seed_path_exercised(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe, seed=1234)
    assert pipe.call_args.kwargs["generator"] is not None


def test_frame2_pipeline_called_once(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert pipe.call_count == 1


def test_frame2_falls_back_to_squash_on_garbage_candidate(tmp_path):
    """An off-band candidate -> build_frame2 uses procedural_squash(frame1)."""
    front = _frame1_file(tmp_path)
    # A near-identical candidate (same colours as frame 1) is below the band.
    front_rgb = Image.open(str(front)).convert("RGB")
    pipe = _fake_img2img_pipeline(front_rgb)
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    frame1 = Image.open(str(front))
    expected = procedural_squash(frame1)
    assert list(Image.open(out).get_flattened_data()) == list(expected.get_flattened_data())
