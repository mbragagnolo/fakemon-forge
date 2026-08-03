"""Sprite tests that need the real ML stack importable.

generate_sprite() / generate_sprite_img2img() call _make_generator(), which
does a real `import torch` even when the pipeline itself is a mock — so these
tests require torch to be installed. They are marked `ml` and auto-skipped
(see conftest.py) in environments without torch, e.g. the keep sandbox.
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from fakemon_forge.sprites import (
    build_prompt,
    generate_sprite,
    generate_sprite_pair,
    generate_sprite_img2img,
    generate_frame2,
    postprocess,
    procedural_squash,
    _content_bbox,
    _NUM_STEPS,
    _CFG_SCALE,
    _NEGATIVE_PROMPT,
)

pytestmark = pytest.mark.ml

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


def _fake_pair_pipeline(*images):
    """A MagicMock pipeline returning each ``images`` entry across successive calls."""
    pipe = MagicMock()
    results = []
    for image in images:
        result = MagicMock()
        result.images = [image]
        results.append(result)
    pipe.side_effect = results
    return pipe


# A 200x100 background canvas with (x0, x1, colour) bands spanning rows 1-98
# (mirroring test_sprites.py's `_split_canvas`), used to drive
# `split_front_back_canvas` inside `generate_sprite_pair`.
_PAIR_BG = (250, 250, 250)
_PAIR_FRONT_COLOR = (30, 60, 90)
_PAIR_BACK_COLOR = (90, 30, 60)


def _pair_canvas(*bands):
    canvas = Image.new("RGB", (200, 100), _PAIR_BG)
    d = ImageDraw.Draw(canvas)
    for x0, x1, color in bands:
        d.rectangle((x0, 1, x1, 98), fill=color)
    return canvas


def _clean_split_canvas():
    # Front square at columns 30-94, back square at columns 105-170: the only
    # full-height background run in the [80, 120) search window is 95-104.
    return _pair_canvas((30, 94, _PAIR_FRONT_COLOR), (105, 170, _PAIR_BACK_COLOR))


def _no_split_canvas(color):
    # A single band spanning the whole [80, 120) search window: no column in
    # it is background for its full height, so no split is found.
    return _pair_canvas((70, 130, color))


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
# prompt= / negative_prompt= passthrough
# ---------------------------------------------------------------------------

def test_pipeline_called_with_prompt_string(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("spiky ice wolf", [], str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt"] == build_prompt("spiky ice wolf")
    assert "prompt_embeds" not in pipe.call_args.kwargs


def test_pipeline_called_with_negative_prompt(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["negative_prompt"] == _NEGATIVE_PROMPT


def test_types_are_not_mechanically_tagged_into_the_prompt(tmp_path):
    # ``types`` is still accepted (main.py passes stage["types"]) but the SD1.5
    # LoRA's "firetype" trigger vocabulary is gone with the backend that trained
    # it. This asserts the absence of the mechanical tags only — the type signal
    # itself did not disappear, it moved into the LLM-authored sprite_prompt,
    # required by the sprite_prompt spec in generator.py and pinned by
    # test_system_prompt_requires_sprite_prompt_to_show_the_types.
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", ["Fire", "Flying"], str(out), pipeline=pipe)
    prompt = pipe.call_args.kwargs["prompt"]
    assert prompt == build_prompt("fire lizard")
    assert "firetype" not in prompt and "flyingtype" not in prompt


def test_extra_tags_included_in_prompt(tmp_path):
    pipe = _fake_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite("fire lizard", [], str(out), pipeline=pipe, extra_tags=["chibi"])
    assert "chibi" in pipe.call_args.kwargs["prompt"]


def test_img2img_pipeline_called_with_prompt_string(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("spiky ice wolf", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["prompt"] == build_prompt("spiky ice wolf")
    assert "prompt_embeds" not in pipe.call_args.kwargs


def test_img2img_pipeline_called_with_negative_prompt(tmp_path):
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image())
    out = tmp_path / "sprite.png"
    generate_sprite_img2img("fire lizard", [], str(init_img), str(out), pipeline=pipe)
    assert pipe.call_args.kwargs["negative_prompt"] == _NEGATIVE_PROMPT


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
# generate_sprite_pair()
# ---------------------------------------------------------------------------

def test_pair_pipeline_called_with_1536x768(tmp_path):
    pipe = _fake_pair_pipeline(_clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    kwargs = pipe.call_args.kwargs
    assert kwargs["width"] == 1536
    assert kwargs["height"] == 768


def test_pair_pipeline_called_with_prompt_and_negative_prompt(tmp_path):
    pipe = _fake_pair_pipeline(_clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    kwargs = pipe.call_args.kwargs
    assert kwargs["prompt"] == build_prompt("fire lizard")
    assert kwargs["negative_prompt"] == _NEGATIVE_PROMPT
    assert kwargs["num_inference_steps"] == _NUM_STEPS
    assert kwargs["guidance_scale"] == _CFG_SCALE


def test_pair_happy_path_calls_pipeline_exactly_once(tmp_path):
    pipe = _fake_pair_pipeline(_clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    assert pipe.call_count == 1


def test_pair_front_output_is_native_768_p_mode(tmp_path):
    pipe = _fake_pair_pipeline(_clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    saved = Image.open(front)
    assert saved.mode == "P"
    assert saved.size == (768, 768)


def test_pair_back_output_shares_front_exact_palette(tmp_path):
    pipe = _fake_pair_pipeline(_clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    assert back.exists()
    saved_back = Image.open(back)
    assert saved_back.mode == "P"
    assert saved_back.getpalette() == Image.open(front).getpalette()


def test_pair_reroll_when_first_canvas_has_no_clean_split(tmp_path, capsys):
    pipe = _fake_pair_pipeline(_no_split_canvas(_PAIR_FRONT_COLOR), _clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe, seed=5)
    assert pipe.call_count == 2
    # a reroll that finds a clean split is the documented happy path, not a
    # degradation -- no warning.
    assert capsys.readouterr().err == ""
    assert back.exists()


def test_pair_reroll_uses_seed_plus_one(tmp_path):
    pipe = _fake_pair_pipeline(_no_split_canvas(_PAIR_FRONT_COLOR), _clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    with patch("fakemon_forge.sprites._make_generator", wraps=lambda seed: MagicMock()) as m_gen:
        generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe, seed=5)
    assert m_gen.call_args_list[0].args[0] == 5
    assert m_gen.call_args_list[1].args[0] == 6


def test_pair_unseeded_reroll_stays_unseeded(tmp_path):
    pipe = _fake_pair_pipeline(_no_split_canvas(_PAIR_FRONT_COLOR), _clean_split_canvas())
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    with patch("fakemon_forge.sprites._make_generator", wraps=lambda seed: MagicMock()) as m_gen:
        generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    assert m_gen.call_args_list[0].args[0] is None
    assert m_gen.call_args_list[1].args[0] is None


def test_pair_never_calls_pipeline_a_third_time_when_both_canvases_fail_to_split(tmp_path, capsys):
    pipe = _fake_pair_pipeline(
        _no_split_canvas(_PAIR_FRONT_COLOR), _no_split_canvas(_PAIR_BACK_COLOR)
    )
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    # A third pipeline() call would raise StopIteration (side_effect exhausted).
    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)
    assert pipe.call_count == 2
    assert capsys.readouterr().err   # naive-midline-fallback warning


def test_pair_pipeline_error_propagates(tmp_path):
    pipe = MagicMock(side_effect=RuntimeError("inference crash"))
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"
    with pytest.raises(RuntimeError):
        generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)


def test_pair_empty_back_half_is_skipped_and_warns_but_still_saves_the_front(tmp_path, capsys):
    """A canvas holding only a front sprite splits "successfully" into a good
    front and an empty back. The back is skipped with a warning rather than
    saved blank — and that degradation must not cost the caller the front."""
    pipe = _fake_pair_pipeline(_pair_canvas((30, 94, _PAIR_FRONT_COLOR)))
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"

    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)

    assert front.exists()
    assert not back.exists()
    assert "empty/background-only" in capsys.readouterr().err


def test_pair_offcentre_split_gives_both_views_the_same_scale(tmp_path):
    """Regression: an off-centre cut must not scale the two views differently.

    Front body at columns 20-59 and back body at 100-139 are both 40x40, and
    the gap at 80-99 cuts the canvas into 90px and 110px halves. Squaring those
    by resize would blow the front's body up and shrink the back's; squaring by
    paste leaves both bodies identical, as drawn.
    """
    canvas = Image.new("RGB", (200, 100), _PAIR_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle((20, 30, 59, 69), fill=_PAIR_FRONT_COLOR)
    d.rectangle((100, 30, 139, 69), fill=_PAIR_BACK_COLOR)
    pipe = _fake_pair_pipeline(canvas)
    front, back = tmp_path / "sprite.png", tmp_path / "sprite_back.png"

    generate_sprite_pair("fire lizard", [], str(front), str(back), pipeline=pipe)

    saved_front, saved_back = Image.open(front), Image.open(back)
    assert saved_front.size == saved_back.size == (768, 768)
    front_box = _content_bbox(saved_front, background=0)
    back_box = _content_bbox(saved_back, background=0)
    assert front_box is not None and back_box is not None
    # Same drawn body, same split, same framing -> same box, within the
    # rounding the 200->768 upscale of a 40px body can introduce.
    for got, want in zip(back_box, front_box):
        assert abs(got - want) <= 4


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
# generate_sprite_img2img(reference_path=...) — shared-palette lock
#
# The back-sprite chain that used to be this parameter's only caller is gone
# (issue #66); the parameter stays public API, so its behaviour stays covered.
# ---------------------------------------------------------------------------

def test_img2img_reference_path_adopts_reference_palette(tmp_path):
    """With reference_path set, the saved sprite locks to that palette
    (proves it adopts the shared palette instead of an adaptive one)."""
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_locked.png"
    generate_sprite_img2img(
        "fire lizard", [], str(init_img), str(out), pipeline=pipe,
        extra_tags=["chibi"], reference_path=str(ref),
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
    invoked exactly once with the unchanged strength / image / prompt."""
    ref = _frame1_file(tmp_path)
    init_img = tmp_path / "drawing.png"
    _rgb_image(100, 100).save(str(init_img))
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_back.png"
    generate_sprite_img2img(
        "fire lizard", [], str(init_img), str(out), pipeline=pipe,
        extra_tags=["chibi"], strength=0.65, reference_path=str(ref),
    )
    assert pipe.call_count == 1
    kwargs = pipe.call_args.kwargs
    assert kwargs["strength"] == 0.65
    assert kwargs["image"].size == (768, 768)
    assert kwargs["prompt"] == build_prompt("fire lizard", ["chibi"])


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
    assert pipe.call_args.kwargs["strength"] == 0.30


def test_frame2_pipeline_init_image_is_squashed_frame1_not_raw_front(tmp_path):
    """Regression (issue #67): the img2img init image must be
    ``procedural_squash(frame1)`` so there's real structural signal to clean
    up, not the raw front sprite (which reads as colour jitter)."""
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)

    frame1 = Image.open(str(front))
    expected_init = procedural_squash(frame1).convert("RGB").resize((768, 768), Image.LANCZOS)
    raw_init = frame1.convert("RGB").resize((768, 768), Image.LANCZOS)

    actual_init = pipe.call_args.kwargs["image"]
    assert actual_init.get_flattened_data() == expected_init.get_flattened_data()
    assert actual_init.get_flattened_data() != raw_init.get_flattened_data()


def test_frame2_default_extra_tags_include_open_mouth(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2("fire lizard", [], str(front), str(out), pipeline=pipe)
    assert "open mouth" in pipe.call_args.kwargs["prompt"]


def test_frame2_honours_caller_supplied_extra_tags(tmp_path):
    front = _frame1_file(tmp_path)
    pipe = _fake_img2img_pipeline(_rgb_image(96, 96, color=(90, 160, 210)))
    out = tmp_path / "sprite_frame2.png"
    generate_frame2(
        "fire lizard", [], str(front), str(out), pipeline=pipe,
        extra_tags=["closed eyes"],
    )
    prompt = pipe.call_args.kwargs["prompt"]
    assert "closed eyes" in prompt
    assert "open mouth" not in prompt


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
