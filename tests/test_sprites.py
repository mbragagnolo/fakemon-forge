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
from PIL import Image, ImageDraw

from fakemon_forge.sprites import (
    postprocess,
    quantize_to_reference,
    generate_shiny,
    build_prompt,
    procedural_squash,
    recenter_to_anchor,
    difference_ratio,
    build_frame2,
    stitch_spritesheet,
    k_centroid,
    _background_index,
    _flatten_background_to_key,
    _quantize_gen3,
    _rgb_distance,
    _KEY_COLOR,
    _MAX_CREATURE_COLORS,
    _KEY_COLLISION_DISTANCE,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    _BASE_MODEL_ID,
    _LORA_SCALE,
)


def _pp96(img):
    """postprocess at 96px: keeps bulk tests fast; the 768 default has its own test."""
    return postprocess(img, size=96)


def _qg96(img):
    return _quantize_gen3(img, size=96)


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


def _sprite_rgb(body=(200, 80, 60)):
    """A 96x96 creature-ish RGB: a large background, a body, eyes, and feet.

    Mostly background so a small squash lands inside the acceptance band, with
    enough internal variation that squashing changes a moderate pixel count.
    """
    img = Image.new("RGB", (96, 96), (40, 40, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((26, 28, 70, 84), fill=body)
    # Horizontal shading bands give the body vertical texture, so a small
    # vertical squash shifts real detail (as on a shaded Gen-3 sprite).
    highlight = tuple(min(c + 40, 255) for c in body)
    for y in range(30, 84, 4):
        d.rectangle((28, y, 68, y + 1), fill=highlight)
    d.ellipse((34, 40, 46, 52), fill=(240, 240, 240))
    d.ellipse((50, 40, 62, 52), fill=(240, 240, 240))
    d.rectangle((38, 74, 58, 84), fill=(80, 60, 40))
    return img


def _content_bbox(img, bg):
    return img.point(lambda p: 255 if p != bg else 0).getbbox()


def _anchor(bbox):
    left, top, right, bottom = bbox
    return ((left + right) / 2, bottom)


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

def test_postprocess_default_size_is_native_768():
    # No downscale on the default path: individual sprites keep SD's full detail.
    assert postprocess(_rgb_image()).size == (768, 768)


def test_postprocess_size_param_resizes_to_96x96():
    assert _pp96(_rgb_image()).size == (96, 96)


def test_postprocess_output_is_palette_mode():
    assert _pp96(_rgb_image()).mode == "P"


def test_postprocess_obeys_gen3_contract():
    # postprocess now delegates to _quantize_gen3, so its output must obey the
    # Gen-3 palette contract. Feed a background-bearing sprite (not the fully
    # random _noisy_image, which hits the gradient-border fallback) so the
    # "background -> index 0" assertion is meaningful.
    out = _pp96(_noisy_border_sprite())
    pal = out.getpalette()
    # Reserved head of the palette: key at index 0, then black and white.
    assert pal[0:3] == [200, 200, 168]
    assert pal[3:6] == [0, 0, 0]
    assert pal[6:9] == [255, 255, 255]
    # At most 13 creature colours (used colours minus the three reserved) and at
    # most 16 total.
    used = _used_colors(out)
    reserved = {_KEY_COLOR, (0, 0, 0), (255, 255, 255)}
    assert len(used - reserved) <= _MAX_CREATURE_COLORS
    assert len(used) <= 16
    # Every border/background pixel decodes to index 0 (the transparency key).
    px = out.load()
    w, h = out.size
    for x in range(w):
        assert px[x, 0] == 0
        assert px[x, h - 1] == 0
    for y in range(h):
        assert px[0, y] == 0
        assert px[w - 1, y] == 0


def test_postprocess_does_not_mutate_input():
    img = _rgb_image()
    original_size = img.size
    _pp96(img)
    assert img.size == original_size


# ---------------------------------------------------------------------------
# quantize_to_reference()
# ---------------------------------------------------------------------------

def test_quantize_to_reference_output_is_palette_96x96():
    ref = _pp96(_noisy_image())
    out = quantize_to_reference(_rgb_image(), ref)
    assert out.mode == "P"
    assert out.size == (96, 96)


def test_quantize_to_reference_reuses_reference_palette():
    ref = _pp96(_noisy_image())
    out = quantize_to_reference(_rgb_image(), ref)
    assert out.getpalette() == ref.getpalette()


def test_quantize_to_reference_at_most_16_colors():
    ref = _pp96(_noisy_image())
    out = quantize_to_reference(_noisy_image(), ref)
    assert len(set(out.get_flattened_data())) <= 16


def test_quantize_to_reference_shares_palette_across_inputs():
    ref = _pp96(_noisy_image())
    out_a = quantize_to_reference(_rgb_image(color=(200, 100, 50)), ref)
    out_b = quantize_to_reference(_rgb_image(color=(20, 180, 220)), ref)
    assert out_a.getpalette() == out_b.getpalette() == ref.getpalette()


def test_quantize_to_reference_does_not_mutate_inputs():
    ref = _pp96(_noisy_image())
    ref_size, ref_mode, ref_palette = ref.size, ref.mode, ref.getpalette()
    img = _rgb_image()
    img_size, img_mode = img.size, img.mode
    quantize_to_reference(img, ref)
    assert (ref.size, ref.mode, ref.getpalette()) == (ref_size, ref_mode, ref_palette)
    assert (img.size, img.mode) == (img_size, img_mode)


def test_quantize_to_reference_rejects_non_palette_reference():
    with pytest.raises(ValueError, match="palette-mode"):
        quantize_to_reference(_rgb_image(), _rgb_image())


def test_quantize_to_reference_noisy_background_maps_to_index_0():
    # A candidate with a noisy near-white background must have its whole
    # background flattened to the key and nearest-map to index 0 (the key
    # slot) — NOT to the reserved white slot.
    reference = _pp96(_sprite_rgb())
    assert reference.getpalette()[0:3] == [200, 200, 168]  # key at index 0
    out = quantize_to_reference(_noisy_border_sprite(), reference)
    assert out.getpalette() == reference.getpalette()
    px = out.load()
    w, h = out.size
    for x in range(w):
        assert px[x, 0] == 0
        assert px[x, h - 1] == 0
    for y in range(h):
        assert px[0, y] == 0
        assert px[w - 1, y] == 0


def test_quantize_to_reference_preserves_reserved_slots_through_lock():
    reference = _pp96(_sprite_rgb())
    out = quantize_to_reference(_noisy_border_sprite(), reference)
    pal = out.getpalette()
    assert pal[0:3] == [200, 200, 168]  # key at index 0
    assert pal[3:6] == [0, 0, 0]        # reserved black
    assert pal[6:9] == [255, 255, 255]  # reserved white


# ---------------------------------------------------------------------------
# Back-sprite palette lock (pure core of the reference-locked back sprite)
# ---------------------------------------------------------------------------

def test_back_sprite_locks_to_reference_frame_palette():
    """A back RGB candidate quantized against frame 1's P-mode palette adopts it
    exactly — the pure core of the shared-palette back-sprite lock."""
    reference = _pp96(_sprite_rgb())
    back_rgb = _sprite_rgb(body=(90, 160, 210))
    locked = quantize_to_reference(back_rgb, reference)
    assert locked.mode == "P"
    assert locked.size == (96, 96)
    assert locked.getpalette() == reference.getpalette()


# ---------------------------------------------------------------------------
# Cross-view shiny consistency (front / frame2 / back share one rotated palette)
# ---------------------------------------------------------------------------

def test_cross_view_shinies_share_one_rotated_palette(tmp_path):
    """Three views sharing one palette yield three identical shiny palettes
    when generate_shiny runs with the same name (achromatic-preserving, palette
    rotation only)."""
    reference = _pp96(_sprite_rgb())
    # Stand-ins for frame1 / frame2 / back: three different RGB inputs quantized
    # against one reference, so all three share its exact palette.
    frame1 = quantize_to_reference(_sprite_rgb(), reference)
    frame2 = quantize_to_reference(_sprite_rgb(body=(90, 160, 210)), reference)
    back = quantize_to_reference(_noisy_image(96, 96), reference)
    assert frame1.getpalette() == frame2.getpalette() == back.getpalette()

    shiny_palettes = []
    for i, view in enumerate((frame1, frame2, back)):
        src = tmp_path / f"view_{i}.png"
        out = tmp_path / f"view_{i}_shiny.png"
        view.save(str(src))
        generate_shiny(str(src), "Flamburr", str(out))
        shiny_palettes.append(Image.open(str(out)).getpalette())

    assert shiny_palettes[0] == shiny_palettes[1] == shiny_palettes[2]
    # Rotation actually happened (mid-tone entries changed).
    assert shiny_palettes[0] != reference.getpalette()
    # Key / white / black are pinned: identical to the normals' across views.
    for shiny_pal in shiny_palettes:
        assert shiny_pal[0:3] == reference.getpalette()[0:3]  # key
        assert shiny_pal[3:6] == reference.getpalette()[3:6]  # black
        assert shiny_pal[6:9] == reference.getpalette()[6:9]  # white


def test_generate_shiny_pins_chromatic_key_at_index_0(tmp_path):
    """The key (200,200,168) is chromatic (lum ~196) but must never rotate."""
    sprite = _pp96(_sprite_rgb())
    assert sprite.getpalette()[0:3] == [200, 200, 168]
    src = tmp_path / "sprite.png"
    out = tmp_path / "sprite_shiny.png"
    sprite.save(str(src))
    generate_shiny(str(src), "Flamburr", str(out))
    assert Image.open(str(out)).getpalette()[0:3] == [200, 200, 168]


def test_generate_shiny_pins_white_black_but_rotates_creature(tmp_path):
    sprite = _pp96(_sprite_rgb())
    src = tmp_path / "sprite.png"
    out = tmp_path / "sprite_shiny.png"
    sprite.save(str(src))
    generate_shiny(str(src), "Flamburr", str(out))
    orig = sprite.getpalette()
    shiny = Image.open(str(out)).getpalette()
    # White and black entries are copied through unchanged.
    for i in range(0, len(orig), 3):
        triple = orig[i:i + 3]
        if triple in ([0, 0, 0], [255, 255, 255]):
            assert shiny[i:i + 3] == triple
    # ... yet rotation still happened for at least one creature (mid-tone) entry.
    assert any(orig[i:i + 3] != shiny[i:i + 3] for i in range(9, len(orig), 3))


# ---------------------------------------------------------------------------
# _flatten_background_to_key()
# ---------------------------------------------------------------------------

def _noisy_border_sprite():
    """96x96 RGB: noisy near-white background with a solid creature blob."""
    img = Image.new("RGB", (96, 96), (255, 255, 253))
    rng = random.Random(7)
    px = img.load()
    for y in range(96):
        for x in range(96):
            px[x, y] = (255 - rng.randint(0, 6), 255 - rng.randint(0, 6), 253 - rng.randint(0, 6))
    ImageDraw.Draw(img).ellipse((30, 30, 66, 66), fill=(200, 80, 60))
    return img


def _ring_sprite():
    """96x96 RGB: a creature disc with a background-coloured hole punched in it."""
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 76, 76), fill=(60, 120, 200))
    d.ellipse((40, 40, 56, 56), fill=(250, 250, 250))  # enclosed background pocket
    return img


def _gradient_border_sprite():
    """96x96 RGB whose border is a wide gradient (not near-uniform)."""
    img = Image.new("RGB", (96, 96))
    px = img.load()
    for y in range(96):
        for x in range(96):
            px[x, y] = (min(x * 2, 255), min(y * 2, 255), 100)
    return img


def test_flatten_keys_every_border_pixel_and_leaves_creature():
    img = _noisy_border_sprite()
    out = _flatten_background_to_key(img)
    assert out.mode == "RGB"
    assert out.size == img.size

    px = out.load()
    w, h = out.size
    for x in range(w):
        assert px[x, 0] == _KEY_COLOR
        assert px[x, h - 1] == _KEY_COLOR
    for y in range(h):
        assert px[0, y] == _KEY_COLOR
        assert px[w - 1, y] == _KEY_COLOR

    # Creature-blob pixels are untouched (well inside the ellipse).
    for point in ((48, 48), (45, 50), (50, 45)):
        assert px[point] == (200, 80, 60)


def test_flatten_keys_enclosed_pocket_via_global_sweep():
    img = _ring_sprite()
    out = _flatten_background_to_key(img)
    px = out.load()
    # The enclosed hole the outer flood cannot reach is keyed by the sweep.
    for point in ((48, 48), (46, 48), (48, 46)):
        assert px[point] == _KEY_COLOR
    # The creature ring itself is unchanged.
    assert px[28, 48] == (60, 120, 200)


def test_flatten_does_not_mutate_input():
    img = _noisy_border_sprite()
    original_data = list(img.get_flattened_data())
    original_size = img.size
    _flatten_background_to_key(img)
    assert img.size == original_size
    assert list(img.get_flattened_data()) == original_data


def test_flatten_gradient_border_warns_without_raising(capsys):
    img = _gradient_border_sprite()
    out = _flatten_background_to_key(img)  # must not raise
    assert out.mode == "RGB"
    assert out.size == img.size
    err = capsys.readouterr().err
    assert err  # a warning was emitted
    assert "border" in err.lower()


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


# ---------------------------------------------------------------------------
# _background_index()
# ---------------------------------------------------------------------------

def test_background_index_is_most_common_index():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    counts = {i: c for c, i in frame1.getcolors()}
    assert bg == max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# procedural_squash()
# ---------------------------------------------------------------------------

def test_procedural_squash_is_96x96_palette_mode():
    out = procedural_squash(_pp96(_sprite_rgb()))
    assert out.size == (96, 96)
    assert out.mode == "P"


def test_procedural_squash_shares_frame1_palette():
    frame1 = _pp96(_sprite_rgb())
    assert procedural_squash(frame1).getpalette() == frame1.getpalette()


def test_procedural_squash_differs_within_acceptance_band():
    frame1 = _pp96(_sprite_rgb())
    ratio = difference_ratio(procedural_squash(frame1), frame1)
    assert 0.0 < ratio
    assert 0.02 <= ratio <= 0.30


def test_procedural_squash_rejects_non_palette_input():
    with pytest.raises(ValueError, match="palette-mode"):
        procedural_squash(_rgb_image(96, 96))


def test_procedural_squash_does_not_mutate_input():
    frame1 = _pp96(_sprite_rgb())
    data = list(frame1.get_flattened_data())
    palette = frame1.getpalette()
    procedural_squash(frame1)
    assert list(frame1.get_flattened_data()) == data
    assert frame1.getpalette() == palette


# ---------------------------------------------------------------------------
# difference_ratio()
# ---------------------------------------------------------------------------

def test_difference_ratio_identical_is_zero():
    frame1 = _pp96(_sprite_rgb())
    assert difference_ratio(frame1, frame1) == 0.0


def _filled_creature(interior):
    """A full-frame creature (distinct interior) with a thin uniform border.

    quantize_to_reference now flattens the border to the key, so a *solid* image
    would collapse entirely to index 0; a bordered fill keeps the interior as
    creature content that quantizes on its own colour.
    """
    img = Image.new("RGB", (96, 96), (10, 10, 10))  # uniform border colour
    ImageDraw.Draw(img).rectangle((1, 1, 94, 94), fill=interior)
    return img


def test_difference_ratio_all_different_is_high():
    ref = _pp96(_sprite_rgb())
    # Two full-frame creatures whose interiors lock to different palette slots;
    # only the thin border keys to index 0, so nearly every pixel differs.
    a = quantize_to_reference(_filled_creature((20, 40, 200)), ref)
    b = quantize_to_reference(_filled_creature((240, 60, 20)), ref)
    assert difference_ratio(a, b) > 0.9


def test_difference_ratio_rejects_size_mismatch():
    a = _pp96(_sprite_rgb())
    b = a.resize((48, 48))
    with pytest.raises(ValueError):
        difference_ratio(a, b)


# ---------------------------------------------------------------------------
# recenter_to_anchor()
# ---------------------------------------------------------------------------

def test_recenter_aligns_shifted_candidate_to_frame1_anchor():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    # Build a shifted candidate that shares frame1's palette.
    shifted = Image.new("P", (96, 96), bg)
    shifted.putpalette(frame1.getpalette())
    shifted.paste(frame1, (12, -9))

    recentred = recenter_to_anchor(shifted, frame1)
    target = _anchor(_content_bbox(frame1, bg))
    got = _anchor(_content_bbox(recentred, bg))
    assert abs(got[0] - target[0]) <= 1
    assert abs(got[1] - target[1]) <= 1


def test_recenter_shares_frame1_palette():
    frame1 = _pp96(_sprite_rgb())
    recentred = recenter_to_anchor(frame1, frame1)
    assert recentred.mode == "P"
    assert recentred.size == (96, 96)
    assert recentred.getpalette() == frame1.getpalette()


def test_recenter_all_background_candidate_does_not_crash():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    blank = Image.new("P", (96, 96), bg)
    blank.putpalette(frame1.getpalette())
    out = recenter_to_anchor(blank, frame1)
    assert out.size == (96, 96)
    assert out.getpalette() == frame1.getpalette()


def test_recenter_rejects_non_palette_frame1():
    frame1 = _pp96(_sprite_rgb())
    with pytest.raises(ValueError, match="palette-mode"):
        recenter_to_anchor(frame1, _rgb_image(96, 96))


def test_recenter_does_not_mutate_inputs():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    shifted = Image.new("P", (96, 96), bg)
    shifted.putpalette(frame1.getpalette())
    shifted.paste(frame1, (12, -9))

    frame1_data = list(frame1.get_flattened_data())
    cand_data = list(shifted.get_flattened_data())
    recenter_to_anchor(shifted, frame1)
    assert list(frame1.get_flattened_data()) == frame1_data
    assert list(shifted.get_flattened_data()) == cand_data


# ---------------------------------------------------------------------------
# build_frame2()
# ---------------------------------------------------------------------------

def test_build_frame2_no_candidate_returns_squash():
    frame1 = _pp96(_sprite_rgb())
    out = build_frame2(frame1)
    assert out.getpalette() == frame1.getpalette()
    assert list(out.get_flattened_data()) == list(procedural_squash(frame1).get_flattened_data())
    assert difference_ratio(out, frame1) > 0.0


def test_build_frame2_near_identical_candidate_falls_back():
    frame1 = _pp96(_sprite_rgb())
    # quantize_to_reference now flattens _sprite_rgb's dark (40, 40, 60) backdrop
    # to the key (index 0) just like frame1's, so the same-creature candidate is
    # ~identical to frame1 -> ratio below low -> rejected -> squash fallback.
    out = build_frame2(frame1, _sprite_rgb())
    assert list(out.get_flattened_data()) == list(procedural_squash(frame1).get_flattened_data())


def test_build_frame2_wildly_different_candidate_falls_back():
    frame1 = _pp96(_sprite_rgb())
    out = build_frame2(frame1, _noisy_image(96, 96))
    assert list(out.get_flattened_data()) == list(procedural_squash(frame1).get_flattened_data())


def _key_background_sprite(body):
    """A _sprite_rgb whose dark backdrop is swapped for the transparency key.

    quantize_to_reference now flattens a candidate's background to the key, so a
    dark backdrop would key to index 0 too; pre-keying the backdrop keeps this
    helper explicit about matching frame 1's key background. Either way the sole
    difference from frame 1 is the recoloured creature region, yielding an
    in-band diff ratio.
    """
    img = _sprite_rgb(body=body)
    px = img.load()
    for y in range(96):
        for x in range(96):
            if px[x, y] == (40, 40, 60):
                px[x, y] = _KEY_COLOR
    return img


def test_build_frame2_in_band_candidate_is_accepted():
    frame1 = _pp96(_sprite_rgb())
    # A moderately different creature with a key background (so its backdrop
    # locks to index 0 like frame 1's) -> only the recoloured body differs ->
    # in-band difference.
    candidate = _key_background_sprite(body=(90, 160, 210))
    out = build_frame2(frame1, candidate)
    squash = list(procedural_squash(frame1).get_flattened_data())
    assert list(out.get_flattened_data()) != squash
    assert out.getpalette() == frame1.getpalette()
    assert 0.02 <= difference_ratio(out, frame1) <= 0.30


def test_build_frame2_rejects_non_palette_frame1():
    with pytest.raises(ValueError, match="palette-mode"):
        build_frame2(_rgb_image(96, 96))


def test_build_frame2_always_shares_palette_96x96():
    frame1 = _pp96(_sprite_rgb())
    for cand in (None, _sprite_rgb(), _noisy_image(96, 96)):
        out = build_frame2(frame1, cand)
        assert out.mode == "P"
        assert out.size == (96, 96)
        assert out.getpalette() == frame1.getpalette()


# ---------------------------------------------------------------------------
# _qg96()
# ---------------------------------------------------------------------------

def _multicolor_creature():
    """96x96 RGB: solid-white background (flattens to key) with a many-colour blob."""
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    rng = random.Random(3)
    px = img.load()
    for y in range(20, 76):
        for x in range(20, 76):
            px[x, y] = (rng.randint(0, 180), rng.randint(0, 180), rng.randint(0, 180))
    return img


def _used_colors(out):
    """Distinct RGB palette colours actually referenced by ``out``'s pixels."""
    pal = out.getpalette()
    return {tuple(pal[i * 3:i * 3 + 3]) for i in set(out.get_flattened_data())}


def test_quantize_gen3_output_is_palette_96x96():
    out = _qg96(_sprite_rgb())
    assert out.mode == "P"
    assert out.size == (96, 96)


def test_quantize_gen3_key_at_index_0():
    out = _qg96(_sprite_rgb())
    assert out.getpalette()[0:3] == [200, 200, 168]


def test_quantize_gen3_reserves_black_and_white_at_fixed_slots():
    out = _qg96(_sprite_rgb())
    pal = out.getpalette()
    assert pal[3:6] == [0, 0, 0]
    assert pal[6:9] == [255, 255, 255]


def test_quantize_gen3_creature_colour_budget():
    out = _qg96(_multicolor_creature())
    used = _used_colors(out)
    reserved = {_KEY_COLOR, (0, 0, 0), (255, 255, 255)}
    creature = used - reserved
    assert len(creature) <= _MAX_CREATURE_COLORS
    assert len(used) <= 16


def test_quantize_gen3_background_maps_to_index_0():
    out = _qg96(_noisy_border_sprite())
    px = out.load()
    w, h = out.size
    for x in range(w):
        assert px[x, 0] == 0
        assert px[x, h - 1] == 0
    for y in range(h):
        assert px[0, y] == 0
        assert px[w - 1, y] == 0


def test_quantize_gen3_enclosed_pocket_maps_to_index_0():
    out = _qg96(_ring_sprite())
    px = out.load()
    for point in ((48, 48), (46, 48), (48, 46)):
        assert px[point] == 0


def test_quantize_gen3_reserves_black_white_even_when_creature_uses_neither():
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((30, 30, 66, 66), fill=(120, 90, 150))
    out = _qg96(img)
    pal = out.getpalette()
    assert pal[3:6] == [0, 0, 0]
    assert pal[6:9] == [255, 255, 255]
    # ... and the creature genuinely uses neither reserved slot.
    used = _used_colors(out)
    assert (0, 0, 0) not in used
    assert (255, 255, 255) not in used


def test_quantize_gen3_nudges_creature_colour_off_key():
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    # Body deliberately near the key (dist 10 < _KEY_COLLISION_DISTANCE).
    ImageDraw.Draw(img).ellipse((30, 30, 66, 66), fill=(200, 200, 178))
    out = _qg96(img)
    pal = out.getpalette()
    indices = set(out.get_flattened_data())
    assert any(i != 0 for i in indices)  # the creature is visible, not swallowed
    for idx in indices:
        if idx == 0:
            continue
        colour = tuple(pal[idx * 3:idx * 3 + 3])
        assert _rgb_distance(colour, _KEY_COLOR) > _KEY_COLLISION_DISTANCE


def test_quantize_gen3_all_background_does_not_crash():
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    out = _qg96(img)
    assert out.mode == "P"
    assert set(out.get_flattened_data()) == {0}
    pal = out.getpalette()
    assert pal[0:3] == [200, 200, 168]
    assert pal[3:6] == [0, 0, 0]
    assert pal[6:9] == [255, 255, 255]


def test_quantize_gen3_does_not_mutate_input():
    img = _sprite_rgb()
    original_data = list(img.get_flattened_data())
    original_size = img.size
    _qg96(img)
    assert img.size == original_size
    assert list(img.get_flattened_data()) == original_data


# ---------------------------------------------------------------------------
# stitch_spritesheet()
# ---------------------------------------------------------------------------

_VIEW_COLORS = {
    "sprite.png": (200, 80, 60),
    "sprite_shiny.png": (60, 80, 200),
    "sprite_back.png": (80, 200, 60),
    "sprite_back_shiny.png": (200, 200, 60),
    "sprite_frame2.png": (200, 60, 200),
    "sprite_frame2_shiny.png": (60, 200, 200),
}


def _make_stage_dir(tmp_path, views=_VIEW_COLORS):
    """Write solid-colour 96x96 P-mode stand-ins for each requested view."""
    for name, color in views.items():
        img = Image.new("RGB", (96, 96), color)
        img.quantize(colors=2).save(str(tmp_path / name))
    return tmp_path


def _cell_color(sheet, col, row, cell=64):
    return sheet.convert("RGB").getpixel((col * cell + cell // 2, row * cell + cell // 2))


def test_spritesheet_default_is_256x128_rgb(tmp_path):
    # The sheet is the GBA deliverable: 64px cells by default, one single
    # downscale from the native-size views.
    stage = _make_stage_dir(tmp_path)
    out = tmp_path / "spritesheet.png"
    stitch_spritesheet(str(stage), str(out))
    sheet = Image.open(out)
    assert sheet.size == (256, 128)
    assert sheet.mode == "RGB"


def test_spritesheet_cell_layout_matches_reference_sheets(tmp_path):
    # Row 0: front, front-shiny, back, back-shiny; row 1: frame2, frame2-shiny.
    stage = _make_stage_dir(tmp_path)
    out = tmp_path / "spritesheet.png"
    stitch_spritesheet(str(stage), str(out))
    sheet = Image.open(out)
    assert _cell_color(sheet, 0, 0) == _VIEW_COLORS["sprite.png"]
    assert _cell_color(sheet, 1, 0) == _VIEW_COLORS["sprite_shiny.png"]
    assert _cell_color(sheet, 2, 0) == _VIEW_COLORS["sprite_back.png"]
    assert _cell_color(sheet, 3, 0) == _VIEW_COLORS["sprite_back_shiny.png"]
    assert _cell_color(sheet, 0, 1) == _VIEW_COLORS["sprite_frame2.png"]
    assert _cell_color(sheet, 1, 1) == _VIEW_COLORS["sprite_frame2_shiny.png"]


def test_spritesheet_empty_cells_are_key(tmp_path):
    stage = _make_stage_dir(tmp_path)
    out = tmp_path / "spritesheet.png"
    stitch_spritesheet(str(stage), str(out))
    sheet = Image.open(out)
    assert _cell_color(sheet, 2, 1) == _KEY_COLOR
    assert _cell_color(sheet, 3, 1) == _KEY_COLOR


def test_spritesheet_missing_view_leaves_cell_key(tmp_path):
    views = {k: v for k, v in _VIEW_COLORS.items() if k != "sprite_frame2.png"}
    stage = _make_stage_dir(tmp_path, views)
    out = tmp_path / "spritesheet.png"
    stitch_spritesheet(str(stage), str(out))   # must not raise
    sheet = Image.open(out)
    assert _cell_color(sheet, 0, 1) == _KEY_COLOR
    assert _cell_color(sheet, 1, 1) == _VIEW_COLORS["sprite_frame2_shiny.png"]


def test_spritesheet_cell_size_override(tmp_path):
    stage = _make_stage_dir(tmp_path)
    out = tmp_path / "sheet96.png"
    stitch_spritesheet(str(stage), str(out), cell_size=96)
    sheet = Image.open(out)
    assert sheet.size == (384, 192)
    assert _cell_color(sheet, 0, 0, cell=96) == _VIEW_COLORS["sprite.png"]


def test_spritesheet_downscale_introduces_no_new_colors(tmp_path):
    # k_centroid downscale must only pick existing tile colours, never blend
    # new ones in — each view fixture is solid-colour, so every cell's
    # dominant colour is exactly that view's colour.
    stage = _make_stage_dir(tmp_path)
    out = tmp_path / "spritesheet.png"
    stitch_spritesheet(str(stage), str(out))
    sheet_colors = set(Image.open(out).convert("RGB").get_flattened_data())
    allowed = set(_VIEW_COLORS.values()) | {_KEY_COLOR}
    assert sheet_colors <= allowed


# --------------------------------------------------------------------------
# k_centroid() — dominant-colour-per-tile RGB downscale
# --------------------------------------------------------------------------

def test_k_centroid_output_size_and_mode():
    img = Image.new("RGB", (12, 12), (10, 20, 30))
    out = k_centroid(img, 4, 3)
    assert out.size == (4, 3)
    assert out.mode == "RGB"


def test_k_centroid_hard_edge_introduces_no_new_colors():
    # Two solid-colour halves: every output pixel must be one of the two
    # source colours, never a blend (unlike LANCZOS ringing at hard edges).
    red, blue = (255, 0, 0), (0, 0, 255)
    img = Image.new("RGB", (12, 12), red)
    for x in range(6, 12):
        for y in range(12):
            img.putpixel((x, y), blue)
    out = k_centroid(img, 6, 6)
    out_colors = set(out.get_flattened_data())
    assert out_colors <= {red, blue}


# --------------------------------------------------------------------------
# Display-depth palette dedupe (Gen 3 shows 5 bits per channel)
# --------------------------------------------------------------------------

def test_display_key_collapses_channel_detail_below_5_bits():
    from fakemon_forge.sprites import _display_key
    assert _display_key((0, 0, 0)) == _display_key((4, 0, 0))
    assert _display_key((0, 0, 0)) == _display_key((0, 0, 7))
    assert _display_key((255, 255, 255)) == _display_key((248, 248, 248))
    assert _display_key((0, 0, 0)) != _display_key((8, 0, 0))


def test_dedupe_by_display_depth_drops_colors_reserved_slots_already_show():
    from fakemon_forge.sprites import _dedupe_by_display_depth, _KEY_COLOR
    reserved = [_KEY_COLOR, (0, 0, 0), (255, 255, 255)]
    # (4,0,0) is reserved black on hardware; (250,250,250) is reserved white.
    creature = [(4, 0, 0), (250, 250, 250), (120, 64, 32)]
    kept = _dedupe_by_display_depth(creature, reserved)
    assert kept == [(120, 64, 32)]


def test_dedupe_by_display_depth_drops_creature_colors_that_match_each_other():
    from fakemon_forge.sprites import _dedupe_by_display_depth
    kept = _dedupe_by_display_depth([(120, 64, 32), (121, 65, 33), (200, 8, 8)], [])
    assert kept == [(120, 64, 32), (200, 8, 8)]


def test_quantized_palette_has_no_two_slots_showing_one_color(tmp_path):
    """A palette entry pair indistinguishable on hardware wastes a slot.

    Two 8-bit-distinct colours that collapse to the same displayed colour
    would occupy two of the 16 slots while rendering identically.
    """
    from PIL import Image
    from fakemon_forge.sprites import postprocess, _display_key

    # A gradient rich in near-black tones: exactly what produced duplicate
    # displayed colours before the dedupe.
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for y in range(64):
        for x in range(32):
            img.putpixel((x, y), (x // 4, y // 8, (x + y) // 6))

    out = postprocess(img)
    palette = out.getpalette()
    used = sorted({idx for _, idx in out.getcolors(maxcolors=4096)})
    colors = [tuple(palette[i * 3:i * 3 + 3]) for i in used]
    keys = [_display_key(c) for c in colors]
    assert len(keys) == len(set(keys)), colors
