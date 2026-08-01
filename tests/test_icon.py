"""Tests for icon.py — the pure-Pillow party-menu icon post-processor.

Every test here is pure PIL: it builds a synthetic ``P``-mode sprite fixture by
running an RGB drawing through ``sprites.postprocess`` (mirroring the
``_frame1_file`` helper in ``test_sprites_ml.py``) and drives ``generate_icon``.
No torch / diffusers is imported, so this is a **regular** test file (not marked
``ml``) and runs in the keep sandbox.
"""

import pytest
from PIL import Image, ImageDraw

from fakemon_forge.sprites import postprocess
from fakemon_forge.icon import generate_icon, _ICON_BG_COLOR

_TEAL = (96, 152, 128)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sprite_fixture(tmp_path, name="sprite.png", size=96):
    """Write a real ``P``-mode front sprite fixture (96px for test speed)."""
    img = Image.new("RGB", (size, size), (40, 40, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((26, 28, 70, 84), fill=(200, 80, 60))
    d.ellipse((34, 40, 46, 52), fill=(240, 240, 240))
    d.rectangle((38, 74, 58, 84), fill=(80, 60, 40))
    sprite = postprocess(img, size=size)
    path = tmp_path / name
    sprite.save(str(path))
    return path


def _generate(tmp_path):
    src = _sprite_fixture(tmp_path)
    out = tmp_path / "sprite_small.png"
    generate_icon(str(src), str(out))
    return out


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_icon_bg_constant_is_teal():
    assert _ICON_BG_COLOR == _TEAL


def test_output_size_is_32x64(tmp_path):
    out = _generate(tmp_path)
    assert Image.open(str(out)).size == (32, 64)


def test_output_is_png(tmp_path):
    out = _generate(tmp_path)
    assert Image.open(str(out)).format == "PNG"


def test_output_is_fully_opaque(tmp_path):
    out = _generate(tmp_path)
    img = Image.open(str(out))
    # No PNG transparency chunk / reserved transparency index in use.
    assert "transparency" not in img.info
    # If the chosen mode carries an alpha channel, every pixel must be opaque.
    if img.mode in ("RGBA", "LA"):
        assert all(a == 255 for a in img.getchannel("A").getdata())


def test_output_has_at_most_16_colors(tmp_path):
    out = _generate(tmp_path)
    rgb = Image.open(str(out)).convert("RGB")
    assert len(set(rgb.get_flattened_data())) <= 16


def test_background_color_is_teal(tmp_path):
    out = _generate(tmp_path)
    img = Image.open(str(out))
    rgb = img.convert("RGB")
    # Teal dominates: it is the most common colour in the image.
    dominant = max(rgb.getcolors(32 * 64))[1]
    assert dominant == _TEAL
    # And it lives on palette index 0 when saved as P-mode.
    if img.mode == "P":
        assert tuple(img.getpalette()[0:3]) == _TEAL


def test_frame2_is_frame1_shifted_down_one_pixel(tmp_path):
    out = _generate(tmp_path)
    rgb = Image.open(str(out)).convert("RGB")
    frame1 = rgb.crop((0, 0, 32, 32)).load()
    frame2 = rgb.crop((0, 32, 32, 64)).load()
    # Rows 1..31 of frame 2 equal rows 0..30 of frame 1.
    for y in range(1, 32):
        for x in range(32):
            assert frame2[x, y] == frame1[x, y - 1]
    # Frame 2's row 0 is the teal background (backfilled by the down-shift).
    for x in range(32):
        assert frame2[x, 0] == _TEAL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_background_source_yields_all_teal(tmp_path):
    """A source that is entirely keyed background -> both frames pure teal."""
    blank = postprocess(Image.new("RGB", (96, 96), (30, 30, 30)), size=96)
    src = tmp_path / "blank.png"
    blank.save(str(src))
    out = tmp_path / "sprite_small.png"
    generate_icon(str(src), str(out))
    rgb = Image.open(str(out)).convert("RGB")
    assert set(rgb.get_flattened_data()) == {_TEAL}
    assert rgb.size == (32, 64)


def test_creature_colour_close_to_teal_keeps_teal_at_index_0(tmp_path):
    """A creature colour near (but beyond the key tolerance of) teal must not
    displace teal from index 0; teal stays the background and the creature
    survives as distinct colours (spec Edge cases: 'creature colour close to
    teal')."""
    img = Image.new("RGB", (96, 96), (40, 40, 60))
    # 40 units of Euclidean distance from teal — beyond _KEY_TOLERANCE (30), so
    # it reads as creature, not backfilled background.
    ImageDraw.Draw(img).ellipse((20, 20, 76, 76), fill=(96, 152, 88))
    src = tmp_path / "near_teal.png"
    postprocess(img, size=96).save(str(src))
    out = tmp_path / "sprite_small.png"
    generate_icon(str(src), str(out))
    icon = Image.open(str(out))
    assert tuple(icon.getpalette()[0:3]) == _TEAL
    colors = set(icon.convert("RGB").get_flattened_data())
    assert _TEAL in colors  # teal still present as the background
    assert any(c != _TEAL for c in colors)  # the near-teal creature survived
    assert len(colors) <= 16


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_non_palette_mode_input_raises_valueerror(tmp_path):
    src = tmp_path / "rgb.png"
    Image.new("RGB", (96, 96), (10, 20, 30)).save(str(src))
    with pytest.raises(ValueError):
        generate_icon(str(src), str(tmp_path / "out.png"))
