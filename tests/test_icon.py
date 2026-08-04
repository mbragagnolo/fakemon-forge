"""Tests for icon.py — the pure-Pillow party-menu icon post-processor.

Every test here is pure PIL: it builds a synthetic ``P``-mode sprite fixture by
running an RGB drawing through ``sprites.postprocess`` (mirroring the
``_frame1_file`` helper in ``test_sprites_ml.py``) and drives ``generate_icon``.
No torch / diffusers is imported, so this is a **regular** test file (not marked
``ml``) and runs in the keep sandbox.
"""

import pytest
from PIL import Image, ImageDraw

from fakemon_forge.sprites import (
    _KEY_COLLISION_DISTANCE,
    _KEY_COLOR,
    _rgb_distance,
    postprocess,
)
from fakemon_forge.icon import generate_icon, _build_frame1, _creature_palette

# The teal the reference icon uses for its own background. The generated icon
# deliberately does *not* use it — see the module docstring in icon.py.
_REFERENCE_TEAL = (96, 152, 128)


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

def test_icon_background_is_the_sprite_key_not_the_reference_teal(tmp_path):
    """Regression (#86): the icon shipped the reference icon's opaque teal at
    index 0 while the sprites shipped ``_KEY_COLOR``, so anything keying on the
    key colour matched the sprites and missed the icon — every party icon drew
    as a solid rectangle. One key colour across every generated asset."""
    out = _generate(tmp_path)
    icon = Image.open(str(out))
    assert tuple(icon.getpalette()[0:3]) == _KEY_COLOR
    colors = set(icon.convert("RGB").get_flattened_data())
    assert _REFERENCE_TEAL not in colors


def test_output_size_is_32x64(tmp_path):
    out = _generate(tmp_path)
    assert Image.open(str(out)).size == (32, 64)


def test_output_is_png(tmp_path):
    out = _generate(tmp_path)
    assert Image.open(str(out)).format == "PNG"


def test_output_carries_no_transparency_chunk(tmp_path):
    """Keying is by colour value, not by alpha — the icon matches the sprites,
    which carry no ``tRNS`` chunk either and key correctly downstream."""
    out = _generate(tmp_path)
    img = Image.open(str(out))
    assert "transparency" not in img.info
    # If the chosen mode carries an alpha channel, every pixel must be opaque.
    if img.mode in ("RGBA", "LA"):
        assert all(a == 255 for a in img.getchannel("A").getdata())


def test_output_has_at_most_16_colors(tmp_path):
    out = _generate(tmp_path)
    rgb = Image.open(str(out)).convert("RGB")
    assert len(set(rgb.get_flattened_data())) <= 16


def test_background_color_is_the_transparency_key(tmp_path):
    out = _generate(tmp_path)
    img = Image.open(str(out))
    rgb = img.convert("RGB")
    # The key dominates: it is the most common colour in the image.
    dominant = max(rgb.getcolors(32 * 64))[1]
    assert dominant == _KEY_COLOR
    # And it lives on palette index 0 when saved as P-mode.
    if img.mode == "P":
        assert tuple(img.getpalette()[0:3]) == _KEY_COLOR


def test_build_frame1_is_16_color_32x32(tmp_path):
    # _build_frame1 downscales via k_centroid (not LANCZOS) — its output
    # contract (size, P-mode, no tRNS, <= 16 colours) must be unchanged.
    src = _sprite_fixture(tmp_path)
    source = Image.open(str(src))
    frame1 = _build_frame1(source)
    assert frame1.mode == "P"
    assert frame1.size == (32, 32)
    assert "transparency" not in frame1.info
    rgb = frame1.convert("RGB")
    assert len(set(rgb.get_flattened_data())) <= 16


def test_frame2_is_frame1_shifted_down_one_pixel(tmp_path):
    out = _generate(tmp_path)
    rgb = Image.open(str(out)).convert("RGB")
    frame1 = rgb.crop((0, 0, 32, 32)).load()
    frame2 = rgb.crop((0, 32, 32, 64)).load()
    # Rows 1..31 of frame 2 equal rows 0..30 of frame 1.
    for y in range(1, 32):
        for x in range(32):
            assert frame2[x, y] == frame1[x, y - 1]
    # Frame 2's row 0 is the key background (backfilled by the down-shift).
    for x in range(32):
        assert frame2[x, 0] == _KEY_COLOR


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_background_source_yields_all_key(tmp_path):
    """A source that is entirely keyed background -> both frames pure key."""
    blank = postprocess(Image.new("RGB", (96, 96), (30, 30, 30)), size=96)
    src = tmp_path / "blank.png"
    blank.save(str(src))
    out = tmp_path / "sprite_small.png"
    generate_icon(str(src), str(out))
    rgb = Image.open(str(out)).convert("RGB")
    assert set(rgb.get_flattened_data()) == {_KEY_COLOR}
    assert rgb.size == (32, 64)


def _near_key_icon(tmp_path):
    """Icon built from a creature whose body colour sits close to the key."""
    img = Image.new("RGB", (96, 96), (40, 40, 60))
    # 40 units of Euclidean distance from the key — beyond _KEY_TOLERANCE (30),
    # so it reads as creature, not backfilled background.
    ImageDraw.Draw(img).ellipse((20, 20, 76, 76), fill=(200, 200, 128))
    src = tmp_path / "near_key.png"
    postprocess(img, size=96).save(str(src))
    out = tmp_path / "sprite_small.png"
    generate_icon(str(src), str(out))
    return Image.open(str(out))


def test_creature_colour_close_to_key_keeps_key_at_index_0(tmp_path):
    """A creature colour near (but beyond the tolerance of) the key must not
    displace the key from index 0; the key stays the background and the creature
    survives as distinct colours."""
    icon = _near_key_icon(tmp_path)
    assert tuple(icon.getpalette()[0:3]) == _KEY_COLOR
    colors = set(icon.convert("RGB").get_flattened_data())
    assert _KEY_COLOR in colors  # the key still present as the background
    assert any(c != _KEY_COLOR for c in colors)  # the near-key creature survived
    assert len(colors) <= 16


def test_no_creature_colour_collides_with_the_key(tmp_path):
    """The icon's analogue of the sprite guarantee: now that the background *is*
    the key, a creature colour sitting within the collision radius of it would be
    keyed away downstream, punching holes in the silhouette."""
    icon = _near_key_icon(tmp_path)
    creature = set(icon.convert("RGB").get_flattened_data()) - {_KEY_COLOR}
    for colour in creature:
        assert _rgb_distance(colour, _KEY_COLOR) > _KEY_COLLISION_DISTANCE


def test_creature_palette_nudges_a_key_coloured_centroid_off_the_key():
    """Median cut averages its inputs, so a centroid can land on the key even
    when the mask kept no key-coloured pixel; ``_creature_palette`` must push it
    clear rather than emit a creature colour that keys out."""
    rgb32 = Image.new("RGB", (32, 32), _KEY_COLOR)
    all_creature = Image.new("L", (32, 32), 0)  # nothing masked as background
    for colour in _creature_palette(rgb32, all_creature):
        assert _rgb_distance(colour, _KEY_COLOR) > _KEY_COLLISION_DISTANCE


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_non_palette_mode_input_raises_valueerror(tmp_path):
    src = tmp_path / "rgb.png"
    Image.new("RGB", (96, 96), (10, 20, 30)).save(str(src))
    with pytest.raises(ValueError):
        generate_icon(str(src), str(tmp_path / "out.png"))
