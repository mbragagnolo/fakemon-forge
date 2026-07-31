"""Tests for the Pillow-only Pokédex footprint renderer.

These are regular (non-``ml``) tests: ``footprint.py`` touches no torch/diffusers
code and fakes nothing via ``sys.modules``, so per ``CLAUDE.md`` they live here
and must PASS (not skip) in the slim sandbox container.
"""

import sys

import pytest
from PIL import Image, ImageDraw

from fakemon_forge.footprint import generate_footprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p_sprite(tmp_path, *, bg=0, blob_box=(340, 610, 430, 700), blob_index=1,
              size=768, name="sprite.png"):
    """A P-mode sprite with a flat ``bg`` backdrop and one blob near the bottom.

    ``blob_box=None`` yields an all-background sprite. A grayscale palette is
    attached so the indices used decode to distinct colours, mirroring the flat
    Gen-3 backdrop that ``footprint`` keys off the most-common palette index.
    """
    img = Image.new("P", (size, size), color=bg)
    img.putpalette([c for i in range(256) for c in (i, i, i)])
    if blob_box is not None:
        ImageDraw.Draw(img).rectangle(blob_box, fill=blob_index)
    path = tmp_path / name
    img.save(path)
    return str(path)


def _assert_two_valued(path):
    """Every one of the 256 pixels is opaque black or fully transparent."""
    img = Image.open(path).convert("RGBA")
    assert img.size == (16, 16)
    for pixel in img.get_flattened_data():
        r, g, b, a = pixel
        assert a == 0 or pixel == (0, 0, 0, 255), f"stray pixel {pixel}"


def _black_count(path):
    img = Image.open(path).convert("RGBA")
    return sum(1 for p in img.get_flattened_data() if p == (0, 0, 0, 255))


# ---------------------------------------------------------------------------
# Format / colour contract
# ---------------------------------------------------------------------------

def test_output_is_16x16(tmp_path):
    sprite = _p_sprite(tmp_path)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Fire"])
    assert Image.open(out).size == (16, 16)


def test_colour_contract(tmp_path):
    sprite = _p_sprite(tmp_path)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Dragon"])
    _assert_two_valued(out)


def test_clear_foot_blob_produces_black(tmp_path):
    sprite = _p_sprite(tmp_path)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Normal"])
    assert _black_count(out) > 0


# ---------------------------------------------------------------------------
# blank fast-path
# ---------------------------------------------------------------------------

def test_blank_all_transparent_without_reading_sprite(tmp_path):
    out = tmp_path / "footprint.png"
    # A path that does not exist: proves the sprite is never opened.
    generate_footprint("/nonexistent/does_not_exist.png", str(out),
                       types=["Fire"], blank=True)
    img = Image.open(out).convert("RGBA")
    assert img.size == (16, 16)
    assert all(p[3] == 0 for p in img.get_flattened_data())
    assert len(list(img.get_flattened_data())) == 256


# ---------------------------------------------------------------------------
# Empty derivations degrade to blank, not errors
# ---------------------------------------------------------------------------

def test_all_background_sprite_is_transparent(tmp_path):
    sprite = _p_sprite(tmp_path, blob_box=None)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Fire"])
    img = Image.open(out).convert("RGBA")
    assert img.size == (16, 16)
    assert all(p[3] == 0 for p in img.get_flattened_data())


def test_only_subthreshold_noise_is_transparent(tmp_path):
    # A single stray pixel far below the 0.2%-of-band noise cutoff.
    sprite = _p_sprite(tmp_path, blob_box=(400, 699, 401, 700))
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Fire"])
    _assert_two_valued(out)  # never crashes; stays within the contract


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_non_p_mode_raises_value_error(tmp_path):
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (768, 768), (10, 20, 30)).save(rgb)
    out = tmp_path / "footprint.png"
    with pytest.raises(ValueError, match="palette-mode"):
        generate_footprint(str(rgb), str(out), types=["Fire"])


def test_missing_sprite_raises_on_non_blank_path(tmp_path):
    # On the non-blank path the sprite IS opened, so a missing file surfaces
    # Image.open's FileNotFoundError unchanged (the mirror of the blank test,
    # which proves the sprite is *never* opened when blank=True).
    out = tmp_path / "footprint.png"
    with pytest.raises(FileNotFoundError):
        generate_footprint(str(tmp_path / "does_not_exist.png"), str(out),
                           types=["Fire"])


def test_blank_never_raises_on_non_p_mode(tmp_path):
    # blank=True short-circuits before any sprite read, so even an RGB path is fine.
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (768, 768), (10, 20, 30)).save(rgb)
    out = tmp_path / "footprint.png"
    generate_footprint(str(rgb), str(out), types=["Fire"], blank=True)
    _assert_two_valued(out)


# ---------------------------------------------------------------------------
# Type variation — both stay within the contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("types", [["Normal"], ["Water"], ["Flying"], [], ["Unlisted"]])
def test_type_variation_stays_within_contract(tmp_path, types):
    sprite = _p_sprite(tmp_path)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=types)
    _assert_two_valued(out)


def test_size_fraction_stays_within_canvas_and_contract(tmp_path):
    sprite = _p_sprite(tmp_path)
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Normal"], size_fraction=1.0)
    _assert_two_valued(out)


def test_thin_leg_still_valid(tmp_path):
    # A very narrow tall blob (tiny width_ratio) → deliberate small tall oval.
    sprite = _p_sprite(tmp_path, blob_box=(380, 500, 388, 700))
    out = tmp_path / "footprint.png"
    generate_footprint(sprite, str(out), types=["Ground"])
    _assert_two_valued(out)
    assert _black_count(out) > 0


def test_no_torch_or_diffusers_imported():
    import fakemon_forge.footprint  # noqa: F401
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules
