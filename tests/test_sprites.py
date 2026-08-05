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
    generate_sprite_img2img,
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
    split_front_back_canvas,
    _border_ring,
    _border_is_uniform,
    _detect_background,
    _split_front_back_with_retry,
    _fit_half_to_square,
    _content_columns,
    _estimate_clip_tokens,
    _is_nw_lit,
    _CLIP_TOKEN_LIMIT,
    _FRAMING_TAGS,
    _KEY_COLOR,
    _KEY_TOLERANCE,
    _MAX_CREATURE_COLORS,
    _KEY_COLLISION_DISTANCE,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    make_img2img_pipeline,
    _BASE_MODEL_ID,
    _LORA_PATH,
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

def test_build_prompt_no_extra_tags_uses_plain_formula():
    assert build_prompt("spiky wolf") == (
        f"gen3, {_FRAMING_TAGS}, spiky wolf, white background"
    )


def test_build_prompt_empty_extra_tags_list_matches_none():
    assert build_prompt("spiky wolf", []) == build_prompt("spiky wolf", None)
    assert build_prompt("spiky wolf", []) == (
        f"gen3, {_FRAMING_TAGS}, spiky wolf, white background"
    )


def test_build_prompt_single_extra_tag_inserted_before_white_background():
    result = build_prompt("fire lizard", ["chibi"])
    assert result == f"gen3, {_FRAMING_TAGS}, fire lizard, chibi, white background"


def test_build_prompt_multiple_extra_tags_joined_with_comma():
    result = build_prompt("chibi crab", ["chibi", "big head", "small body"])
    assert result == (
        f"gen3, {_FRAMING_TAGS}, chibi crab, chibi, big head, small body, white background"
    )


# ---------------------------------------------------------------------------
# build_prompt() framing anchors
# ---------------------------------------------------------------------------
# The sprite_prompt vocabulary is shape/colour/features and never states the
# framing, which left the model free to read a size word as "fill the canvas".
# A live stage-2 prompt ("large ceramic mug ... imposing stance") rendered a
# full-bleed close-up with no background left — and so no uniform border for
# the pair split to find. generator.py now bans those words; these pin the
# code-level half, which holds whatever the LLM writes.

def test_build_prompt_states_the_framing_every_time():
    for prompt in (build_prompt("fire lizard"), build_prompt("fire lizard", ["chibi"])):
        assert "single creature" in prompt
        assert "full body" in prompt
        assert "centered" in prompt


def test_build_prompt_framing_anchors_lead_the_prompt():
    """Directly after the LoRA trigger: earliest tokens carry the most weight,
    and nothing appended later can push them out of CLIP's window."""
    assert build_prompt("fire lizard").startswith(f"gen3, {_FRAMING_TAGS}, ")


def test_build_prompt_strips_a_framing_word_but_keeps_its_tag():
    """The generator spec bans these, but the LLM does not reliably comply, so
    the guarantee lives here. Only the word goes — the creature stays described."""
    result = build_prompt("large ceramic cauldron, glowing red cracks")
    assert "ceramic cauldron" in result
    assert "large" not in result


def test_build_prompt_strips_every_banned_framing_word(capsys):
    result = build_prompt(
        "towering hulk, imposing stance, dramatic close-up, massive claws, epic aura"
    )
    for banned in ("towering", "imposing", "dramatic", "close-up", "massive", "epic"):
        assert banned not in result.lower()
    assert "hulk" in result and "claws" in result and "aura" in result
    assert "stripped framing/scale words" in capsys.readouterr().err


def test_build_prompt_framing_strip_is_case_insensitive_and_word_bounded():
    """"Large" capitalised must still go; "enlarged" must not be mangled."""
    assert "Large" not in build_prompt("Large horn")
    assert "enlarged" in build_prompt("enlarged horn")


def test_build_prompt_strips_small_size_words_but_keeps_their_tags():
    """The small direction of the same failure: on the first ROM-injection
    round, 94 of 99 stage-1 prompts said "tiny"/"small" and filled a median
    48% of the canvas against 78% without — a speck once a GBA cell divides
    that by 12. Only the size word goes; the anatomy it modified stays."""
    result = build_prompt("tiny round rodent, small pointed ears, little claws")
    for banned in ("tiny", "small", "little"):
        assert banned not in result.lower()
    assert "round rodent" in result
    assert "pointed ears" in result
    assert "claws" in result


def test_build_prompt_small_strip_is_word_bounded():
    """"smallpox pattern" is fanciful but "minicorn" makes the point: no
    substring mangling on either side of the new words."""
    assert "minicorn horn" in build_prompt("minicorn horn")
    assert "smallish" in build_prompt("smallish frill")


def test_build_prompt_leaves_proportion_words_alone():
    """"stubby"/"short" describe the creature's shape, not the framing — they
    are exactly where juvenile-ness goes now that size words are stripped."""
    result = build_prompt("stubby limbs, short tail, plump body")
    assert "stubby limbs" in result
    assert "short tail" in result


def test_build_prompt_drops_a_tag_that_was_only_a_framing_word():
    result = build_prompt("round blob, imposing, teal scales")
    body = result[len(f"gen3, {_FRAMING_TAGS}, "):-len(", white background")]
    assert body == "round blob, teal scales"


def test_build_prompt_leaves_a_clean_prompt_unstripped(capsys):
    """No warning when there was nothing to strip — the message means a real
    contract violation happened, so it must not cry wolf."""
    build_prompt("round ceramic mug, glossy white porcelain")
    assert "stripped framing/scale words" not in capsys.readouterr().err


def test_build_prompt_framing_anchors_survive_a_trim():
    """They are in the prefix, so the trim budget subtracts them rather than
    dropping them — an over-long description cannot cost the framing."""
    result = build_prompt(_long_sprite_prompt(30))
    assert result.startswith(f"gen3, {_FRAMING_TAGS}, ")
    assert result.endswith(", white background")


# ---------------------------------------------------------------------------
# build_prompt() CLIP token guard
# ---------------------------------------------------------------------------
# CLIP drops everything past 77 tokens silently, and the style anchors sit at
# the END of the assembled prompt — so an over-long sprite_prompt deletes
# "white background" and the extra_tags, not itself. Observed live before the
# generator.py contract was tightened: every prompt in a 3-stage run lost its
# background instruction, and the vignetted backdrops that followed defeated
# split_front_back_canvas on all 3 stages.

def _long_sprite_prompt(n=14):
    return ", ".join(f"distinctive shimmering feature number {i}" for i in range(n))


def test_build_prompt_leaves_an_in_spec_prompt_completely_untouched():
    """The guard is a safety net, not a rewriter: a prompt that already fits
    must come out byte-identical to the plain formula."""
    assert build_prompt("fire lizard") == (
        f"gen3, {_FRAMING_TAGS}, fire lizard, white background"
    )
    assert build_prompt("fire lizard", ["chibi"]) == (
        f"gen3, {_FRAMING_TAGS}, fire lizard, chibi, white background"
    )


def test_build_prompt_trims_the_description_not_the_style_tags():
    result = build_prompt(_long_sprite_prompt(), ["chibi", "big head", "small body"])
    assert result.startswith(f"gen3, {_FRAMING_TAGS}, distinctive shimmering feature number 0")
    assert result.endswith(", chibi, big head, small body, white background")


def test_build_prompt_trimmed_result_fits_the_token_limit():
    result = build_prompt(_long_sprite_prompt(30))
    assert _estimate_clip_tokens(result) <= _CLIP_TOKEN_LIMIT


def test_build_prompt_drops_whole_tags_never_half_of_one():
    """Cutting mid-tag would feed the model a fragment ("neon cyan and") that
    reads as a different feature than the tag it came from."""
    result = build_prompt(_long_sprite_prompt())
    body = result[len(f"gen3, {_FRAMING_TAGS}, "):-len(", white background")]
    for tag in body.split(", "):
        assert tag.startswith("distinctive shimmering feature number ")


def test_build_prompt_warns_when_it_trims(capsys):
    build_prompt(_long_sprite_prompt())
    err = capsys.readouterr().err
    assert "build_prompt dropped" in err
    assert "77-token window" in err


def test_build_prompt_keeps_one_tag_even_when_that_alone_overruns():
    """Degrade, never return an empty description — a prompt of pure style
    tags would render an arbitrary creature rather than a trimmed one."""
    huge = " ".join(["word"] * 200)
    result = build_prompt(huge)
    assert result == f"gen3, {_FRAMING_TAGS}, {huge}, white background"


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


def test_quantize_to_reference_never_keys_creature_midtones():
    # Regression (#90): the reference palette holds no beige, so a midtone body
    # sat closer to the key than to any reference entry and the lock mapped it
    # to index 0 — transparency holes through the creature. Index 0 must come
    # only from the flatten's background, never from colour proximity.
    reference = _pp96(_sprite_rgb())
    out = quantize_to_reference(_midtone_creature(), reference)
    data = out.get_flattened_data()
    body = [data[y * 96 + x] for y in range(28, 68) for x in range(28, 68)]
    assert 0 not in body


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


_OUTLINE = (20, 24, 40)   # dark silhouette edge, as the pixel-art LoRA renders it
_BODY = (60, 120, 200)


def _ring_sprite():
    """96x96 RGB: a creature disc with a background-coloured hole punched in it.

    Drawn with the dark silhouette outline a real render carries, around the
    outside *and* around the hole — seeing through the sprite is exactly what
    puts an outline there, and it is the only thing that distinguishes this
    hole from ``_highlight_sprite``'s painted patch. Without it the two images
    are structurally identical (a near-bg island surrounded by body colour) and
    no rule could key one and spare the other.
    """
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 76, 76), fill=_OUTLINE)
    d.ellipse((23, 23, 73, 73), fill=_BODY)
    d.ellipse((40, 40, 56, 56), fill=_OUTLINE)         # the hole's outline rim
    d.ellipse((43, 43, 53, 53), fill=(250, 250, 250))  # enclosed background pocket
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


def test_flatten_keys_enclosed_pocket_via_connected_component_scan():
    img = _ring_sprite()
    out = _flatten_background_to_key(img)
    px = out.load()
    # The enclosed hole the outer flood cannot reach is keyed by the scan.
    for point in ((48, 48), (46, 48), (48, 46)):
        assert px[point] == _KEY_COLOR
    # The creature ring itself is unchanged.
    assert px[28, 48] == (60, 120, 200)


def _highlight_sprite(box=(44, 44, 52, 52)):
    """96x96 RGB: a creature disc with a near-bg detail patch painted on it.

    Unlike ``_ring_sprite``'s hole, this patch is not a background pocket — it's
    a same-coloured detail (a shield highlight, a white belly patch) that just
    happens to be within ``_KEY_TOLERANCE`` of the background colour. It sits on
    body colour with no outline around it, because you cannot see through it.
    """
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 76, 76), fill=_OUTLINE)
    d.ellipse((23, 23, 73, 73), fill=_BODY)
    d.rectangle(box, fill=(245, 245, 245))  # near-bg detail, not a pocket
    return img


def test_flatten_leaves_isolated_near_background_detail_patch_untouched():
    img = _highlight_sprite()
    out = _flatten_background_to_key(img)
    px = out.load()
    # The highlight is colour-close to bg, but it is walled by body colour
    # rather than by the silhouette outline, so it must survive untouched.
    for point in ((48, 48), (46, 48), (48, 46)):
        assert px[point] == (245, 245, 245)
    # The creature disc itself is unchanged.
    assert px[28, 48] == _BODY


def test_flatten_leaves_large_near_background_detail_patch_untouched():
    """A belly patch is spared however big it is — size does not decide this.

    Regression test: gating on a minimum area keyed anything past the gate, so
    a patch this size came out as a hole punched through the creature.
    """
    img = _highlight_sprite(box=(38, 38, 58, 58))  # 21x21, ~4.8% of the image
    out = _flatten_background_to_key(img)
    px = out.load()
    for point in ((48, 48), (40, 40), (56, 56)):
        assert px[point] == (245, 245, 245)


def _small_gap_sprite():
    """96x96 RGB: an outlined creature disc with a tiny see-through gap in it."""
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 86, 86), fill=_OUTLINE)
    d.ellipse((13, 13, 83, 83), fill=_BODY)
    d.ellipse((43, 43, 53, 53), fill=_OUTLINE)
    d.ellipse((45, 45, 51, 51), fill=(250, 250, 250))  # 7x7, ~0.5% of the image
    return img


def test_flatten_keys_small_enclosed_pocket():
    """A real gap is keyed however small — size does not decide this either.

    Regression test: gating on a minimum area left gaps under the gate
    background-coloured, so a between-the-legs gap shipped as a white wedge.
    """
    out = _flatten_background_to_key(_small_gap_sprite())
    px = out.load()
    assert px[48, 48] == _KEY_COLOR
    assert px[28, 48] == _BODY  # the creature body is unchanged


def _open_notch_sprite():
    """96x96 RGB: a U-shaped creature whose notch opens onto the top border.

    The notch is background-coloured, and the corner flood fill cannot reach it
    (the creature walls it off from every corner) — but it runs to the image
    border, so it is outer background seen between the walls, not creature.
    """
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 0, 37, 50), fill=_BODY)   # left wall, touches top border
    d.rectangle((58, 0, 65, 50), fill=_BODY)   # right wall, touches top border
    d.rectangle((30, 44, 65, 50), fill=_BODY)  # floor
    return img


def test_flatten_keys_border_touching_background_component():
    """Background that reaches the image edge is keyed even if the flood misses it.

    Regression test: treating "touches the border" as *disqualifying* left this
    notch unkeyed, so a leg gap opening onto an edge shipped opaque.
    """
    img = _open_notch_sprite()
    out = _flatten_background_to_key(img)
    px = out.load()
    for point in ((48, 0), (48, 20), (40, 40)):
        assert px[point] == _KEY_COLOR
    # Outside the walls, the corner flood still keys the outer background.
    assert px[10, 20] == _KEY_COLOR
    assert px[90, 20] == _KEY_COLOR
    # The walls themselves survive.
    assert px[34, 20] == _BODY


def _dark_creature(gap: bool):
    """96x96 RGB: a near-black creature with either a see-through gap or a highlight.

    The hard case for telling outline from body by brightness: the whole
    creature is dark, so any *absolute* dark-cutoff calls the body outline too.
    """
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 86, 86), fill=(8, 8, 12))     # outline
    d.ellipse((13, 13, 83, 83), fill=(34, 30, 46))   # near-black body
    if gap:
        d.ellipse((42, 42, 54, 54), fill=(8, 8, 12))
        d.ellipse((45, 45, 51, 51), fill=(250, 250, 250))
    else:
        d.rectangle((42, 42, 54, 54), fill=(245, 245, 245))
    return img


def test_flatten_keys_see_through_gap_in_a_near_black_creature():
    out = _flatten_background_to_key(_dark_creature(gap=True))
    assert out.load()[48, 48] == _KEY_COLOR


def test_flatten_leaves_highlight_on_a_near_black_creature_untouched():
    """A white marking on a black body is detail, not a pocket.

    The outline cutoff scales with the creature's own mean luma for this: a
    fixed margin below the mean goes negative on a body this dark, at which
    point nothing counts as outline and every real gap would ship opaque.
    """
    out = _flatten_background_to_key(_dark_creature(gap=False))
    assert out.load()[48, 48] == (245, 245, 245)


def test_flatten_keys_background_fleck_the_flood_cannot_cross():
    """Open-background pixels the stage-1 flood steps over are still keyed.

    ``ImageDraw.floodfill`` thresholds on Manhattan distance and walks
    4-connected, while the stage-2 scan uses Euclidean ``_rgb_distance`` and
    8-connectivity, so pixels like ``(230, 230, 250)`` against a ``(250, 250,
    250)`` background are inside one metric and outside the other. Regression
    test: such flecks survived into the sprite, inflating its content bbox and
    misaligning the frame-2 recentre.
    """
    img = Image.new("RGB", (96, 96), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.ellipse((30, 30, 66, 66), fill=_BODY)
    d.point((12, 12), fill=(230, 230, 250))  # euclid 28.3 (in), manhattan 40 (out)
    d.point((84, 70), fill=(230, 230, 250))
    out = _flatten_background_to_key(img)
    px = out.load()
    assert px[12, 12] == _KEY_COLOR
    assert px[84, 70] == _KEY_COLOR


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

def test_backend_constants_point_at_the_sdxl_stack():
    # Pins the backend swap itself: the SDXL base model and the "back&front"
    # kohya LoRA filename (a manual Civitai download, never committed — the
    # exact name is the contract with whoever downloads it).
    assert _BASE_MODEL_ID == "Laxhar/noobai-XL-1.1"
    assert _LORA_PATH.name == "pkspbf_nb_v1.safetensors"
    assert _LORA_PATH.parent.name == "loras"


def _make_lora_pipeline_mock(state_dict=None, network_alphas=None):
    mock_mixin = MagicMock()
    mock_mixin.lora_state_dict.return_value = (
        {} if state_dict is None else state_dict,
        {} if network_alphas is None else network_alphas,
        None,
    )
    mock_mod = MagicMock()
    mock_mod.StableDiffusionXLLoraLoaderMixin = mock_mixin
    return mock_mod


def _mock_modules(pipe_side_effect=None, cuda=False, lora_mod=None):
    mock_pipe_cls = MagicMock()
    if pipe_side_effect:
        mock_pipe_cls.from_pretrained.side_effect = pipe_side_effect
    else:
        mock_pipe_cls.from_pretrained.return_value = MagicMock()

    mock_diffusers = MagicMock()
    mock_diffusers.StableDiffusionXLPipeline = mock_pipe_cls

    mock_torch = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.cuda.is_available.return_value = cuda

    return {
        "diffusers": mock_diffusers,
        "torch": mock_torch,
        "diffusers.loaders": MagicMock(),
        "diffusers.loaders.lora_pipeline": lora_mod or _make_lora_pipeline_mock(),
        # _apply_lora imports these too; every module it touches must be faked
        # here, or the import leaks past the sandbox and half-initializes the
        # real transformers/numpy stack, breaking later ml-marked tests.
        "transformers": MagicMock(),
        "transformers.conversion_mapping": MagicMock(),
    }, mock_pipe_cls


class _NoTilingPipe:
    """A pipeline stand-in that lacks ``enable_vae_tiling`` (unlike MagicMock,
    which auto-creates any attribute), so hasattr() genuinely reports False and
    the ``pipe.vae.enable_tiling()`` fallback path can be exercised."""

    def __init__(self):
        self.to = MagicMock(side_effect=lambda device: self)
        self.enable_model_cpu_offload = MagicMock()
        self.vae = MagicMock()
        self.unet = MagicMock()
        self.text_encoder = MagicMock()
        self.text_encoder_2 = MagicMock()
        self.scheduler = MagicMock()
        self.lora_scale = 1.0
        self.load_lora_into_unet = MagicMock()
        self.load_lora_into_text_encoder = MagicMock()
        self.fuse_lora = MagicMock()


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


def test_load_does_not_pass_safety_checker():
    # SDXL pipeline classes don't accept safety_checker (unlike SD1.5's) — a
    # real .from_pretrained call would raise TypeError if it were still passed.
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    assert "safety_checker" not in mock_pipe_cls.from_pretrained.call_args.kwargs


def test_load_applies_lora_weights():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.load_lora_into_unet.assert_called_once()
    assert pipe.load_lora_into_text_encoder.call_count == 2
    pipe.fuse_lora.assert_called_once_with(lora_scale=_LORA_SCALE)


def test_load_lora_state_dict_passes_unet_config():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    mixin = mods["diffusers.loaders.lora_pipeline"].StableDiffusionXLLoraLoaderMixin
    mixin.lora_state_dict.assert_called_once_with(
        str(_LORA_PATH), return_lora_metadata=True, unet_config=pipe.unet.config
    )


def test_load_applies_lora_to_both_text_encoders():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    calls = pipe.load_lora_into_text_encoder.call_args_list
    encoders = {c.kwargs["text_encoder"] for c in calls}
    prefixes = {c.kwargs["prefix"] for c in calls}
    assert encoders == {pipe.text_encoder, pipe.text_encoder_2}
    assert prefixes == {"text_encoder", "text_encoder_2"}


# Kohya key shapes as diffusers hands them back from lora_state_dict(): both
# text encoders arrive under a "{prefix}.text_model." level. te1
# (CLIPTextModel) does NOT have that wrapper in its named_modules(), so its
# keys must be stripped; te2 (CLIPTextModelWithProjection) does, so its keys
# must survive untouched.
_TE_KEY_1 = "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"
_TE_KEY_2 = "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight"
_UNET_KEY = "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight"


def _kohya_state_dict():
    return {_UNET_KEY: "u", _TE_KEY_1: "a", _TE_KEY_2: "b"}


def _text_encoder_calls(pipe):
    """The two load_lora_into_text_encoder calls, keyed by their ``prefix``."""
    return {c.kwargs["prefix"]: c for c in pipe.load_lora_into_text_encoder.call_args_list}


def test_load_strips_text_model_level_for_text_encoder_1_only():
    state_dict = _kohya_state_dict()
    mods, mock_pipe_cls = _mock_modules(
        lora_mod=_make_lora_pipeline_mock(state_dict, dict(state_dict))
    )
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    calls = _text_encoder_calls(mock_pipe_cls.from_pretrained.return_value)

    te1 = calls["text_encoder"].args[0]
    assert "text_encoder.encoder.layers.0.self_attn.q_proj.lora_A.weight" in te1
    assert _TE_KEY_1 not in te1
    # Only te1's own level is stripped: te2 and unet keys ride along untouched.
    assert _TE_KEY_2 in te1 and _UNET_KEY in te1


def test_load_leaves_text_encoder_2_keys_untouched():
    state_dict = _kohya_state_dict()
    mods, mock_pipe_cls = _mock_modules(
        lora_mod=_make_lora_pipeline_mock(state_dict, dict(state_dict))
    )
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    calls = _text_encoder_calls(mock_pipe_cls.from_pretrained.return_value)

    te2 = calls["text_encoder_2"].args[0]
    # Blanket-applying the te1 strip here would silently break te2's weights.
    assert _TE_KEY_2 in te2
    assert "text_encoder_2.encoder.layers.0.self_attn.q_proj.lora_A.weight" not in te2


def test_load_applies_the_same_key_fix_to_network_alphas():
    state_dict = _kohya_state_dict()
    mods, mock_pipe_cls = _mock_modules(
        lora_mod=_make_lora_pipeline_mock(state_dict, dict(state_dict))
    )
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    calls = _text_encoder_calls(mock_pipe_cls.from_pretrained.return_value)

    assert _TE_KEY_1 not in calls["text_encoder"].kwargs["network_alphas"]
    assert _TE_KEY_2 in calls["text_encoder_2"].kwargs["network_alphas"]


def test_load_passes_unstripped_state_dict_to_the_unet():
    state_dict = _kohya_state_dict()
    mods, mock_pipe_cls = _mock_modules(lora_mod=_make_lora_pipeline_mock(state_dict))
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    assert pipe.load_lora_into_unet.call_args.args[0] == state_dict


def test_load_tolerates_empty_network_alphas():
    # lora_state_dict() returns an empty/None alphas mapping for LoRAs that
    # carry no alpha entries; the key fix must pass it straight through.
    mods, mock_pipe_cls = _mock_modules(
        lora_mod=_make_lora_pipeline_mock(_kohya_state_dict(), network_alphas=None)
    )
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    calls = _text_encoder_calls(mock_pipe_cls.from_pretrained.return_value)
    assert calls["text_encoder"].kwargs["network_alphas"] == {}


def test_load_sets_euler_ancestral_scheduler():
    mods, mock_pipe_cls = _mock_modules()
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    scheduler_cls = mods["diffusers"].EulerAncestralDiscreteScheduler
    scheduler_cls.from_config.assert_called_once()
    assert pipe.scheduler is scheduler_cls.from_config.return_value


def test_load_enables_model_cpu_offload_on_cuda():
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.enable_model_cpu_offload.assert_called_once()


def test_load_enables_vae_tiling_on_cuda():
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.enable_vae_tiling.assert_called_once()


def test_load_falls_back_to_vae_enable_tiling_when_pipeline_method_absent():
    pipe_instance = _NoTilingPipe()
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    mock_pipe_cls.from_pretrained.return_value = pipe_instance
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe_instance.vae.enable_tiling.assert_called_once()


def test_load_skips_offload_and_tiling_on_cpu():
    mods, mock_pipe_cls = _mock_modules(cuda=False)
    with patch.dict("sys.modules", mods):
        load_txt2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.enable_model_cpu_offload.assert_not_called()
    pipe.enable_vae_tiling.assert_not_called()
    pipe.vae.enable_tiling.assert_not_called()


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


def test_load_leaves_device_placement_to_the_offload_on_cuda():
    # enable_model_cpu_offload() owns placement from the moment it is called;
    # a .to("cuda") after it would make every component GPU-resident at once
    # and give the offload's memory savings straight back.
    mods, mock_pipe_cls = _mock_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        pipe = load_txt2img_pipeline()
    assert pipe is mock_pipe_cls.from_pretrained.return_value
    pipe.to.assert_not_called()


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
    mock_diffusers.StableDiffusionXLImg2ImgPipeline = mock_pipe_cls

    mock_torch = MagicMock()
    mock_torch.float32 = "float32"
    mock_torch.float16 = "float16"
    mock_torch.cuda.is_available.return_value = cuda

    return {
        "diffusers": mock_diffusers,
        "torch": mock_torch,
        "diffusers.loaders": MagicMock(),
        "diffusers.loaders.lora_pipeline": _make_lora_pipeline_mock(),
        # See _mock_modules: seal the sandbox around _apply_lora's imports.
        "transformers": MagicMock(),
        "transformers.conversion_mapping": MagicMock(),
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
    assert pipe.load_lora_into_text_encoder.call_count == 2
    pipe.fuse_lora.assert_called_once_with(lora_scale=_LORA_SCALE)


def test_load_img2img_sets_euler_ancestral_scheduler():
    mods, mock_pipe_cls = _mock_img2img_modules()
    with patch.dict("sys.modules", mods):
        load_img2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    scheduler_cls = mods["diffusers"].EulerAncestralDiscreteScheduler
    assert pipe.scheduler is scheduler_cls.from_config.return_value


def test_load_img2img_enables_offload_and_tiling_on_cuda():
    mods, mock_pipe_cls = _mock_img2img_modules(cuda=True)
    with patch.dict("sys.modules", mods):
        load_img2img_pipeline()
    pipe = mock_pipe_cls.from_pretrained.return_value
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.enable_vae_tiling.assert_called_once()


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
# make_img2img_pipeline()
# ---------------------------------------------------------------------------

def _mock_img2img_class(cuda=False):
    mock_cls = MagicMock()
    mock_diffusers = MagicMock()
    mock_diffusers.StableDiffusionXLImg2ImgPipeline = mock_cls

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda

    return {"diffusers": mock_diffusers, "torch": mock_torch}, mock_cls


def _txt2img_stub():
    txt2img = MagicMock()
    # SDXL's component set — note the second text encoder/tokenizer, which the
    # SD1.5 pipeline this replaced did not have.
    txt2img.components = {
        "vae": "vae", "text_encoder": "te1", "text_encoder_2": "te2",
        "tokenizer": "tok1", "tokenizer_2": "tok2", "unet": "unet",
        "scheduler": "sched",
    }
    return txt2img


def test_make_img2img_pipeline_reuses_txt2img_components():
    mods, mock_cls = _mock_img2img_class()
    txt2img = _txt2img_stub()
    with patch.dict("sys.modules", mods):
        pipe = make_img2img_pipeline(txt2img)
    mock_cls.assert_called_once_with(**txt2img.components)
    assert pipe is mock_cls.return_value


def test_make_img2img_pipeline_enables_offload_and_tiling_on_cuda():
    # Sharing the parent's components does NOT share the offload: `_all_hooks`
    # is per-pipeline, so a derived pipe without its own enable_model_cpu_offload
    # has a no-op maybe_free_model_hooks() and leaves its last-run component
    # (the fp32-upcast VAE) sitting in VRAM between calls.
    mods, mock_cls = _mock_img2img_class(cuda=True)
    with patch.dict("sys.modules", mods):
        pipe = make_img2img_pipeline(_txt2img_stub())
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.enable_vae_tiling.assert_called_once()


def test_make_img2img_pipeline_leaves_device_placement_to_the_offload():
    mods, _ = _mock_img2img_class(cuda=True)
    with patch.dict("sys.modules", mods):
        pipe = make_img2img_pipeline(_txt2img_stub())
    pipe.to.assert_not_called()


def test_make_img2img_pipeline_skips_offload_and_tiling_off_cuda():
    # enable_model_cpu_offload() raises without an accelerator, and the shared
    # components are already wherever the parent put them — nothing to do.
    mods, _ = _mock_img2img_class(cuda=False)
    with patch.dict("sys.modules", mods):
        pipe = make_img2img_pipeline(_txt2img_stub())
    pipe.enable_model_cpu_offload.assert_not_called()
    pipe.enable_vae_tiling.assert_not_called()
    pipe.to.assert_not_called()


# ---------------------------------------------------------------------------
# Chibi icon enhancement against the SDXL img2img pipeline (issue #68)
# ---------------------------------------------------------------------------

def test_chibi_tags_reach_the_prompt_of_an_sdxl_built_pipeline(tmp_path):
    """The chibi enhancement's ``extra_tags`` must survive the SD1.5 -> SDXL
    swap. Runs the real ``generate_sprite_img2img`` against the pipeline
    instance built from diffusers' ``StableDiffusionXLImg2ImgPipeline`` (the
    sys.modules seam the pipeline-swap tests use), so the chibi prompt is
    confirmed against the XL class rather than an anonymous pipeline mock.
    """
    from fakemon_forge.main import _CHIBI_TAGS

    mods, mock_cls = _mock_img2img_class()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mods["torch"] = mock_torch          # _make_generator's function-local import
    txt2img = MagicMock()
    txt2img.components = {"vae": "vae", "unet": "unet", "scheduler": "sched"}

    init_img = tmp_path / "sprite.png"
    _rgb_image(96, 96).save(str(init_img))
    out = tmp_path / "sprite_chibi.png"
    with patch.dict("sys.modules", mods):
        pipeline = make_img2img_pipeline(txt2img)
        pipeline.return_value.images = [_rgb_image(768, 768)]
        generate_sprite_img2img(
            "fire lizard", [], str(init_img), str(out),
            pipeline=pipeline, extra_tags=_CHIBI_TAGS, seed=7,
        )

    assert pipeline is mock_cls.return_value        # the XL class's instance
    assert pipeline.call_count == 1
    prompt = pipeline.call_args.kwargs["prompt"]
    for tag in _CHIBI_TAGS:
        assert tag in prompt
    # SDXL img2img call shape: an init image plus strength, no width/height.
    assert "image" in pipeline.call_args.kwargs
    assert "width" not in pipeline.call_args.kwargs
    assert "height" not in pipeline.call_args.kwargs
    assert Image.open(str(out)).mode == "P"


# ---------------------------------------------------------------------------
# _background_index()
# ---------------------------------------------------------------------------

def test_background_index_is_most_common_index():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    counts = {i: c for c, i in frame1.getcolors()}
    assert bg == max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# split_front_back_canvas()
# ---------------------------------------------------------------------------
# All canvases below are 200x100 RGB, background (250, 250, 250). The middle
# 20% search window is columns [80, 120) (`int(0.4 * 200)` to `int(0.6 * 200)`).
#
# Content never touches row 0 or row 99: the background colour is detected from
# the 1-px border ring, so a band bleeding into it would skew `bg` away from
# (250, 250, 250) and the tests would stop exercising what they claim to.
# `_SPLIT_ROWS` is the safe row span for a full-height-blocking band.

_SPLIT_BG = (250, 250, 250)
_SPLIT_FRONT_COLOR = (30, 60, 90)
_SPLIT_BACK_COLOR = (90, 30, 60)
_SPLIT_ROWS = (1, 98)


def _split_canvas(*bands):
    """A 200x100 background canvas with ``(x0, x1, colour)`` bands drawn on it.

    Each band spans `_SPLIT_ROWS`, so every column it covers has at least one
    non-background pixel (disqualifying it as a full-height background column)
    while the border ring stays pure background.
    """
    canvas = Image.new("RGB", (200, 100), _SPLIT_BG)
    d = ImageDraw.Draw(canvas)
    for x0, x1, color in bands:
        d.rectangle((x0, _SPLIT_ROWS[0], x1, _SPLIT_ROWS[1]), fill=color)
    return canvas


def test_split_clean_centered_gap_cuts_at_gap_centre():
    # Front square at columns 30-94 (rows 20-79), back square at columns
    # 105-170 (rows 20-79): the only full-height background run in the
    # [80, 120) window is columns 95-104, centred in the window.
    canvas = Image.new("RGB", (200, 100), _SPLIT_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle((30, 20, 94, 79), fill=_SPLIT_FRONT_COLOR)
    d.rectangle((105, 20, 170, 79), fill=_SPLIT_BACK_COLOR)
    before = canvas.tobytes()

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)

    # Front half holds the front square untouched.
    assert front_half.getpixel((60, 50)) == _SPLIT_FRONT_COLOR
    # Back half holds the back square, shifted left by the cut column.
    assert back_half.getpixel((105 - 100, 50)) == _SPLIT_BACK_COLOR
    assert back_half.getpixel((170 - 100, 50)) == _SPLIT_BACK_COLOR
    # The halves are fresh crops — the caller's canvas is untouched.
    assert canvas.tobytes() == before


def test_split_widest_run_wins_over_narrower_earlier_run():
    # Column bands across the [80, 120) window:
    #   80-95 content, 96-99 bg (narrow gap, width 4), 100-105 content,
    #   106-118 bg (wide gap, width 13), 119 content.
    # The narrow gap is both encountered first *and* closer to the literal
    # midline (column 100), yet the wide gap must still win the cut.
    canvas = _split_canvas(
        (80, 95, _SPLIT_FRONT_COLOR),
        (100, 105, _SPLIT_BACK_COLOR),
        (119, 119, _SPLIT_FRONT_COLOR),
    )

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    # Widest run is columns 106-118 (run_start=106, run_end=119); centre = 112.
    assert front_half.size == (112, 100)
    assert back_half.size == (200 - 112, 100)
    # The narrow gap's own centre (97) would have landed mid-front-square.
    assert front_half.getpixel((103, 50)) == _SPLIT_BACK_COLOR
    assert back_half.getpixel((119 - 112, 50)) == _SPLIT_FRONT_COLOR


def test_split_equal_width_runs_tie_break_to_the_leftmost():
    # Two background gaps of the same width (88-92 and 106-110): the documented
    # tie-break picks the leftmost, so the cut is that run's centre (90).
    canvas = _split_canvas(
        (80, 87, _SPLIT_FRONT_COLOR),
        (93, 105, _SPLIT_BACK_COLOR),
        (111, 119, _SPLIT_FRONT_COLOR),
    )

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, _back_half = result
    assert front_half.size == (90, 100)


def test_split_single_column_run_is_a_valid_run():
    # Exactly one background column (100) in the window; its "centre" is itself.
    canvas = _split_canvas(
        (80, 99, _SPLIT_FRONT_COLOR),
        (101, 119, _SPLIT_BACK_COLOR),
    )

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)


def test_split_entire_window_background_cuts_at_window_centre():
    # Both subjects sit well outside the [80, 120) window, so the whole window
    # is one run — no special casing, the cut is just its centre (100).
    canvas = _split_canvas(
        (20, 70, _SPLIT_FRONT_COLOR),
        (130, 180, _SPLIT_BACK_COLOR),
    )

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)
    assert front_half.getpixel((50, 50)) == _SPLIT_FRONT_COLOR
    assert back_half.getpixel((150 - 100, 50)) == _SPLIT_BACK_COLOR


def test_split_no_full_height_background_run_returns_none():
    # A single band spans columns 70-130, covering the entire [80, 120) search
    # window, so no column in it is background for its full height.
    canvas = _split_canvas((70, 130, _SPLIT_FRONT_COLOR))

    assert split_front_back_canvas(canvas) is None


def test_split_one_off_tolerance_pixel_disqualifies_the_column():
    # Same layout as the clean-gap test (gap at columns 95-104, centre 100) but
    # with a single speck at (99, 50) further than `_KEY_TOLERANCE` from the
    # background. "Full height" is strict, so column 99 is no longer background
    # and the gap splits into runs 95-98 (width 4) and 100-104 (width 5) — the
    # wider right-hand one wins, moving the cut to 102.
    canvas = _split_canvas(
        (30, 94, _SPLIT_FRONT_COLOR),
        (105, 170, _SPLIT_BACK_COLOR),
    )
    speck = (210, 250, 250)
    assert _rgb_distance(speck, _SPLIT_BG) > _KEY_TOLERANCE
    canvas.putpixel((99, 50), speck)

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, _back_half = result
    assert front_half.size == (102, 100)


def test_split_within_tolerance_noise_still_counts_as_background():
    # The mirror of the test above: a speck *within* `_KEY_TOLERANCE` (the
    # near-white noise SD actually paints) leaves the column background, so the
    # gap stays whole and the cut stays at its centre (100).
    canvas = _split_canvas(
        (30, 94, _SPLIT_FRONT_COLOR),
        (105, 170, _SPLIT_BACK_COLOR),
    )
    speck = (235, 235, 245)
    assert _rgb_distance(speck, _SPLIT_BG) <= _KEY_TOLERANCE
    canvas.putpixel((99, 50), speck)

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, _back_half = result
    assert front_half.size == (100, 100)


def test_split_detects_a_tinted_background_from_the_border_ring():
    # SD paints tints/vignettes, not pure white, so the gap is only found if the
    # background colour comes from the border ring (as `_flatten_background_to_key`
    # computes it) rather than being assumed white. This canvas's backdrop is far
    # enough from white that a hardcoded one would match no column at all.
    tint = (180, 200, 210)
    assert _rgb_distance(tint, (255, 255, 255)) > _KEY_TOLERANCE
    canvas = Image.new("RGB", (200, 100), tint)
    d = ImageDraw.Draw(canvas)
    d.rectangle((30, 20, 94, 79), fill=_SPLIT_FRONT_COLOR)
    d.rectangle((105, 20, 170, 79), fill=_SPLIT_BACK_COLOR)

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)


def test_split_degenerate_search_window_returns_none():
    # A canvas so narrow that `int(0.6 * w) <= int(0.4 * w)`: the window holds
    # zero columns, so no run can exist and there is nothing to cut at.
    canvas = Image.new("RGB", (1, 100), _SPLIT_BG)

    assert split_front_back_canvas(canvas) is None


def test_split_vignetted_background_returns_none():
    """A gradient backdrop reports "no band" rather than splitting on a bad `bg`.

    One mean colour does not describe a vignette — the ring mean lands far from
    the gap's actual pixels — so the scan would be measuring against a colour
    that matches nothing. Rejected up front, the same condition
    `_flatten_background_to_key` branches on, leaving the caller to reroll.
    """
    canvas = Image.new("RGB", (200, 100))
    px = canvas.load()
    for y in range(100):
        for x in range(200):
            # Bright at the centre, falling off towards every edge.
            fade = 1 - (abs(x - 100) / 100) * 0.5 - (abs(y - 50) / 50) * 0.2
            v = int(250 * fade)
            px[x, y] = (v, v, v)
    d = ImageDraw.Draw(canvas)
    d.rectangle((30, 20, 94, 79), fill=_SPLIT_FRONT_COLOR)
    d.rectangle((105, 20, 170, 79), fill=_SPLIT_BACK_COLOR)

    ring = _border_ring(canvas)
    assert not _border_is_uniform(ring, _detect_background(ring))
    assert split_front_back_canvas(canvas) is None


def test_split_halves_are_unequal_when_the_gap_is_off_centre():
    """The cut tracks the gap, so the halves are *not* each half the width.

    Pins the contract a caller has to honour: resizing both halves to one
    square would stretch the two sprites by different aspect ratios.
    """
    # Only full-height background run in [80, 120) is columns 81-89, so the cut
    # lands at 85 — well off the 100px midline. Bands stop short of columns 0
    # and 199 for the same reason they stop short of rows 0 and 99: the border
    # ring is where `bg` comes from.
    canvas = _split_canvas((20, 80, _SPLIT_FRONT_COLOR), (90, 180, _SPLIT_BACK_COLOR))

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert front_half.size == (85, 100)
    assert back_half.size == (115, 100)
    assert front_half.width != back_half.width


def test_split_blank_canvas_succeeds_with_two_empty_halves():
    """`None` means "no band", not "two sprites present".

    An empty canvas is all background, so the widest run is the whole window
    and the split "succeeds" into two sprite-less halves. Pins the gap a caller
    branching only on `None` has to close itself.
    """
    result = split_front_back_canvas(Image.new("RGB", (200, 100), _SPLIT_BG))
    assert result is not None
    front_half, back_half = result
    assert set(front_half.get_flattened_data()) == {_SPLIT_BG}
    assert set(back_half.get_flattened_data()) == {_SPLIT_BG}


def test_split_single_sprite_canvas_succeeds_with_one_empty_half():
    """The same gap with real content: one sprite off to the left still splits.

    Everything right of it is background, so a run is found and the back half
    comes back empty — indistinguishable from a good split by return type.
    """
    canvas = _split_canvas((30, 94, _SPLIT_FRONT_COLOR))

    result = split_front_back_canvas(canvas)
    assert result is not None
    front_half, back_half = result
    assert _SPLIT_FRONT_COLOR in set(front_half.get_flattened_data())
    assert set(back_half.get_flattened_data()) == {_SPLIT_BG}


# ---------------------------------------------------------------------------
# _split_front_back_with_retry()
# ---------------------------------------------------------------------------
# Reuses `_split_canvas` / `_SPLIT_*` above: a "clean" canvas has a full-height
# background gap in the search window, a "dirty" one has a single band
# spanning the whole window (no clean split possible).

def _clean_split_canvas(front_color=_SPLIT_FRONT_COLOR, back_color=_SPLIT_BACK_COLOR):
    return _split_canvas((30, 94, front_color), (105, 170, back_color))


def _dirty_canvas(color):
    return _split_canvas((70, 130, color))


def test_retry_clean_first_canvas_never_regenerates():
    canvas = _clean_split_canvas()

    def _regenerate():
        raise AssertionError("regenerate must not be called on a clean first split")

    front_half, back_half = _split_front_back_with_retry(canvas, _regenerate)
    assert front_half.getpixel((60, 50)) == _SPLIT_FRONT_COLOR
    assert back_half.getpixel((5, 50)) == _SPLIT_BACK_COLOR


def test_retry_falls_back_to_regenerated_canvas_when_first_has_no_split():
    dirty = _dirty_canvas(_SPLIT_FRONT_COLOR)
    clean = _clean_split_canvas()

    front_half, back_half = _split_front_back_with_retry(dirty, lambda: clean)
    assert front_half.getpixel((60, 50)) == _SPLIT_FRONT_COLOR
    assert back_half.getpixel((5, 50)) == _SPLIT_BACK_COLOR


def test_retry_naive_midline_fallback_uses_the_second_canvas_and_warns(capsys):
    # Distinct colours per canvas so the naive-split result can be traced back
    # to whichever canvas it actually came from.
    first = _dirty_canvas(_SPLIT_FRONT_COLOR)
    second = _dirty_canvas(_SPLIT_BACK_COLOR)

    front_half, back_half = _split_front_back_with_retry(first, lambda: second)
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)
    # The band (columns 70-130) straddles the naive cut at column 100; a pixel
    # from it on either side must show the SECOND canvas's colour.
    assert front_half.getpixel((90, 50)) == _SPLIT_BACK_COLOR
    assert back_half.getpixel((10, 50)) == _SPLIT_BACK_COLOR
    assert capsys.readouterr().err


def test_retry_reroll_render_failure_degrades_to_the_first_canvas_and_warns(capsys):
    """A raising `regenerate` must not cost the caller the first canvas.

    The contract is a best-effort result plus a warning, never a raise — so a
    transient failure on the second wide render (an OOM, say) falls back to a
    naive midline split of the canvas already in hand rather than discarding a
    front sprite that canvas could still have produced.
    """
    first = _dirty_canvas(_SPLIT_FRONT_COLOR)

    def _regenerate():
        raise RuntimeError("CUDA out of memory")

    front_half, back_half = _split_front_back_with_retry(first, _regenerate)
    assert front_half.size == (100, 100)
    assert back_half.size == (100, 100)
    # The band (columns 70-130) straddles the naive cut at column 100, so a
    # pixel from it on either side must show the FIRST canvas's colour.
    assert front_half.getpixel((90, 50)) == _SPLIT_FRONT_COLOR
    assert back_half.getpixel((10, 50)) == _SPLIT_FRONT_COLOR
    assert "CUDA out of memory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _fit_half_to_square()
# ---------------------------------------------------------------------------
# The content-aware cut lands wherever the gap is, so the two halves come out
# unequal widths. Squaring them by *resize* (what postprocess and
# quantize_to_reference do on their own) would stretch one and squeeze the
# other; these pin the paste-onto-a-square behaviour that replaced it.

def _squared_content(half):
    """(width, height) of the non-background content after squaring ``half``."""
    square = _fit_half_to_square(half)
    bg = _detect_background(_border_ring(square))
    columns = _content_columns(square, bg)
    assert columns is not None
    rows = _content_columns(square.transpose(Image.TRANSPOSE), bg)
    return square, (columns[1] - columns[0] + 1, rows[1] - rows[0] + 1)


def _offcentre_pair_canvas():
    """200x100 canvas whose two 40x40 squares sit either side of an off-centre gap.

    Front occupies columns 20-59, back columns 100-139; the only full-height
    background run inside the [80, 120) search window is columns 80-99, so the
    cut lands at 90 and the halves come out 90 and 110 wide — neither of them
    the 100 they will be squared to.
    """
    canvas = Image.new("RGB", (200, 100), _SPLIT_BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle((20, 30, 59, 69), fill=_SPLIT_FRONT_COLOR)
    d.rectangle((100, 30, 139, 69), fill=_SPLIT_BACK_COLOR)
    return canvas


def test_fit_half_to_square_keeps_both_halves_of_an_offcentre_split_undistorted():
    """Regression: an off-centre cut must not change either view's proportions.

    Resizing the 90px and 110px halves to 100x100 would stretch the front's
    40x40 body to ~44x40 and squeeze the back's to ~36x40, so front and back
    of one creature would come out visibly differently proportioned — worse
    geometry than the naive-midline fallback, which distorts nothing.
    """
    result = split_front_back_canvas(_offcentre_pair_canvas())
    assert result is not None
    front_half, back_half = result
    assert (front_half.width, back_half.width) == (90, 110)   # unequal, as split

    front_square, front_content = _squared_content(front_half)
    back_square, back_content = _squared_content(back_half)

    assert front_square.size == back_square.size == (100, 100)
    # Both bodies keep their drawn 40x40 shape, so both keep the same scale.
    assert front_content == (40, 40)
    assert back_content == (40, 40)


def test_fit_half_to_square_pads_a_narrow_half_and_centres_its_content():
    # Front half: 90 wide, body at columns 20-59 (centre 40). Squaring to 100
    # pads 10 columns, and centring the body on the square shifts it +10.
    front_half, _back = split_front_back_canvas(_offcentre_pair_canvas())
    square = _fit_half_to_square(front_half)

    assert square.size == (100, 100)
    columns = _content_columns(square, _SPLIT_BG)
    assert columns == (30, 69)                       # body centred on the square
    assert square.getpixel((0, 50)) == _SPLIT_BG     # padding is background
    assert square.getpixel((99, 50)) == _SPLIT_BG


def test_fit_half_to_square_crops_a_wide_half_without_losing_content():
    # Back half: 110 wide, body at columns 10-49 within the half. Centring the
    # body needs a +20 shift, so the square pads 20 columns on the left and the
    # half's trailing 30 columns (all background) fall outside the window.
    _front, back_half = split_front_back_canvas(_offcentre_pair_canvas())
    square = _fit_half_to_square(back_half)

    assert square.size == (100, 100)
    columns = _content_columns(square, _SPLIT_BG)
    assert columns == (30, 69)                       # whole body survives, centred
    assert square.getpixel((50, 50)) == _SPLIT_BACK_COLOR
    assert square.getpixel((0, 50)) == _SPLIT_BG     # padding, not cropped content


def test_fit_half_to_square_centres_an_empty_half_without_raising():
    """An empty half has no content to centre on — it centres itself instead."""
    empty = Image.new("RGB", (110, 100), _SPLIT_BG)
    square = _fit_half_to_square(empty)

    assert square.size == (100, 100)
    assert set(square.get_flattened_data()) == {_SPLIT_BG}


def test_fit_half_to_square_recentres_an_already_square_half_without_resizing():
    """An already-square half still gets its content centred — a pure
    translation, so the body keeps its size and only its position changes."""
    front_half, _back = split_front_back_canvas(_clean_split_canvas())
    assert front_half.size == (100, 100)   # centred gap -> already square
    assert _content_columns(front_half, _SPLIT_BG) == (30, 94)   # body off-centre

    square, content = _squared_content(front_half)
    assert square.size == (100, 100)
    assert content == (65, 98)                                   # 30-94 wide, unresized
    assert _content_columns(square, _SPLIT_BG) == (18, 82)       # now centred


# ---------------------------------------------------------------------------
# Empty-back-half check (pure PIL logic `generate_sprite_pair` relies on)
# ---------------------------------------------------------------------------
# The back half is treated as empty/background-only when every pixel of the
# palette-locked result decodes to index 0 (the Gen-3 contract's guaranteed
# key slot) -- exactly what `_content_bbox(back, background=0) is None` tests.

def test_empty_back_half_locks_to_all_key_index():
    front = _pp96(_sprite_rgb())
    back_raw = Image.new("RGB", (96, 96), _SPLIT_BG)  # pure background, no content

    back = quantize_to_reference(back_raw, front)
    assert _content_bbox(back, 0) is None


def test_nonempty_back_half_has_content_outside_key_index():
    front = _pp96(_sprite_rgb())
    back_raw = _sprite_rgb()  # same creature shape used to build `front`

    back = quantize_to_reference(back_raw, front)
    assert _content_bbox(back, 0) is not None


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


def test_procedural_squash_visibly_compresses_the_creature():
    # Regression (#90 follow-up): the old canvas-relative squash (h // 48 on
    # the whole frame) moved a mid-canvas creature ~2% — "basically the same
    # sprite" when flipped. The squash must be content-aware: the creature's
    # own bbox compresses by a visible fraction of ITS height, widens slightly
    # (squash-and-stretch), and keeps its feet planted.
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    b1 = _content_bbox(frame1, bg)
    out = procedural_squash(frame1)
    b2 = _content_bbox(out, bg)
    h1 = b1[3] - b1[1]
    h2 = b2[3] - b2[1]
    assert h2 == h1 - max(1, h1 // 16)   # creature-proportional, visible squash
    assert b2[3] == b1[3]                # feet stay planted
    assert (b2[2] - b2[0]) >= (b1[2] - b1[0])  # stretch: never narrower


def test_procedural_squash_all_background_frame_does_not_crash():
    frame1 = _pp96(_sprite_rgb())
    bg = _background_index(frame1)
    blank = Image.new("P", (96, 96), bg)
    blank.putpalette(frame1.getpalette())
    out = procedural_squash(blank)
    assert out.mode == "P"
    assert out.size == (96, 96)
    assert set(out.get_flattened_data()) == {bg}


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
    difference from frame 1 is the recoloured creature region — palette churn on
    a static silhouette.
    """
    img = _sprite_rgb(body=body)
    px = img.load()
    for y in range(96):
        for x in range(96):
            if px[x, y] == (40, 40, 60):
                px[x, y] = _KEY_COLOR
    return img


def test_build_frame2_recoloured_static_candidate_falls_back():
    # Regression (#90): a candidate that is frame 1's silhouette with the body
    # recoloured is pure colour flicker, not motion — 25/31 of the 2026-08-04
    # round's candidates were exactly this shape and were accepted. The gate
    # must reject it and fall back to the squash.
    frame1 = _pp96(_sprite_rgb())
    candidate = _key_background_sprite(body=(90, 160, 210))
    out = build_frame2(frame1, candidate)
    assert list(out.get_flattened_data()) == list(procedural_squash(frame1).get_flattened_data())


def test_build_frame2_structural_motion_with_low_churn_is_accepted():
    # The positive control for the structural gate: a candidate that genuinely
    # moved (a squash-like pose in frame 1's own colours, deviating from the
    # default squash the way an img2img cleanup would) passes, so the gate is
    # strict against flicker without becoming squash-only.
    frame1 = _pp96(_sprite_rgb())
    candidate = procedural_squash(frame1, amount_px=4).convert("RGB")
    out = build_frame2(frame1, candidate)
    squash = list(procedural_squash(frame1).get_flattened_data())
    assert list(out.get_flattened_data()) != squash
    assert out.getpalette() == frame1.getpalette()


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


def _midtone_creature():
    """96x96 RGB: white background, outlined body of beige tones near the key.

    Reproduces the 2026-08-04 key bleed: light beige/gray midtones (like the
    anti-aliased seams between 768-res faux pixels) sit closer to the key
    ``(200, 200, 168)`` than to some of the surviving creature centroids, so a
    key that competes in the nearest-colour mapping swallows them as
    transparency speckle inside the body.
    """
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    rng = random.Random(7)
    px = img.load()
    for y in range(24, 72):
        for x in range(24, 72):
            px[x, y] = (
                210 + rng.randint(-15, 15),
                200 + rng.randint(-15, 15),
                160 + rng.randint(-15, 15),
            )
    ImageDraw.Draw(img).rectangle((22, 22, 73, 73), outline=(30, 30, 30), width=2)
    return img


def test_quantize_gen3_never_keys_creature_midtones():
    # Regression (#90): every mon of the 2026-08-04 round carried enclosed key
    # speckle inside the body (up to ~5,800 px at 768) because creature pixels
    # nearer the key than any centroid nearest-mapped onto index 0. The key may
    # only be assigned where the flatten keyed background, never by proximity.
    out = _qg96(_midtone_creature())
    data = out.get_flattened_data()
    body = [data[y * 96 + x] for y in range(28, 68) for x in range(28, 68)]
    assert 0 not in body


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
# _is_nw_lit() — lighting gate measurement
# ---------------------------------------------------------------------------
# Real Gen-3 sprites are NW-lit (21/22 measured Ruby/Sapphire sprites pair
# near-black SE edges with lit NW edges). The gate vetoes only on CLEAR
# inversion — flat or ambiguous renders pass, because a reroll costs a full
# render and the sheet cleanup imposes the outline convention anyway.

_LIT_BG = (250, 250, 250)
_LIT_BRIGHT = (230, 180, 120)
_LIT_DARK = (60, 40, 30)
_LIT_MID = (150, 110, 80)


def _shaded_block(top_color, bottom_color):
    """A creature block whose top half is ``top_color``, bottom ``bottom_color``."""
    img = Image.new("RGB", (100, 100), _LIT_BG)
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 79, 49), fill=top_color)
    d.rectangle((20, 50, 79, 79), fill=bottom_color)
    return img


def test_nw_lit_block_passes():
    assert _is_nw_lit(_shaded_block(_LIT_BRIGHT, _LIT_DARK))


def test_se_lit_block_fails():
    """Bright SE edges against dark NW edges: the shadowed side faces the
    light — the edge signal's clear inversion."""
    assert not _is_nw_lit(_shaded_block(_LIT_DARK, _LIT_BRIGHT))


def test_flat_block_passes():
    """No lighting direction at all is not an inversion: only a clear
    wrong-side render is worth the cost of a reroll."""
    assert _is_nw_lit(_shaded_block(_LIT_MID, _LIT_MID))


def _ringed_interior_block(top_color, bottom_color):
    """A block with a uniform mid-tone edge ring, so only the INTERIOR
    carries the lighting: the edge signal ties and the verdict comes from
    the highlight/shadow mass centroids."""
    img = Image.new("RGB", (100, 100), _LIT_BG)
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 79, 79), fill=_LIT_MID)
    d.rectangle((24, 24, 75, 49), fill=top_color)
    d.rectangle((24, 50, 75, 75), fill=bottom_color)
    return img


def test_interior_highlight_on_top_passes():
    assert _is_nw_lit(_ringed_interior_block(_LIT_BRIGHT, _LIT_DARK))


def test_interior_highlight_on_bottom_fails():
    """The failure no post-pass can fix: edges neutral, but the highlight
    mass sits SE of the shadow mass."""
    assert not _is_nw_lit(_ringed_interior_block(_LIT_DARK, _LIT_BRIGHT))


def test_all_background_passes():
    """Nothing to judge must never trigger a lighting reroll on top of
    whatever went wrong first."""
    assert _is_nw_lit(Image.new("RGB", (100, 100), _LIT_BG))


def test_tiny_creature_passes():
    img = Image.new("RGB", (100, 100), _LIT_BG)
    ImageDraw.Draw(img).rectangle((48, 48, 51, 51), fill=_LIT_DARK)
    assert _is_nw_lit(img)


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


def _mixed_tile(majority, minority):
    """4x4 tile, 3:1 majority, with the minority on the row NEAREST samples.

    PIL's ``NEAREST`` reduction of 4x4 -> 1x1 reads source pixel (2, 2), so
    putting the minority colour on row 2 makes the two algorithms disagree:
    NEAREST returns the minority, k-centroid must return the majority.
    """
    img = Image.new("RGB", (4, 4), majority)
    for x in range(4):
        img.putpixel((x, 2), minority)
    return img


def test_k_centroid_mixed_tile_picks_dominant_color():
    # The property NEAREST does not have: when a tile straddles an edge the
    # majority colour wins, rather than whichever pixel the sampler lands on.
    red, blue = (255, 0, 0), (0, 0, 255)

    img = _mixed_tile(majority=red, minority=blue)
    assert img.resize((1, 1), Image.NEAREST).getpixel((0, 0)) == blue  # guard
    assert k_centroid(img, 1, 1).getpixel((0, 0)) == red

    # Flip the majority and the answer must flip with it.
    flipped = _mixed_tile(majority=blue, minority=red)
    assert flipped.resize((1, 1), Image.NEAREST).getpixel((0, 0)) == red  # guard
    assert k_centroid(flipped, 1, 1).getpixel((0, 0)) == blue


def test_k_centroid_upscale_falls_back_to_nearest():
    # Target larger than source: no source tile per output pixel, so the
    # dominant-colour path cannot run. Must still return a correctly sized
    # image built from existing colours instead of raising.
    red, blue = (255, 0, 0), (0, 0, 255)
    img = _mixed_tile(majority=red, minority=blue)

    out = k_centroid(img, 16, 16)
    assert out.size == (16, 16)
    assert out.mode == "RGB"
    assert set(out.get_flattened_data()) <= {red, blue}

    # Mixed axes (width up, height down) hit the same empty-tile crop.
    mixed = k_centroid(img, 8, 2)
    assert mixed.size == (8, 2)
    assert set(mixed.get_flattened_data()) <= {red, blue}

    # Equal size is a genuine 1x1-tile downscale, not the fallback: every
    # source pixel survives verbatim.
    same = k_centroid(img, 4, 4)
    assert same.size == (4, 4)
    assert list(same.get_flattened_data()) == list(img.get_flattened_data())


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
