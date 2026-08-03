import re
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fakemon_forge.main import main, _CHIBI_TAGS

# ---------------------------------------------------------------------------
# Shared stage data
# ---------------------------------------------------------------------------

_STAGE_1 = {
    "name": "Flamburr", "stage": 1, "types": ["Fire"], "ability": "Blaze",
    "base_stats": {"hp": 45, "attack": 52, "defense": 43, "sp_atk": 60, "sp_def": 50, "speed": 65},
    "pokedex_entry": "A small fiery creature.", "sprite_prompt": "Fire lizard GBA pixel art",
}
_STAGE_2 = {**_STAGE_1, "name": "Flamburro", "stage": 2, "pokedex_entry": "Grows bolder."}
_STAGE_3 = {**_STAGE_1, "name": "Flamburron", "stage": 3, "pokedex_entry": "Melts rock."}

# A single form that levitates, for exercising blank=True footprints.
_STAGE_LEVITATE = {**_STAGE_1, "name": "Floatburr", "levitates": True}


# ---------------------------------------------------------------------------
# Fixture: patch every external call in main
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Yield a dict of all mocked collaborators with MISTRAL_API_KEY set."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")

    stage_dir = tmp_path / "Flamburr" / "stage1_Flamburr"
    stage_dir.mkdir(parents=True)

    with (
        patch("fakemon_forge.main.Mistral")                        as m_mistral,
        patch("fakemon_forge.main.describe_image",  return_value="a fire lizard") as m_vision,
        patch("fakemon_forge.main.generate_fakemon", return_value=[_STAGE_1])     as m_gen,
        patch("fakemon_forge.main.load_txt2img_pipeline", return_value=MagicMock()) as m_t2i,
        patch("fakemon_forge.main.load_img2img_pipeline", return_value=MagicMock()) as m_i2i,
        patch("fakemon_forge.main.make_img2img_pipeline", return_value=MagicMock()) as m_make_i2i,
        patch("fakemon_forge.main.generate_sprite_pair")            as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img")        as m_sprite_i2i,
        patch("fakemon_forge.main.generate_frame2")                as m_frame2,
        patch("fakemon_forge.main.generate_shiny")                 as m_shiny,
        patch("fakemon_forge.main.stitch_spritesheet")             as m_stitch,
        patch("fakemon_forge.main.generate_footprint")             as m_footprint,
        patch("fakemon_forge.main.generate_icon")                  as m_icon,
        patch("fakemon_forge.main.generate_cry")                   as m_cry,
        patch("fakemon_forge.main.write_output", return_value=[stage_dir]) as m_write,
        patch("fakemon_forge.main.export_ini")                     as m_export,
    ):
        yield {
            "mistral": m_mistral, "vision": m_vision, "gen": m_gen,
            "t2i": m_t2i, "i2i": m_i2i, "make_i2i": m_make_i2i,
            "sprite": m_sprite, "sprite_i2i": m_sprite_i2i,
            "frame2": m_frame2, "shiny": m_shiny, "stitch": m_stitch,
            "footprint": m_footprint,
            "icon": m_icon,
            "cry": m_cry,
            "write": m_write, "export": m_export, "stage_dir": stage_dir,
        }


@pytest.fixture
def ctx_line(tmp_path, monkeypatch):
    """Like ctx but generate_fakemon returns 3 stages."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")

    dirs = []
    for i, name in enumerate(["stage1_Flamburr", "stage2_Flamburro", "stage3_Flamburron"], 1):
        d = tmp_path / "Flamburr" / name
        d.mkdir(parents=True)
        dirs.append(d)

    with (
        patch("fakemon_forge.main.Mistral"),
        patch("fakemon_forge.main.describe_image",
              return_value="a fire lizard")               as m_vision,
        patch("fakemon_forge.main.generate_fakemon",
              return_value=[_STAGE_1, _STAGE_2, _STAGE_3]) as m_gen,
        patch("fakemon_forge.main.load_txt2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.load_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.make_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.generate_sprite_pair")      as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img"),
        patch("fakemon_forge.main.generate_frame2")           as m_frame2,
        patch("fakemon_forge.main.generate_shiny")            as m_shiny,
        patch("fakemon_forge.main.stitch_spritesheet")        as m_stitch,
        patch("fakemon_forge.main.generate_footprint")        as m_footprint,
        patch("fakemon_forge.main.generate_icon")             as m_icon,
        patch("fakemon_forge.main.generate_cry")              as m_cry,
        patch("fakemon_forge.main.write_output", return_value=dirs),
        patch("fakemon_forge.main.export_ini"),
    ):
        yield {"gen": m_gen, "vision": m_vision,
               "sprite": m_sprite, "frame2": m_frame2, "shiny": m_shiny,
               "stitch": m_stitch, "footprint": m_footprint,
               "icon": m_icon, "cry": m_cry, "dirs": dirs}


# ---------------------------------------------------------------------------
# MISTRAL_API_KEY
# ---------------------------------------------------------------------------

def test_exits_if_no_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--description", "fire lizard"])
    assert exc.value.code == 1


def test_missing_api_key_error_mentions_env_var(monkeypatch, capsys):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["--description", "fire lizard"])
    assert "MISTRAL_API_KEY" in capsys.readouterr().err


def test_api_key_passed_to_mistral_client(ctx):
    main(["--description", "fire lizard"])
    ctx["mistral"].assert_called_once_with(api_key="test-key-123")


# ---------------------------------------------------------------------------
# txt2img path (description only)
# ---------------------------------------------------------------------------

def test_txt2img_path_uses_txt2img_pipeline(ctx):
    main(["--description", "fire lizard"])
    ctx["t2i"].assert_called_once()
    ctx["i2i"].assert_not_called()


def test_txt2img_path_calls_generate_sprite_pair(ctx):
    main(["--description", "fire lizard"])
    ctx["sprite"].assert_called_once()
    args = ctx["sprite"].call_args.args
    assert args[2] == str(ctx["stage_dir"] / "sprite.png")        # front_output_path
    assert args[3] == str(ctx["stage_dir"] / "sprite_back.png")   # back_output_path
    # txt2img path now has exactly 1 img2img call per stage: chibi only (the
    # old "backside" back-sprite call site is deleted).
    calls = ctx["sprite_i2i"].call_args_list
    chibi = [c for c in calls if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert len(calls) == 1
    assert len(chibi) == 1


def test_txt2img_sprite_called_with_stage_prompt(ctx):
    main(["--description", "fire lizard"])
    kwargs = ctx["sprite"].call_args.kwargs
    assert kwargs["pipeline"] is not None
    assert ctx["sprite"].call_args.args[0] == _STAGE_1["sprite_prompt"]


def test_txt2img_vision_step_skipped(ctx):
    main(["--description", "fire lizard"])
    ctx["vision"].assert_not_called()


# ---------------------------------------------------------------------------
# --image mode (a drawing is provided). Since issue #69 this is no longer an
# img2img path: the drawing feeds the vision step only, and sprite generation
# runs on the same txt2img front+back call text-only mode uses.
# ---------------------------------------------------------------------------

def test_image_mode_uses_txt2img_pipeline(ctx, tmp_path):
    """Issue #69: --image mode now loads the same txt2img pipeline as
    text-only mode (load_img2img_pipeline is never called for the primary
    sprite call — see spec.md's Approach B)."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["t2i"].assert_called_once()
    ctx["i2i"].assert_not_called()


def test_image_mode_calls_generate_sprite_pair(ctx, tmp_path):
    """Issue #69: --image mode now produces a front+back pair via the same
    generate_sprite_pair call txt2img mode uses; generate_sprite_img2img is
    only called for the chibi enhancement."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["sprite"].assert_called_once()
    args = ctx["sprite"].call_args.args
    assert args[2] == str(ctx["stage_dir"] / "sprite.png")        # front_output_path
    assert args[3] == str(ctx["stage_dir"] / "sprite_back.png")   # back_output_path
    calls = ctx["sprite_i2i"].call_args_list
    chibi = [c for c in calls if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert len(calls) == 1
    assert len(chibi) == 1


def test_image_mode_vision_step_called(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["vision"].assert_called_once()


def test_image_mode_vision_image_path_passed(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    assert ctx["vision"].call_args.args[0] == str(img)


# ---------------------------------------------------------------------------
# --image mode produces a back-sprite pair (issue #69, fixing the regression
# tracked in an earlier slice of this same issue)
# ---------------------------------------------------------------------------

def test_image_mode_produces_back_sprite_pair(ctx, tmp_path):
    """--image mode's sprite_back.png regression (an earlier slice deleted the
    old img2img backside chain without a replacement) is fixed by routing
    --image mode through the same generate_sprite_pair call txt2img mode
    uses (spec.md's Approach B: the drawing feeds sprite generation only via
    describe_image's vision output, not via img2img on the raw pixels)."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])

    ctx["sprite"].assert_called_once()
    args = ctx["sprite"].call_args.args
    assert args[3] == str(ctx["stage_dir"] / "sprite_back.png")   # back_output_path
    calls = ctx["sprite_i2i"].call_args_list
    assert len(calls) == 1   # chibi only — no img2img call against the raw drawing


def test_image_mode_sprite_called_with_stage_prompt(ctx, tmp_path):
    """The drawing reaches sprite generation only through the LLM-authored
    sprite_prompt (vision -> combined -> generate_fakemon -> sprite_prompt),
    so --image mode passes exactly what text-only mode passes."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    args = ctx["sprite"].call_args.args
    assert args[0] == _STAGE_1["sprite_prompt"]
    assert args[1] == _STAGE_1["types"]
    assert ctx["sprite"].call_args.kwargs["pipeline"] is ctx["t2i"].return_value


def test_image_mode_back_shiny_derived_from_the_new_back_sprite(ctx, tmp_path):
    """sprite_back_shiny.png starts working for --image runs too, since the
    back sprite now exists — no change to generate_shiny required."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    back_shiny = [
        c for c in ctx["shiny"].call_args_list
        if c.args[2] == str(ctx["stage_dir"] / "sprite_back_shiny.png")
    ]
    assert len(back_shiny) == 1
    assert back_shiny[0].args[0] == str(ctx["stage_dir"] / "sprite_back.png")


# --- edge case: --image with no --description -------------------------------

def test_image_only_run_produces_the_sprite_pair(ctx, tmp_path):
    """`--image` with no `--description` is valid (cli.validate_args); the
    vision output is the whole prompt and the front+back pair is produced the
    same way."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img)])

    assert ctx["gen"].call_args.args[0] == "a fire lizard"   # vision output only
    ctx["t2i"].assert_called_once()
    ctx["i2i"].assert_not_called()
    ctx["sprite"].assert_called_once()
    args = ctx["sprite"].call_args.args
    assert args[2] == str(ctx["stage_dir"] / "sprite.png")
    assert args[3] == str(ctx["stage_dir"] / "sprite_back.png")


# --- edge case: --image in line mode ----------------------------------------

def test_image_line_mode_describes_once_and_pairs_per_stage(ctx_line, tmp_path):
    """The drawing is described once for the whole line; every stage then gets
    its own front+back pair in its own stage directory."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--mode", "line"])

    ctx_line["vision"].assert_called_once()
    assert ctx_line["sprite"].call_count == 3
    pairs = [(c.args[2], c.args[3]) for c in ctx_line["sprite"].call_args_list]
    assert pairs == [
        (str(d / "sprite.png"), str(d / "sprite_back.png")) for d in ctx_line["dirs"]
    ]
    # Each stage still gets its own seed rather than sharing one.
    seeds = [c.kwargs["seed"] for c in ctx_line["sprite"].call_args_list]
    assert len(set(seeds)) == 3


# ---------------------------------------------------------------------------
# Description combination
# ---------------------------------------------------------------------------

def test_vision_and_description_combined_for_llm(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "breathes fire"])
    combined = ctx["gen"].call_args.args[0]
    assert "a fire lizard" in combined   # vision output
    assert "breathes fire" in combined   # user description


def test_description_only_passed_to_llm(ctx):
    main(["--description", "spiky ice wolf"])
    combined = ctx["gen"].call_args.args[0]
    assert "spiky ice wolf" in combined


def test_mode_passed_to_llm(ctx):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx["gen"].call_args.args[1] == "line"


# ---------------------------------------------------------------------------
# Line mode
# ---------------------------------------------------------------------------

def test_line_mode_calls_sprite_three_times(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["sprite"].call_count == 3


# ---------------------------------------------------------------------------
# Frame 2 (two-frame front animation)
# ---------------------------------------------------------------------------

def test_txt2img_frame2_written_per_stage(ctx):
    main(["--description", "fire lizard"])
    ctx["frame2"].assert_called_once()
    kwargs = ctx["frame2"].call_args.kwargs
    args = ctx["frame2"].call_args.args
    assert args[3] == str(ctx["stage_dir"] / "sprite_frame2.png")   # output_path
    assert args[2] == str(ctx["stage_dir"] / "sprite.png")          # front sprite


def test_txt2img_frame2_uses_img2img_pipeline_and_seed(ctx):
    main(["--description", "fire lizard"])
    kwargs = ctx["frame2"].call_args.kwargs
    assert kwargs["pipeline"] is not None
    assert "seed" in kwargs


def test_image_mode_frame2_written_per_stage(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["frame2"].assert_called_once()
    args = ctx["frame2"].call_args.args
    assert args[3] == str(ctx["stage_dir"] / "sprite_frame2.png")
    assert args[2] == str(ctx["stage_dir"] / "sprite.png")


def test_frame2_shiny_written_per_stage(ctx):
    main(["--description", "fire lizard"])
    shiny_calls = ctx["shiny"].call_args_list
    frame2_shiny = [
        c for c in shiny_calls
        if c.args[2] == str(ctx["stage_dir"] / "sprite_frame2_shiny.png")
    ]
    assert len(frame2_shiny) == 1
    # reads sprite_frame2.png, keyed on stage name
    assert frame2_shiny[0].args[0] == str(ctx["stage_dir"] / "sprite_frame2.png")
    assert frame2_shiny[0].args[1] == "Flamburr"


def test_generate_shiny_called_three_times_per_stage(ctx):
    """front + back + frame2 shinies."""
    main(["--description", "fire lizard"])
    assert ctx["shiny"].call_count == 3


def test_line_mode_frame2_called_three_times(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["frame2"].call_count == 3
    assert ctx_line["shiny"].call_count == 9   # 3 shinies x 3 stages


def test_frame2_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["frame2"].side_effect = RuntimeError("frame2 crash")
    main(["--description", "fire lizard"])   # must not raise
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "Flamburr" in err
    # a frame-2 failure must NOT skip the rest of the stage: the front shiny
    # (a later block in the same stage) still runs.
    front_shiny = [
        c for c in ctx["shiny"].call_args_list
        if c.args[2] == str(ctx["stage_dir"] / "sprite_shiny.png")
    ]
    assert len(front_shiny) == 1


# ---------------------------------------------------------------------------
# Blank / corrupt sprite warning
# ---------------------------------------------------------------------------

def test_sprite_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["sprite"].side_effect = RuntimeError("pipeline crash")
    main(["--description", "fire lizard"])   # must not raise
    assert "Warning" in capsys.readouterr().err


def test_sprite_failure_warning_includes_name(ctx, capsys):
    ctx["sprite"].side_effect = RuntimeError("pipeline crash")
    main(["--description", "fire lizard"])
    assert "Flamburr" in capsys.readouterr().err


def test_front_sprite_failure_skips_chibi_and_icon(ctx):
    """When the front sprite fails the stage continues past the icon block, so
    neither the chibi render nor the icon runs against a missing sprite.png."""
    ctx["sprite"].side_effect = RuntimeError("pipeline crash")
    main(["--description", "fire lizard"])
    # No chibi img2img render is attempted for the failed stage.
    chibi = [c for c in ctx["sprite_i2i"].call_args_list
             if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert chibi == []
    ctx["icon"].assert_not_called()


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def test_write_output_called_with_stages(ctx):
    main(["--description", "fire lizard"])
    ctx["write"].assert_called_once()
    assert ctx["write"].call_args.args[0] == [_STAGE_1]


def test_sprite_saved_inside_stage_dir(ctx):
    main(["--description", "fire lizard"])
    sprite_path = ctx["sprite"].call_args.args[2]
    assert sprite_path == str(ctx["stage_dir"] / "sprite.png")


# ---------------------------------------------------------------------------
# Tier passthrough
# ---------------------------------------------------------------------------

def test_tier_defaults_to_standard_in_llm_call(ctx):
    main(["--description", "fire lizard"])
    assert ctx["gen"].call_args.kwargs.get("tier", "standard") == "standard"


def test_tier_pseudo_passed_to_llm(ctx):
    main(["--description", "fire lizard", "--tier", "pseudo", "--mode", "line"])
    assert ctx["gen"].call_args.kwargs["tier"] == "pseudo"


def test_tier_legendary_passed_to_llm(ctx):
    main(["--description", "fire lizard", "--tier", "legendary"])
    assert ctx["gen"].call_args.kwargs["tier"] == "legendary"


# ---------------------------------------------------------------------------
# Party-menu icon (sprite_small.png)
# ---------------------------------------------------------------------------

def test_icon_generated_once_per_stage(ctx):
    main(["--description", "fire lizard"])
    ctx["icon"].assert_called_once()
    # Happy path: the icon is now derived from the chibi render, not sprite.png.
    assert ctx["icon"].call_args.args == (
        str(ctx["stage_dir"] / "sprite_chibi.png"),
        str(ctx["stage_dir"] / "sprite_small.png"),
    )


def test_chibi_render_feeds_the_icon(ctx):
    """Happy path: a chibi img2img render is produced from sprite.png and its
    output feeds generate_icon."""
    main(["--description", "fire lizard"])

    calls = ctx["sprite_i2i"].call_args_list
    chibi = [c for c in calls if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert len(chibi) == 1
    chibi_call = chibi[0]
    assert chibi_call.args[2] == str(ctx["stage_dir"] / "sprite.png")        # init image
    assert chibi_call.args[3] == str(ctx["stage_dir"] / "sprite_chibi.png")  # output
    assert chibi_call.kwargs.get("reference_path") is None                   # own palette
    assert "seed" in chibi_call.kwargs

    ctx["icon"].assert_called_once()
    assert ctx["icon"].call_args.args == (
        str(ctx["stage_dir"] / "sprite_chibi.png"),
        str(ctx["stage_dir"] / "sprite_small.png"),
    )


def test_chibi_render_failure_falls_back_to_plain_downscale(ctx, capsys):
    """If the chibi img2img render raises, the icon is built from sprite.png
    (plain downscale), silently, and the stage keeps going."""
    def _side_effect(*args, **kwargs):
        # Fail only the chibi render (output path ends in sprite_chibi.png).
        if args[3].endswith("sprite_chibi.png"):
            raise RuntimeError("chibi crash")
        return MagicMock()

    ctx["sprite_i2i"].side_effect = _side_effect
    main(["--description", "fire lizard"])   # must not raise

    ctx["icon"].assert_called_once()
    assert ctx["icon"].call_args.args == (
        str(ctx["stage_dir"] / "sprite.png"),
        str(ctx["stage_dir"] / "sprite_small.png"),
    )
    # A failed enhancement is silent — no icon warning printed.
    assert "icon generation failed" not in capsys.readouterr().err
    # The stage does not abort: the spritesheet is still stitched.
    ctx["stitch"].assert_called_once()


def test_chibi_render_uses_the_txt2img_derived_img2img_pipeline(ctx):
    """Issue #68: the chibi pass runs on the SDXL img2img pipeline
    make_img2img_pipeline builds from the txt2img components — not on the
    txt2img pipeline itself, and not on a separately loaded one."""
    main(["--description", "fire lizard"])

    chibi = [c for c in ctx["sprite_i2i"].call_args_list
             if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert len(chibi) == 1
    assert chibi[0].kwargs["pipeline"] is ctx["make_i2i"].return_value


def test_chibi_render_uses_the_txt2img_derived_img2img_pipeline_in_image_mode(ctx, tmp_path):
    """Issue #69: --image mode now loads only the txt2img pipeline (same as
    text-only mode), so the chibi pass runs on make_img2img_pipeline's
    derived pipeline, not a separately loaded img2img one."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])

    chibi = [c for c in ctx["sprite_i2i"].call_args_list
             if c.kwargs.get("extra_tags") == _CHIBI_TAGS]
    assert len(chibi) == 1
    assert chibi[0].kwargs["pipeline"] is ctx["make_i2i"].return_value


def test_icon_generated_three_times_in_line_mode(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["icon"].call_count == 3


def test_icon_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["icon"].side_effect = RuntimeError("icon crash")
    main(["--description", "fire lizard"])   # must not raise
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "Flamburr" in err
    # an icon failure must NOT skip the rest of the stage: a later block in the
    # same stage still runs (the spritesheet is still stitched).
    ctx["stitch"].assert_called_once()


def test_icon_not_added_to_spritesheet_layout():
    from fakemon_forge.sprites import _SHEET_LAYOUT
    names = {name for name, *_ in _SHEET_LAYOUT}
    assert len(_SHEET_LAYOUT) == 6
    assert "sprite_small.png" not in names
    # The intermediate chibi render is likewise never stitched into the sheet.
    assert "sprite_chibi.png" not in names


# ---------------------------------------------------------------------------
# Spritesheet stitching
# ---------------------------------------------------------------------------

def test_spritesheet_stitched_once_at_default_cell_size(ctx):
    main(["--description", "fire lizard"])
    calls = ctx["stitch"].call_args_list
    assert len(calls) == 1   # one sheet: the 64-cell GBA deliverable
    assert calls[0].args == (ctx["stage_dir"], str(ctx["stage_dir"] / "spritesheet.png"))
    assert calls[0].kwargs == {}


def test_spritesheet_stitched_per_stage_in_line_mode(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["stitch"].call_count == 3   # one sheet per stage


def test_spritesheet_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["stitch"].side_effect = RuntimeError("stitch crash")
    main(["--description", "fire lizard"])   # must not raise
    err = capsys.readouterr().err
    assert "Warning" in err and "Flamburr" in err


# ---------------------------------------------------------------------------
# Footprint generation
# ---------------------------------------------------------------------------

def test_footprint_generated_once_per_single_stage(ctx):
    main(["--description", "fire lizard"])
    ctx["footprint"].assert_called_once()


def test_footprint_generated_once_per_stage_in_line_mode(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["footprint"].call_count == 3


def test_footprint_paths_and_types(ctx):
    main(["--description", "fire lizard"])
    call = ctx["footprint"].call_args
    assert call.args[0] == str(ctx["stage_dir"] / "sprite.png")     # sprite_path
    assert call.args[1] == str(ctx["stage_dir"] / "footprint.png")  # output_path
    assert call.kwargs["types"] == ["Fire"]


def test_footprint_blank_false_when_levitates_missing(ctx):
    main(["--description", "fire lizard"])
    assert ctx["footprint"].call_args.kwargs["blank"] is False


def test_footprint_blank_true_when_levitates(ctx):
    ctx["gen"].return_value = [_STAGE_LEVITATE]
    main(["--description", "fire lizard"])
    assert ctx["footprint"].call_args.kwargs["blank"] is True


def test_footprint_size_fraction_single_form_is_point_nine(ctx):
    main(["--description", "fire lizard"])
    assert ctx["footprint"].call_args.kwargs["size_fraction"] == 0.9


def test_footprint_size_fraction_mapping_across_line(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    fractions = [c.kwargs["size_fraction"] for c in ctx_line["footprint"].call_args_list]
    assert fractions == [0.6, 0.75, 0.9]


def test_footprint_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["footprint"].side_effect = RuntimeError("footprint crash")
    main(["--description", "fire lizard"])   # must not raise
    err = capsys.readouterr().err
    assert "Warning: footprint generation failed for Flamburr" in err
    # The run still proceeds to the export_ini loop / normal completion.
    ctx["export"].assert_called_once()


# ---------------------------------------------------------------------------
# Cry generation (cry.wav per stage)
# ---------------------------------------------------------------------------

def test_cry_generated_once_per_stage(ctx):
    main(["--description", "fire lizard"])
    ctx["cry"].assert_called_once()
    args = ctx["cry"].call_args.args
    assert args[0] == "Flamburr"                                    # line_name = stage 1's name
    assert args[1] == 1                                             # stage int
    assert args[2] == ["Fire"]                                      # types
    assert args[3] == str(ctx["stage_dir"] / "cry.wav")            # output_path


def test_cry_generated_per_stage_in_line_mode(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line["cry"].call_count == 3
    calls = ctx_line["cry"].call_args_list
    for call, stage_dir, stage_int in zip(calls, ctx_line["dirs"], (1, 2, 3)):
        assert call.args[0] == "Flamburr"                          # whole line shares stage 1's name
        assert call.args[1] == stage_int
        assert call.args[3] == str(stage_dir / "cry.wav")


def test_cry_generated_even_when_sprite_fails(ctx):
    """Audio does not depend on the images: a sprite failure (which hits the
    sprite block's `continue`) must not skip cry generation."""
    ctx["sprite"].side_effect = RuntimeError("pipeline crash")
    main(["--description", "fire lizard"])   # must not raise
    ctx["cry"].assert_called_once()


def test_cry_failure_warns_but_does_not_exit(ctx, capsys):
    ctx["cry"].side_effect = RuntimeError("cry crash")
    main(["--description", "fire lizard"])   # must not raise
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "Flamburr" in err
    # cry failure is isolated: the sprite block still runs afterward.
    ctx["sprite"].assert_called_once()


# ---------------------------------------------------------------------------
# --stages wiring (#59)
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx_two(tmp_path, monkeypatch):
    """Like ctx_line but a 2-stage line."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")

    dirs = []
    for name in ["stage1_Flamburr", "stage2_Flamburro"]:
        d = tmp_path / "Flamburr" / name
        d.mkdir(parents=True)
        dirs.append(d)

    with (
        patch("fakemon_forge.main.Mistral"),
        patch("fakemon_forge.main.describe_image", return_value="a fire lizard"),
        patch("fakemon_forge.main.generate_fakemon",
              return_value=[_STAGE_1, _STAGE_2]) as m_gen,
        patch("fakemon_forge.main.load_txt2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.load_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.make_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.generate_sprite_pair")      as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img"),
        patch("fakemon_forge.main.generate_frame2"),
        patch("fakemon_forge.main.generate_shiny"),
        patch("fakemon_forge.main.stitch_spritesheet"),
        patch("fakemon_forge.main.generate_footprint")        as m_footprint,
        patch("fakemon_forge.main.generate_icon")             as m_icon,
        patch("fakemon_forge.main.generate_cry")              as m_cry,
        patch("fakemon_forge.main.write_output", return_value=dirs) as m_write,
        patch("fakemon_forge.main.export_ini"),
    ):
        yield {"gen": m_gen, "sprite": m_sprite, "footprint": m_footprint,
               "icon": m_icon, "cry": m_cry, "write": m_write, "dirs": dirs}


def _stages_kwarg(mock_gen):
    return mock_gen.call_args.kwargs["stages"]


# --- the count reaches the generator -----------------------------------------

def test_stages_two_reaches_generate_fakemon(ctx_two):
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    assert _stages_kwarg(ctx_two["gen"]) == 2


def test_stages_three_reaches_generate_fakemon(ctx_line):
    main(["--description", "fire lizard", "--mode", "line", "--stages", "3"])
    assert _stages_kwarg(ctx_line["gen"]) == 3


def test_omitting_stages_sends_three(ctx_line):
    """The default must survive the trip through main, not be dropped."""
    main(["--description", "fire lizard", "--mode", "line"])
    assert _stages_kwarg(ctx_line["gen"]) == 3


def test_single_mode_still_sends_the_default(ctx):
    main(["--description", "fire lizard"])
    assert _stages_kwarg(ctx["gen"]) == 3


def test_stages_count_is_not_confused_with_the_returned_list(ctx_two):
    """`args.stages` is a count; main's local holds the returned stage dicts.
    Passing the list here would make _size_defaults raise on an unhashable
    key -- the same shadowing that bit generator.py in task 10."""
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    assert isinstance(_stages_kwarg(ctx_two["gen"]), int)


# --- the run is sized by what came back --------------------------------------

def test_two_stage_run_writes_two_stage_dirs(ctx_two):
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    written = ctx_two["write"].call_args.args[0]
    assert len(written) == 2
    assert [s["stage"] for s in written] == [1, 2]


def test_two_stage_run_creates_no_third_stage_dir(ctx_two):
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    names = [d.name for d in ctx_two["dirs"]]
    assert not any(n.startswith("stage3_") for n in names)
    # The injector filters directories that don't match stage<digits>_, so a
    # branched-style name would be silently dropped rather than rejected.
    assert all(re.fullmatch(r"stage\d+_.+", n) for n in names)


@pytest.mark.parametrize("asset", ["sprite", "icon", "cry", "footprint"])
def test_two_stage_run_generates_each_asset_twice(ctx_two, asset):
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    assert ctx_two[asset].call_count == 2


@pytest.mark.parametrize("asset", ["sprite", "icon", "cry", "footprint"])
def test_three_stage_run_generates_each_asset_three_times(ctx_line, asset):
    main(["--description", "fire lizard", "--mode", "line"])
    assert ctx_line[asset].call_count == 3


# --- footprint scaling across a 2-stage line ---------------------------------

def _size_fractions(mock_footprint):
    return [c.kwargs["size_fraction"] for c in mock_footprint.call_args_list]


def test_two_stage_footprints_mirror_the_three_stage_endpoints(ctx_two):
    """A 2-stage line takes the first and last fractions, skipping the middle
    -- the same shape as its height/weight defaults, where the final form
    takes the stage-3 row."""
    main(["--description", "fire lizard", "--mode", "line", "--stages", "2"])
    assert _size_fractions(ctx_two["footprint"]) == [0.6, 0.9]


def test_three_stage_footprint_scaling_is_unchanged(ctx_line):
    main(["--description", "fire lizard", "--mode", "line"])
    assert _size_fractions(ctx_line["footprint"]) == [0.6, 0.75, 0.9]


def test_single_form_footprint_is_full_size(ctx):
    main(["--description", "fire lizard"])
    assert _size_fractions(ctx["footprint"]) == [0.9]
