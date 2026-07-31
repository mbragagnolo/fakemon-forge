import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fakemon_forge.main import main

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
        patch("fakemon_forge.main.generate_sprite")                as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img")        as m_sprite_i2i,
        patch("fakemon_forge.main.generate_frame2")                as m_frame2,
        patch("fakemon_forge.main.generate_shiny")                 as m_shiny,
        patch("fakemon_forge.main.stitch_spritesheet")             as m_stitch,
        patch("fakemon_forge.main.generate_cry")                   as m_cry,
        patch("fakemon_forge.main.write_output", return_value=[stage_dir]) as m_write,
        patch("fakemon_forge.main.export_ini")                     as m_export,
    ):
        yield {
            "mistral": m_mistral, "vision": m_vision, "gen": m_gen,
            "t2i": m_t2i, "i2i": m_i2i, "make_i2i": m_make_i2i,
            "sprite": m_sprite, "sprite_i2i": m_sprite_i2i,
            "frame2": m_frame2, "shiny": m_shiny, "stitch": m_stitch,
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
        patch("fakemon_forge.main.describe_image",  return_value="a fire lizard"),
        patch("fakemon_forge.main.generate_fakemon", return_value=[_STAGE_1, _STAGE_2, _STAGE_3]),
        patch("fakemon_forge.main.load_txt2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.load_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.make_img2img_pipeline", return_value=MagicMock()),
        patch("fakemon_forge.main.generate_sprite")           as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img"),
        patch("fakemon_forge.main.generate_frame2")           as m_frame2,
        patch("fakemon_forge.main.generate_shiny")            as m_shiny,
        patch("fakemon_forge.main.stitch_spritesheet")        as m_stitch,
        patch("fakemon_forge.main.generate_cry")              as m_cry,
        patch("fakemon_forge.main.write_output", return_value=dirs),
        patch("fakemon_forge.main.export_ini"),
    ):
        yield {"sprite": m_sprite, "frame2": m_frame2, "shiny": m_shiny,
               "stitch": m_stitch, "cry": m_cry, "dirs": dirs}


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


def test_txt2img_path_calls_generate_sprite(ctx):
    main(["--description", "fire lizard"])
    ctx["sprite"].assert_called_once()
    ctx["sprite_i2i"].assert_called_once()   # back sprite only
    assert ctx["sprite_i2i"].call_args.kwargs["extra_tags"] == ["backside"]


def test_txt2img_sprite_called_with_user_description(ctx):
    main(["--description", "fire lizard"])
    kwargs = ctx["sprite"].call_args.kwargs
    assert kwargs["pipeline"] is not None
    assert ctx["sprite"].call_args.args[0] == "fire lizard"


def test_txt2img_vision_step_skipped(ctx):
    main(["--description", "fire lizard"])
    ctx["vision"].assert_not_called()


# ---------------------------------------------------------------------------
# img2img path (image provided)
# ---------------------------------------------------------------------------

def test_img2img_path_uses_img2img_pipeline(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["i2i"].assert_called_once()
    ctx["t2i"].assert_not_called()


def test_img2img_path_calls_generate_sprite_img2img(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    assert ctx["sprite_i2i"].call_count == 2   # front + back sprite
    ctx["sprite"].assert_not_called()


def test_img2img_vision_step_called(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    ctx["vision"].assert_called_once()


def test_img2img_vision_image_path_passed(ctx, tmp_path):
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])
    assert ctx["vision"].call_args.args[0] == str(img)


# ---------------------------------------------------------------------------
# Back sprite locked to frame 1's palette (reference_path=sprite.png)
# ---------------------------------------------------------------------------

def test_txt2img_back_sprite_reference_is_frame1(ctx):
    """The back-sprite call locks to frame 1's palette (sprite.png)."""
    main(["--description", "fire lizard"])
    back_call = ctx["sprite_i2i"].call_args   # only one img2img call in txt2img path
    assert back_call.kwargs["reference_path"] == str(ctx["stage_dir"] / "sprite.png")


def test_img2img_back_sprite_inits_from_front_sprite(ctx, tmp_path):
    """In the img2img path the back sprite inits from the generated front
    sprite — not the user's drawing, which holds no backside information
    (regression: #10). Palette reference stays frame 1 (sprite.png)."""
    img = tmp_path / "drawing.png"
    img.write_bytes(b"\x89PNG\r\n")
    main(["--image", str(img), "--description", "fire lizard"])

    # Two img2img calls: the front (no reference_path) and the back (locked).
    calls = ctx["sprite_i2i"].call_args_list
    assert len(calls) == 2
    front = [c for c in calls if c.kwargs.get("reference_path") is None]
    back = [c for c in calls if c.kwargs.get("reference_path") is not None]
    assert len(front) == 1 and len(back) == 1

    assert front[0].args[2] == str(img)   # front still seeds from the drawing
    back_call = back[0]
    assert back_call.args[2] == str(ctx["stage_dir"] / "sprite.png")   # init = front sprite
    assert back_call.kwargs["reference_path"] == str(ctx["stage_dir"] / "sprite.png")
    assert back_call.kwargs["extra_tags"] == ["backside"]
    assert back_call.kwargs["strength"] == 0.65


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


def test_img2img_frame2_written_per_stage(ctx, tmp_path):
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
