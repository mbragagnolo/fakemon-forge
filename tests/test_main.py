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
        patch("fakemon_forge.main.generate_sprite")                as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img")        as m_sprite_i2i,
        patch("fakemon_forge.main.write_output", return_value=[stage_dir]) as m_write,
    ):
        yield {
            "mistral": m_mistral, "vision": m_vision, "gen": m_gen,
            "t2i": m_t2i, "i2i": m_i2i,
            "sprite": m_sprite, "sprite_i2i": m_sprite_i2i,
            "write": m_write, "stage_dir": stage_dir,
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
        patch("fakemon_forge.main.generate_sprite")           as m_sprite,
        patch("fakemon_forge.main.generate_sprite_img2img"),
        patch("fakemon_forge.main.write_output", return_value=dirs),
    ):
        yield {"sprite": m_sprite, "dirs": dirs}


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
    ctx["sprite_i2i"].assert_not_called()


def test_txt2img_sprite_called_with_sprite_prompt(ctx):
    main(["--description", "fire lizard"])
    kwargs = ctx["sprite"].call_args.kwargs
    assert kwargs["pipeline"] is not None
    assert ctx["sprite"].call_args.args[0] == _STAGE_1["sprite_prompt"]


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
    ctx["sprite_i2i"].assert_called_once()
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
    sprite_path = ctx["sprite"].call_args.args[1]
    assert sprite_path == str(ctx["stage_dir"] / "sprite.png")
