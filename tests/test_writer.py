import json
import re
import shlex
import pytest
from pathlib import Path
from unittest.mock import patch

import fakemon_forge.writer as writer
from fakemon_forge.cli import parse_args, validate_args
from fakemon_forge.writer import write_output

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STAGE_1 = {
    "name": "Flamburr",
    "stage": 1,
    "types": ["Fire"],
    "ability": "Blaze",
    "base_stats": {
        "hp": 45, "attack": 52, "defense": 43,
        "sp_atk": 60, "sp_def": 50, "speed": 65,
    },
    "pokedex_entry": "A small fiery creature with a burning tail tip.",
    "sprite_prompt": "A small fire lizard, GBA pixel art, white background",
}

_STAGE_2 = {
    **_STAGE_1,
    "name": "Flamburro",
    "stage": 2,
    "pokedex_entry": "Flamburro grows bolder, its flames intensifying.",
    "sprite_prompt": "A medium fire lizard, more muscular, GBA pixel art",
}

_STAGE_3 = {
    **_STAGE_1,
    "name": "Flamburron",
    "stage": 3,
    "pokedex_entry": "Flamburron's inferno can melt solid rock.",
    "sprite_prompt": "A large fire dragon, imposing, GBA pixel art",
}

_SINGLE = [_STAGE_1]
_LINE   = [_STAGE_1, _STAGE_2, _STAGE_3]

_STATS_KEYS = {"name", "stage", "types", "ability", "base_stats"}
_LLM_ONLY   = {"pokedex_entry", "sprite_prompt"}

# Default run_info used by tests that don't care about run.json's contents —
# just something write_output can build a manifest from.
_RUN_INFO = {
    "description": "a fire lizard with blue flames",
    "image": None,
    "vision_description": "",
    "mode": "single",
    "tier": "standard",
    "requested_stages": None,
}

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

def test_creates_top_level_fakemon_dir(tmp_path):
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert (tmp_path / "Flamburr").is_dir()


def test_creates_stage1_subdir(tmp_path):
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert (tmp_path / "Flamburr" / "stage1_Flamburr").is_dir()


def test_line_mode_creates_three_stage_dirs(tmp_path):
    write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    root = tmp_path / "Flamburr"
    assert (root / "stage1_Flamburr").is_dir()
    assert (root / "stage2_Flamburro").is_dir()
    assert (root / "stage3_Flamburron").is_dir()


def test_returns_list_of_stage_dirs(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert len(dirs) == 1
    assert dirs[0] == tmp_path / "Flamburr" / "stage1_Flamburr"


def test_returns_three_dirs_for_line(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    assert len(dirs) == 3


def test_returned_dirs_match_stage_names(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    assert dirs[0].name == "stage1_Flamburr"
    assert dirs[1].name == "stage2_Flamburro"
    assert dirs[2].name == "stage3_Flamburron"


# ---------------------------------------------------------------------------
# stats.json
# ---------------------------------------------------------------------------

def test_creates_stats_json(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert (dirs[0] / "stats.json").exists()


def test_stats_json_has_required_fields(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert _STATS_KEYS <= set(data.keys())


def test_stats_json_excludes_llm_only_fields(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert not (_LLM_ONLY & set(data.keys()))


def test_stats_json_values_are_correct(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["name"] == "Flamburr"
    assert data["stage"] == 1
    assert data["types"] == ["Fire"]
    assert data["ability"] == "Blaze"
    assert data["base_stats"]["hp"] == 45


def test_stats_json_is_valid_json(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    raw = (dirs[0] / "stats.json").read_text(encoding="utf-8")
    json.loads(raw)  # must not raise


# ---------------------------------------------------------------------------
# entry.md
# ---------------------------------------------------------------------------

def test_creates_entry_md(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert (dirs[0] / "entry.md").exists()


def test_entry_md_contains_pokedex_text(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    text = (dirs[0] / "entry.md").read_text(encoding="utf-8")
    assert "A small fiery creature with a burning tail tip." in text


def test_each_stage_has_its_own_entry(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    assert "Flamburro grows bolder" in (dirs[1] / "entry.md").read_text()
    assert "Flamburron's inferno"   in (dirs[2] / "entry.md").read_text()


# ---------------------------------------------------------------------------
# Name collision handling
# ---------------------------------------------------------------------------

def test_collision_appends_suffix(tmp_path):
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))   # creates Flamburr/
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))  # should create Flamburr_2/
    assert dirs[0].parent.name == "Flamburr_2"


def test_collision_increments_suffix(tmp_path):
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))   # Flamburr/
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))   # Flamburr_2/
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))  # Flamburr_3/
    assert dirs[0].parent.name == "Flamburr_3"


def test_no_collision_no_suffix(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert dirs[0].parent.name == "Flamburr"


# ---------------------------------------------------------------------------
# levitates flag in stats.json
# ---------------------------------------------------------------------------

def test_stats_json_persists_levitates_true(tmp_path):
    stage = {**_STAGE_1, "levitates": True}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["levitates"] is True


def test_stats_json_persists_levitates_false(tmp_path):
    stage = {**_STAGE_1, "levitates": False}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["levitates"] is False


def test_stats_json_defaults_missing_levitates_to_false(tmp_path):
    # _STAGE_1 has no "levitates" key.
    assert "levitates" not in _STAGE_1
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["levitates"] is False


def test_stats_json_still_excludes_llm_only_with_levitates(tmp_path):
    stage = {**_STAGE_1, "levitates": True}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert not (_LLM_ONLY & set(data.keys()))


# ---------------------------------------------------------------------------
# height_dm / weight_hg in stats.json
# ---------------------------------------------------------------------------

def test_stats_json_persists_height_dm_and_weight_hg(tmp_path):
    stage = {**_STAGE_1, "height_dm": 12, "weight_hg": 345}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["height_dm"] == 12
    assert data["weight_hg"] == 345


def test_stats_json_defaults_missing_height_dm_to_five(tmp_path):
    assert "height_dm" not in _STAGE_1
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["height_dm"] == 5


def test_stats_json_defaults_missing_weight_hg_to_thirty(tmp_path):
    assert "weight_hg" not in _STAGE_1
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["weight_hg"] == 30


def test_stats_json_does_not_reclamp_out_of_range_values(tmp_path):
    """Clamping is _normalize's job; the writer trusts its input like every
    other field, so a dict that bypassed _normalize is persisted as-is."""
    stage = {**_STAGE_1, "height_dm": 50000, "weight_hg": 0}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["height_dm"] == 50000
    assert data["weight_hg"] == 0


def test_stats_json_defaults_height_and_weight_per_stage(tmp_path):
    """Writer defaults are flat 5/30 — the stage/tier-scaled table is
    generator-only, so every stage of a line gets the same fallback."""
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    for stage_dir in dirs:
        data = json.loads((stage_dir / "stats.json").read_text())
        assert (data["height_dm"], data["weight_hg"]) == (5, 30)


# ---------------------------------------------------------------------------
# abilities_gen3 in stats.json
# ---------------------------------------------------------------------------

def test_stats_json_persists_abilities_gen3(tmp_path):
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze", "Flash Fire"]}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["abilities_gen3"] == ["Blaze", "Flash Fire"]


def test_stats_json_defaults_missing_abilities_gen3_to_empty_list(tmp_path):
    assert "abilities_gen3" not in _STAGE_1
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["abilities_gen3"] == []


def test_stats_json_still_excludes_llm_only_with_abilities_gen3(tmp_path):
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze"]}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert not (_LLM_ONLY & set(data.keys()))


# ---------------------------------------------------------------------------
# category in stats.json
# ---------------------------------------------------------------------------

def test_stats_json_persists_category(tmp_path):
    stage = {**_STAGE_1, "category": "SEED"}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["category"] == "SEED"


def test_stats_json_defaults_missing_category_to_empty_string(tmp_path):
    assert "category" not in _STAGE_1
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["category"] == ""


def test_stats_json_still_excludes_llm_only_with_category(tmp_path):
    stage = {**_STAGE_1, "category": "SEED"}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert not (_LLM_ONLY & set(data.keys()))


# ---------------------------------------------------------------------------
# run.json — location & validity
# ---------------------------------------------------------------------------

def _read_run_json(dirs):
    return json.loads((dirs[0].parent / "run.json").read_text(encoding="utf-8"))


def test_run_json_created_at_fakemon_dir_root(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert (dirs[0].parent / "run.json").exists()


def test_run_json_not_created_inside_stage_dir(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert not (dirs[0] / "run.json").exists()


def test_run_json_is_valid_json(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    _read_run_json(dirs)  # must not raise


def test_run_json_written_into_collision_resolved_dir(tmp_path):
    write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))          # Flamburr/
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))   # Flamburr_2/
    assert dirs[0].parent.name == "Flamburr_2"
    assert (tmp_path / "Flamburr_2" / "run.json").exists()


# ---------------------------------------------------------------------------
# run.json — description / image / vision_description
# ---------------------------------------------------------------------------

def test_run_json_records_description(tmp_path):
    run_info = {**_RUN_INFO, "description": "a fire lizard with blue flames"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["description"] == "a fire lizard with blue flames"


def test_run_json_description_null_when_not_given(tmp_path):
    run_info = {**_RUN_INFO, "description": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["description"] is None


def test_run_json_records_image_path(tmp_path):
    run_info = {**_RUN_INFO, "image": "scan.png", "description": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["image"] == "scan.png"


def test_run_json_image_null_when_not_given(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["image"] is None


def test_run_json_records_vision_description_when_image_given(tmp_path):
    run_info = {**_RUN_INFO, "image": "scan.png", "vision_description": "a fire lizard"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["vision_description"] == "a fire lizard"


def test_run_json_vision_description_empty_when_no_image(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["vision_description"] == ""


def test_run_json_both_description_and_image_recorded_independently(tmp_path):
    run_info = {
        **_RUN_INFO,
        "description": "a fire lizard with blue flames",
        "image": "scan.png",
        "vision_description": "a small reptile sketch",
    }
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    data = _read_run_json(dirs)
    assert data["description"] == "a fire lizard with blue flames"
    assert data["image"] == "scan.png"
    assert data["vision_description"] == "a small reptile sketch"


# ---------------------------------------------------------------------------
# run.json — mode / tier / requested_stages
# ---------------------------------------------------------------------------

def test_run_json_records_mode(tmp_path):
    run_info = {**_RUN_INFO, "mode": "line", "requested_stages": 3}
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["mode"] == "line"


def test_run_json_records_tier(tmp_path):
    run_info = {**_RUN_INFO, "tier": "legendary"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["tier"] == "legendary"


def test_run_json_requested_stages_recorded_for_line_mode(tmp_path):
    run_info = {**_RUN_INFO, "mode": "line", "requested_stages": 3}
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["requested_stages"] == 3


def test_run_json_requested_stages_null_for_single_mode(tmp_path):
    run_info = {**_RUN_INFO, "mode": "single", "requested_stages": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["requested_stages"] is None


def test_run_json_requested_stages_can_diverge_from_generated_stages(tmp_path):
    """generate_fakemon returning fewer/more stages than requested is an
    existing possible divergence -- run.json reflects both independently."""
    run_info = {**_RUN_INFO, "mode": "line", "requested_stages": 3}
    dirs = write_output([_STAGE_1, _STAGE_2], run_info, base_dir=str(tmp_path))
    data = _read_run_json(dirs)
    assert data["requested_stages"] == 3
    assert len(data["generated_stages"]) == 2


# ---------------------------------------------------------------------------
# run.json — timestamp
# ---------------------------------------------------------------------------

def test_run_json_timestamp_is_iso8601_with_utc_offset(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    timestamp = _read_run_json(dirs)["timestamp"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+00:00|Z)", timestamp)


# ---------------------------------------------------------------------------
# run.json — package_version
# ---------------------------------------------------------------------------

def test_run_json_package_version_falls_back_to_pyproject_toml(tmp_path):
    """In this repo's own checkout fakemon-forge is not pip-installed (per
    CLAUDE.md), so resolution falls through to parsing pyproject.toml."""
    pyproject_text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    expected = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_text).group(1)
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["package_version"] == expected


def test_run_json_package_version_uses_installed_metadata_when_available(tmp_path):
    with patch("fakemon_forge.writer.importlib.metadata.version", return_value="9.9.9"):
        dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["package_version"] == "9.9.9"


def test_run_json_package_version_unknown_when_both_lookups_fail(tmp_path):
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    out_dir = tmp_path / "out"
    with (
        patch("fakemon_forge.writer.importlib.metadata.version",
              side_effect=Exception("not installed")),
        patch.object(writer, "_REPO_ROOT", fake_repo),
    ):
        dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(out_dir))
    assert _read_run_json(dirs)["package_version"] == "unknown"


# ---------------------------------------------------------------------------
# run.json — git_sha
# ---------------------------------------------------------------------------

def test_run_json_git_sha_resolves_a_real_short_sha(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert re.fullmatch(r"[0-9a-f]{4,40}", _read_run_json(dirs)["git_sha"])


def test_run_json_git_sha_unknown_when_git_not_on_path(tmp_path):
    with patch("fakemon_forge.writer.subprocess.run",
               side_effect=FileNotFoundError("no such file: git")):
        dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["git_sha"] == "unknown"


def test_run_json_git_sha_unknown_when_git_command_fails(tmp_path):
    import subprocess
    with patch("fakemon_forge.writer.subprocess.run",
               side_effect=subprocess.CalledProcessError(128, ["git"])):
        dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["git_sha"] == "unknown"


# ---------------------------------------------------------------------------
# run.json — rerun_command
# ---------------------------------------------------------------------------

def test_rerun_command_includes_description_only(tmp_path):
    run_info = {**_RUN_INFO, "description": "a fire lizard", "image": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    cmd = _read_run_json(dirs)["rerun_command"]
    assert "--description 'a fire lizard'" in cmd
    assert "--image" not in cmd


def test_rerun_command_includes_image_only(tmp_path):
    run_info = {**_RUN_INFO, "description": None, "image": "scan.png"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    cmd = _read_run_json(dirs)["rerun_command"]
    assert "--image scan.png" in cmd
    assert "--description" not in cmd


def test_rerun_command_includes_both_when_both_given(tmp_path):
    run_info = {**_RUN_INFO, "description": "a fire lizard", "image": "scan.png"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    cmd = _read_run_json(dirs)["rerun_command"]
    assert "--image scan.png" in cmd
    assert "--description 'a fire lizard'" in cmd


def test_rerun_command_omits_stages_for_single_mode(tmp_path):
    run_info = {**_RUN_INFO, "mode": "single", "requested_stages": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    assert "--stages" not in _read_run_json(dirs)["rerun_command"]


def test_rerun_command_includes_stages_for_line_mode(tmp_path):
    run_info = {**_RUN_INFO, "mode": "line", "requested_stages": 3}
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    assert "--stages 3" in _read_run_json(dirs)["rerun_command"]


def test_rerun_command_includes_stages_even_when_default_was_implicit(tmp_path):
    """requested_stages holds the parser's default when --stages wasn't
    passed explicitly; rerun_command still spells it out so the recorded
    command stays reproducible if the default ever changes."""
    run_info = {**_RUN_INFO, "mode": "line", "requested_stages": 3}
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    assert "--stages 3" in _read_run_json(dirs)["rerun_command"]


def test_rerun_command_quotes_special_characters(tmp_path):
    raw_description = "a \"fire\" lizard $(rm -rf /) `whoami`"
    run_info = {**_RUN_INFO, "description": raw_description}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    cmd = _read_run_json(dirs)["rerun_command"]
    # shlex.quote wraps the whole value in single quotes, which neutralizes
    # $(...) and `...` for a POSIX shell even though the raw text still
    # appears as a substring — round-tripping through shlex.split is the
    # actual safety check.
    tokens = shlex.split(cmd)
    assert raw_description in tokens


def test_rerun_command_parses_back_into_the_same_line_request(tmp_path):
    """"Ready to paste" means the CLI accepts it: split the recorded command
    and it must parse *and validate* back into the run's own inputs."""
    run_info = {
        **_RUN_INFO, "mode": "line", "requested_stages": 2,
        "description": 'a "fire" lizard $(whoami)', "tier": "standard",
    }
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    tokens = shlex.split(_read_run_json(dirs)["rerun_command"])

    assert tokens[0] == "fakemon-forge"
    args = parse_args(tokens[1:])
    validate_args(args)  # must not sys.exit
    assert args.description == 'a "fire" lizard $(whoami)'
    assert (args.mode, args.tier, args.stages) == ("line", "standard", 2)


def test_rerun_command_parses_back_into_the_same_single_request(tmp_path):
    """Single mode is where omitting --stages is load-bearing: cli.validate_args
    exits on `--stages` with `--mode single`, so a manifest that spelled the
    ignored default out would hand back an unrunnable command."""
    run_info = {**_RUN_INFO, "mode": "single", "tier": "legendary",
                "requested_stages": None}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    tokens = shlex.split(_read_run_json(dirs)["rerun_command"])

    args = parse_args(tokens[1:])
    validate_args(args)  # must not sys.exit
    assert (args.mode, args.tier) == ("single", "legendary")
    assert args.stages_given is False


def test_rerun_command_flag_order(tmp_path):
    run_info = {
        **_RUN_INFO, "mode": "line", "requested_stages": 3,
        "description": "a fire lizard", "image": "scan.png",
    }
    dirs = write_output(_LINE, run_info, base_dir=str(tmp_path))
    cmd = _read_run_json(dirs)["rerun_command"]
    for flag in ("--image", "--description", "--mode", "--tier", "--stages"):
        assert flag in cmd
    assert (cmd.index("--image") < cmd.index("--description")
            < cmd.index("--mode") < cmd.index("--tier") < cmd.index("--stages"))


def test_rerun_command_starts_with_console_script_name(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["rerun_command"].startswith("fakemon-forge ")


# ---------------------------------------------------------------------------
# run.json — generated_stages
# ---------------------------------------------------------------------------

def test_generated_stages_one_entry_per_stage(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    assert len(_read_run_json(dirs)["generated_stages"]) == 3


def test_generated_stages_records_stage_name_and_sprite_prompt(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    stages = _read_run_json(dirs)["generated_stages"]
    assert stages[0] == {
        "stage": 1, "name": "Flamburr",
        "sprite_prompt": "A small fire lizard, GBA pixel art, white background",
    }
    assert stages[2] == {
        "stage": 3, "name": "Flamburron",
        "sprite_prompt": "A large fire dragon, imposing, GBA pixel art",
    }


def test_generated_stages_in_stage_order(tmp_path):
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    stages = _read_run_json(dirs)["generated_stages"]
    assert [s["stage"] for s in stages] == [1, 2, 3]


def test_generated_stages_excludes_outcome_data(tmp_path):
    """run.json is inputs-only: no sprite paths, no success/failure flags."""
    dirs = write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    stages = _read_run_json(dirs)["generated_stages"]
    for stage in stages:
        assert set(stage.keys()) == {"stage", "name", "sprite_prompt"}


def test_generated_stages_sprite_prompt_null_when_stage_omits_it(tmp_path):
    """The model is not required to return sprite_prompt (generator._normalize
    never fills it in), and main.py's sprite call sites tolerate its absence by
    warning and moving on. The manifest records the gap as null rather than
    raising and costing the run every other output."""
    stage = {k: v for k, v in _STAGE_1.items() if k != "sprite_prompt"}
    dirs = write_output([stage], _RUN_INFO, base_dir=str(tmp_path))
    assert _read_run_json(dirs)["generated_stages"][0]["sprite_prompt"] is None
    assert (dirs[0] / "stats.json").exists()


def test_run_json_has_no_outcome_fields_at_top_level(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    data = _read_run_json(dirs)
    for forbidden in ("sprite_path", "success", "warnings", "errors"):
        assert forbidden not in data


# ---------------------------------------------------------------------------
# run.json — written up front, before any stage subfolder
# ---------------------------------------------------------------------------

def test_run_json_written_before_any_stage_dir_is_created(tmp_path):
    """Not just "before the stage files" — before the stage *folders*, so the
    manifest is on disk from the first moment the run folder is non-empty."""
    seen = []
    real_mkdir = Path.mkdir

    def spy_mkdir(self, *args, **kwargs):
        seen.append((self.name, (self.parent / "run.json").exists()))
        return real_mkdir(self, *args, **kwargs)

    with patch.object(Path, "mkdir", spy_mkdir):
        write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))

    stage_mkdirs = [entry for entry in seen if entry[0].startswith("stage")]
    assert len(stage_mkdirs) == 3
    assert all(run_json_existed for _, run_json_existed in stage_mkdirs)


def test_run_json_exists_even_if_a_later_stage_write_fails(tmp_path):
    with patch("fakemon_forge.writer._write_entry", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            write_output(_LINE, _RUN_INFO, base_dir=str(tmp_path))
    run_json = tmp_path / "Flamburr" / "run.json"
    assert run_json.exists()
    json.loads(run_json.read_text(encoding="utf-8"))  # complete, not partial


def test_run_json_write_failure_propagates_uncaught(tmp_path):
    with patch("fakemon_forge.writer.json.dumps", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# run.json — serialization convention (matches stats.json/entry.md)
# ---------------------------------------------------------------------------

def test_run_json_non_ascii_is_escaped(tmp_path):
    run_info = {**_RUN_INFO, "description": "un lézard de feu ♂"}
    dirs = write_output(_SINGLE, run_info, base_dir=str(tmp_path))
    raw = (dirs[0].parent / "run.json").read_text(encoding="utf-8")
    assert "é" not in raw
    assert "\\u00e9" in raw
    assert json.loads(raw)["description"] == "un lézard de feu ♂"


def test_run_json_indented_like_other_json_outputs(tmp_path):
    dirs = write_output(_SINGLE, _RUN_INFO, base_dir=str(tmp_path))
    raw = (dirs[0].parent / "run.json").read_text(encoding="utf-8")
    assert raw.startswith("{\n  ")
