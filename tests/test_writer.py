import json
import pytest
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

def test_creates_top_level_fakemon_dir(tmp_path):
    write_output(_SINGLE, base_dir=str(tmp_path))
    assert (tmp_path / "Flamburr").is_dir()


def test_creates_stage1_subdir(tmp_path):
    write_output(_SINGLE, base_dir=str(tmp_path))
    assert (tmp_path / "Flamburr" / "stage1_Flamburr").is_dir()


def test_line_mode_creates_three_stage_dirs(tmp_path):
    write_output(_LINE, base_dir=str(tmp_path))
    root = tmp_path / "Flamburr"
    assert (root / "stage1_Flamburr").is_dir()
    assert (root / "stage2_Flamburro").is_dir()
    assert (root / "stage3_Flamburron").is_dir()


def test_returns_list_of_stage_dirs(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    assert len(dirs) == 1
    assert dirs[0] == tmp_path / "Flamburr" / "stage1_Flamburr"


def test_returns_three_dirs_for_line(tmp_path):
    dirs = write_output(_LINE, base_dir=str(tmp_path))
    assert len(dirs) == 3


def test_returned_dirs_match_stage_names(tmp_path):
    dirs = write_output(_LINE, base_dir=str(tmp_path))
    assert dirs[0].name == "stage1_Flamburr"
    assert dirs[1].name == "stage2_Flamburro"
    assert dirs[2].name == "stage3_Flamburron"


# ---------------------------------------------------------------------------
# stats.json
# ---------------------------------------------------------------------------

def test_creates_stats_json(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    assert (dirs[0] / "stats.json").exists()


def test_stats_json_has_required_fields(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert _STATS_KEYS <= set(data.keys())


def test_stats_json_excludes_llm_only_fields(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert not (_LLM_ONLY & set(data.keys()))


def test_stats_json_values_are_correct(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    data = json.loads((dirs[0] / "stats.json").read_text())
    assert data["name"] == "Flamburr"
    assert data["stage"] == 1
    assert data["types"] == ["Fire"]
    assert data["ability"] == "Blaze"
    assert data["base_stats"]["hp"] == 45


def test_stats_json_is_valid_json(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    raw = (dirs[0] / "stats.json").read_text(encoding="utf-8")
    json.loads(raw)  # must not raise


# ---------------------------------------------------------------------------
# entry.md
# ---------------------------------------------------------------------------

def test_creates_entry_md(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    assert (dirs[0] / "entry.md").exists()


def test_entry_md_contains_pokedex_text(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    text = (dirs[0] / "entry.md").read_text(encoding="utf-8")
    assert "A small fiery creature with a burning tail tip." in text


def test_each_stage_has_its_own_entry(tmp_path):
    dirs = write_output(_LINE, base_dir=str(tmp_path))
    assert "Flamburro grows bolder" in (dirs[1] / "entry.md").read_text()
    assert "Flamburron's inferno"   in (dirs[2] / "entry.md").read_text()


# ---------------------------------------------------------------------------
# Name collision handling
# ---------------------------------------------------------------------------

def test_collision_appends_suffix(tmp_path):
    write_output(_SINGLE, base_dir=str(tmp_path))   # creates Flamburr/
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))  # should create Flamburr_2/
    assert dirs[0].parent.name == "Flamburr_2"


def test_collision_increments_suffix(tmp_path):
    write_output(_SINGLE, base_dir=str(tmp_path))   # Flamburr/
    write_output(_SINGLE, base_dir=str(tmp_path))   # Flamburr_2/
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))  # Flamburr_3/
    assert dirs[0].parent.name == "Flamburr_3"


def test_no_collision_no_suffix(tmp_path):
    dirs = write_output(_SINGLE, base_dir=str(tmp_path))
    assert dirs[0].parent.name == "Flamburr"
