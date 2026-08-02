import json
from pathlib import Path

# Keys persisted only if the stage dict carries them, with the fallback used
# when it does not — hand-built/partial stage dicts must still write cleanly.
_STATS_DEFAULTS = {"levitates": False, "height_dm": 5, "weight_hg": 30, "abilities_gen3": []}

_STATS_KEYS = {"name", "stage", "types", "ability", "base_stats", *_STATS_DEFAULTS}


def _resolve_dir(name: str, base: Path) -> Path:
    candidate = base / name
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = base / f"{name}_{n}"
        if not candidate.exists():
            return candidate
        n += 1


def _write_stats(stage: dict, stage_dir: Path) -> None:
    data = {k: stage[k] for k in _STATS_KEYS if k not in _STATS_DEFAULTS}
    for key, fallback in _STATS_DEFAULTS.items():
        data[key] = stage.get(key, fallback)
    (stage_dir / "stats.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _write_entry(stage: dict, stage_dir: Path) -> None:
    (stage_dir / "entry.md").write_text(
        stage["pokedex_entry"], encoding="utf-8"
    )


def write_output(stages: list[dict], base_dir: str = "output") -> list[Path]:
    """Create folder tree, write stats.json and entry.md. Returns stage dirs."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    fakemon_dir = _resolve_dir(stages[0]["name"], base)
    fakemon_dir.mkdir()

    stage_dirs = []
    for stage in stages:
        stage_dir = fakemon_dir / f"stage{stage['stage']}_{stage['name']}"
        stage_dir.mkdir()
        _write_stats(stage, stage_dir)
        _write_entry(stage, stage_dir)
        stage_dirs.append(stage_dir)

    return stage_dirs
