import importlib.metadata
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Keys persisted only if the stage dict carries them, with the fallback used
# when it does not — hand-built/partial stage dicts must still write cleanly.
_STATS_DEFAULTS = {"levitates": False, "height_dm": 5, "weight_hg": 30,
                   "abilities_gen3": [], "category": "", "traits": []}

_STATS_KEYS = {"name", "stage", "types", "ability", "base_stats", *_STATS_DEFAULTS}

# Repo root, resolved the same way generator.py/export_ini.py locate
# resources/: two levels up from this file (fakemon_forge/../).
_REPO_ROOT = Path(__file__).parent.parent


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


def _package_version() -> str:
    try:
        return importlib.metadata.version("fakemon-forge")
    except Exception:
        pass
    try:
        text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _rerun_command(run_info: dict) -> str:
    parts = ["fakemon-forge"]
    if run_info.get("image") is not None:
        parts += ["--image", shlex.quote(str(run_info["image"]))]
    if run_info.get("description") is not None:
        parts += ["--description", shlex.quote(str(run_info["description"]))]
    parts += ["--mode", shlex.quote(str(run_info["mode"]))]
    parts += ["--tier", shlex.quote(str(run_info["tier"]))]
    if run_info["mode"] == "line":
        parts += ["--stages", shlex.quote(str(run_info["requested_stages"]))]
    return " ".join(parts)


def _build_run_json(stages: list[dict], run_info: dict) -> dict:
    return {
        "description": run_info.get("description"),
        "image": run_info.get("image"),
        "vision_description": run_info.get("vision_description", ""),
        # The exact blob handed to generate_fakemon. On a text-only run this
        # duplicates `description`; on an --image run it is the only field
        # showing what the LLM actually received, since the drawing reaches it
        # solely as vision text. Recorded either way so both run shapes share
        # one schema.
        "combined_prompt": run_info.get("combined_prompt"),
        "mode": run_info["mode"],
        "tier": run_info["tier"],
        "requested_stages": run_info.get("requested_stages"),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_version": _package_version(),
        "git_sha": _git_sha(),
        "rerun_command": _rerun_command(run_info),
        # sprite_prompt is read with a fallback, like the optional stats.json
        # keys above: the model is not guaranteed to return it, and a stage
        # dict missing it must not turn a run that would merely have skipped
        # its sprite into one that writes nothing at all.
        "generated_stages": [
            {"stage": s["stage"], "name": s["name"],
             "sprite_prompt": s.get("sprite_prompt")}
            for s in stages
        ],
    }


def _write_run_json(stages: list[dict], run_info: dict, fakemon_dir: Path) -> None:
    data = _build_run_json(stages, run_info)
    (fakemon_dir / "run.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


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


def write_output(
    stages: list[dict], run_info: dict = None, base_dir: str = "output"
) -> list[Path]:
    """Create folder tree, write run.json, stats.json and entry.md. Returns stage dirs.

    `run_info` is optional: omitting it writes no run.json and leaves the rest
    of the tree byte-identical to what this function produced before manifests
    existed. A manifest describes a *CLI invocation*, and this function is also
    called with hand-built stage dicts that never came from one -- the same
    reason `_STATS_DEFAULTS` exists. `main` always passes it.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    fakemon_dir = _resolve_dir(stages[0]["name"], base)
    fakemon_dir.mkdir()

    # Written before any stage subfolder so a run that dies mid-way through
    # sprite generation still leaves behind a manifest of what it was asked
    # to do.
    if run_info is not None:
        _write_run_json(stages, run_info, fakemon_dir)

    stage_dirs = []
    for stage in stages:
        stage_dir = fakemon_dir / f"stage{stage['stage']}_{stage['name']}"
        stage_dir.mkdir()
        _write_stats(stage, stage_dir)
        _write_entry(stage, stage_dir)
        stage_dirs.append(stage_dir)

    return stage_dirs
