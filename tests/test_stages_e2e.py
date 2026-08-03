"""End-to-end tests for the stage count (#59).

Everything from ``main`` down runs for real here — argparse, ``validate_args``,
``generate_fakemon``, ``write_output`` and ``export_ini``. Only two things are
faked: the Mistral client (so no API call) and the sprite/audio calls (so no
torch, no GPU). That is deliberate — the unit tests in ``test_generator.py``
and ``test_cli.py`` already pin the pieces; what is untested until here is that
the pieces are wired to each other, and that a stage count asked for on the
command line is the stage count that reaches disk.

Non-ML by construction: the ML entry points are patched out in ``main``'s
namespace, so ``sprites.py``'s function-local ``import torch`` never runs.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fakemon_forge.main import main


# ---------------------------------------------------------------------------
# A fake Mistral client
# ---------------------------------------------------------------------------

class _FakeClient:
    """Records the messages it is sent and replays a canned JSON array.

    The recorded messages are the point: they are how the prompt guarantee
    below is asserted at full-CLI level rather than by calling ``_user_prompt``
    directly.
    """

    def __init__(self, payload):
        self._payload = payload
        self.calls = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, *, model, messages):
        self.calls.append(messages)
        content = json.dumps(self._payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    @property
    def user_prompt(self) -> str:
        """The user turn of the first request — the stage/BST prompt."""
        return self.calls[0][1]["content"]


def _stage(number: int, name: str) -> dict:
    """One well-formed stage dict, as the model would return it.

    Deliberately clean — names inside the 10-character contract, a real Gen 3
    ability, all six base stats — so that any failure below is about stage
    wiring and not about repair paths that are already covered elsewhere.
    """
    return {
        "name": name,
        "stage": number,
        "types": ["Fire"],
        "category": "EMBER",
        "ability": "Blaze",
        "abilities_gen3": ["Blaze"],
        "base_stats": {
            "hp": 45, "attack": 52, "defense": 43,
            "sp_atk": 60, "sp_def": 50, "speed": 65,
        },
        "pokedex_entry": "A small fiery creature. It burns brightly at dusk.",
        "sprite_prompt": "fire lizard, GBA pixel art",
        "levitates": False,
        "height_dm": 5,
        "weight_hg": 30,
    }


_NAMES = ["Flamburr", "Flamburro", "Flamburron"]


def _payload(count: int) -> list[dict]:
    return [_stage(i, _NAMES[i - 1]) for i in range(1, count + 1)]


# ---------------------------------------------------------------------------
# Fixture: run main for real, with only the client and the ML calls faked
# ---------------------------------------------------------------------------

@pytest.fixture
def forge(tmp_path, monkeypatch):
    """Yield ``run(argv, payload)`` -> (client, output_root).

    ``write_output`` writes to ``output/`` relative to the cwd, so the cwd is
    moved into tmp_path and the real writer is left in place.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")
    monkeypatch.chdir(tmp_path)

    def run(argv, payload):
        client = _FakeClient(payload)
        with (
            patch("fakemon_forge.main.Mistral", return_value=client),
            patch("fakemon_forge.main.load_txt2img_pipeline", return_value=MagicMock()),
            patch("fakemon_forge.main.load_img2img_pipeline", return_value=MagicMock()),
            patch("fakemon_forge.main.make_img2img_pipeline", return_value=MagicMock()),
            patch("fakemon_forge.main.generate_sprite"),
            patch("fakemon_forge.main.generate_sprite_img2img"),
            patch("fakemon_forge.main.generate_frame2"),
            patch("fakemon_forge.main.generate_shiny"),
            patch("fakemon_forge.main.stitch_spritesheet"),
            patch("fakemon_forge.main.generate_footprint"),
            patch("fakemon_forge.main.generate_icon"),
            patch("fakemon_forge.main.generate_cry"),
        ):
            main(argv)
        return client, tmp_path / "output"

    return run


def _stage_dirs(output_root: Path) -> list[Path]:
    """Every stage directory written, in stage order."""
    return sorted((output_root / "Flamburr").glob("stage*_*"))


def _assert_stage_is_complete(stage_dir: Path, number: int, name: str) -> None:
    """A stage directory carries everything the injector contract needs."""
    assert stage_dir.name == f"stage{number}_{name}"

    stats = json.loads((stage_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["name"] == name
    assert stats["stage"] == number
    assert stats["types"] == ["Fire"]
    assert set(stats["base_stats"]) == {
        "hp", "attack", "defense", "sp_atk", "sp_def", "speed"
    }

    assert (stage_dir / "entry.md").read_text(encoding="utf-8").strip()

    # export_ini ran and produced a parseable file, not just an empty one.
    ini = (stage_dir / f"{name}.ini").read_text(encoding="utf-8")
    assert ini.startswith("[Pokemon]")
    assert f"PokemonName={name.upper()}" in ini


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_two_stage_line_writes_two_complete_stages(forge):
    """`--mode line --stages 2` produces exactly two stages, both exportable."""
    client, output_root = forge(
        ["--description", "fire lizard", "--mode", "line", "--stages", "2"],
        _payload(2),
    )

    dirs = _stage_dirs(output_root)
    assert len(dirs) == 2
    _assert_stage_is_complete(dirs[0], 1, "Flamburr")
    _assert_stage_is_complete(dirs[1], 2, "Flamburro")


def test_two_stage_prompt_asks_for_two_stages(forge):
    """The count from the command line is the count that reaches the model."""
    client, _ = forge(
        ["--description", "fire lizard", "--mode", "line", "--stages", "2"],
        _payload(2),
    )

    prompt = client.user_prompt
    assert "two evolutionary stages (stages 1 and 2)" in prompt
    assert "BST targets: stage 1 ~305, stage 2 ~468." in prompt
    # A 2-stage line has no adolescent form to describe.
    assert "adolescent" not in prompt
    assert "Stage 3" not in prompt


def test_three_stage_line_writes_three_complete_stages(forge):
    """The pre-#59 behaviour, unchanged: `--mode line` alone is three stages."""
    client, output_root = forge(
        ["--description", "fire lizard", "--mode", "line"],
        _payload(3),
    )

    dirs = _stage_dirs(output_root)
    assert len(dirs) == 3
    for i, name in enumerate(_NAMES, start=1):
        _assert_stage_is_complete(dirs[i - 1], i, name)


def test_single_mode_writes_one_stage(forge):
    """A single form is still one stage and still exports."""
    client, output_root = forge(["--description", "fire lizard"], _payload(1))

    dirs = _stage_dirs(output_root)
    assert len(dirs) == 1
    _assert_stage_is_complete(dirs[0], 1, "Flamburr")


# ---------------------------------------------------------------------------
# The structure-identical guarantee, at full-prompt level
# ---------------------------------------------------------------------------

# The prompt as it read before #59, verbatim. Restated here rather than
# imported from test_generator.py on purpose: this is the second, independent
# pin of the same guarantee, and a pin that shares its expected value with the
# thing it is double-checking is not a second pin at all. Task 10 asserts this
# against `_user_prompt`; this file asserts it against whatever actually
# reaches the client after argparse, `validate_args` and `generate_fakemon`
# have each had a turn.
_PRE_59_LINE_PROMPT = (
    "Generate three evolutionary stages (stages 1, 2, and 3) for a Fakemon "
    "based on this description:\n"
    "\n"
    "fire lizard\n"
    "\n"
    "BST targets: stage 1 ~300, stage 2 ~420, stage 3 ~520.\n"
    "Evolutionary progression — each stage must look and feel visually "
    "distinct:\n"
    "  Stage 1: juvenile/child form — small and simple, cute or curious "
    "expression, limited limbs or features, undeveloped power.\n"
    "  Stage 2: adolescent/teenage form — noticeably larger, silhouette more "
    "defined, signature features emerging, power becoming apparent.\n"
    "  Stage 3: adult/final form — fully developed, imposing presence, complex "
    "design with a different silhouette from stage 1, design complexity at its "
    "peak."
)


def test_default_line_prompt_through_main_is_structure_identical_to_pre_59(forge):
    """`--mode line` with no `--stages` must send the pre-#59 prompt with the
    corrected BST numbers substituted in, and nothing else changed.

    Full-string equality, through the whole CLI path — a reworded hint, a
    reordered section, a lost line or a stray flag leaking into the prompt all
    fail here. Literal byte-identity is impossible: the BST values are rendered
    into this string and correcting them is the point of #59.
    """
    client, _ = forge(["--description", "fire lizard", "--mode", "line"], _payload(3))

    expected = _PRE_59_LINE_PROMPT.replace(
        "stage 1 ~300, stage 2 ~420, stage 3 ~520",
        "stage 1 ~295, stage 2 ~405, stage 3 ~518",
    )
    assert client.user_prompt == expected


def test_explicit_three_stages_sends_the_same_prompt_as_the_default(forge):
    """Passing the default explicitly changes nothing on the wire."""
    default_client, _ = forge(
        ["--description", "fire lizard", "--mode", "line"], _payload(3)
    )
    explicit_client, _ = forge(
        ["--description", "fire lizard", "--mode", "line", "--stages", "3"],
        _payload(3),
    )
    assert explicit_client.user_prompt == default_client.user_prompt


# ---------------------------------------------------------------------------
# Both CLI rejections, driven through main
# ---------------------------------------------------------------------------

def test_single_mode_with_stages_exits_1(forge, capsys):
    """`--stages` is meaningless for a single form and is refused, not ignored."""
    with pytest.raises(SystemExit) as exc:
        forge(
            ["--description", "fire lizard", "--mode", "single", "--stages", "2"],
            _payload(1),
        )
    assert exc.value.code == 1
    assert "--stages applies only to --mode line" in capsys.readouterr().err


def test_pseudo_tier_with_two_stages_exits_1(forge, capsys):
    """A pseudo-legendary is defined by its three-stage climb."""
    with pytest.raises(SystemExit) as exc:
        forge(
            ["--description", "deep-sea serpent", "--mode", "line",
             "--tier", "pseudo", "--stages", "2"],
            _payload(2),
        )
    assert exc.value.code == 1
    assert "--tier pseudo is always a 3-stage line" in capsys.readouterr().err


def test_rejections_happen_before_any_api_call(forge, tmp_path):
    """A refused run must not spend a request or leave a half-written tree."""
    with pytest.raises(SystemExit):
        forge(
            ["--description", "fire lizard", "--mode", "single", "--stages", "2"],
            _payload(1),
        )
    assert not (tmp_path / "output").exists()


# ---------------------------------------------------------------------------
# Pinned non-goal: the model's stage count is not validated
# ---------------------------------------------------------------------------

def test_model_returning_three_stages_for_a_two_stage_request_is_accepted(forge):
    """A model that ignores the requested count is taken at its word.

    This is a deliberate non-goal, not an oversight: the 2-attempt retry budget
    is spent on the name contract, and truncating a line the model considered
    coherent would produce a worse result than honouring it. The test exists so
    the behaviour is not silently "fixed" later — if you are here because you
    want to add stage-count validation, that is a spec change, not a bug fix.
    """
    client, output_root = forge(
        ["--description", "fire lizard", "--mode", "line", "--stages", "2"],
        _payload(3),
    )

    # Asked for two...
    assert "two evolutionary stages (stages 1 and 2)" in client.user_prompt
    # ...got three, kept three, and did not spend a retry arguing about it.
    assert len(client.calls) == 1
    dirs = _stage_dirs(output_root)
    assert len(dirs) == 3
    for i, name in enumerate(_NAMES, start=1):
        _assert_stage_is_complete(dirs[i - 1], i, name)


# ---------------------------------------------------------------------------
# The fixtures directory must never carry game data
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"

# Aggregate band statistics are small integers. A ROM offset or a species index
# stored as a number would land far outside this range.
_MAX_PLAUSIBLE_NUMBER = 10_000

# Keys are lowercase identifiers or digit strings. A species name used as a key
# ("Treecko", "TREECKO") fails this without the check ever naming one.
_KEY_RE = re.compile(r"^_?[a-z0-9_]+$")

# 0x-prefixed literals and bare long hex runs are what ROM offsets look like in
# free text; three or more numbers in a row is what an ID list looks like.
_HEX_RE = re.compile(r"0x[0-9A-Fa-f]+|\b[0-9A-Fa-f]{6,}\b")
_ID_LIST_RE = re.compile(r"\b\d+\b[\s,]+\b\d+\b[\s,]+\b\d+\b")


def _assert_clean(value, path: str) -> None:
    """Recursively assert a fixture value carries only aggregate numbers.

    The check is structural rather than a blocklist of species names, for two
    reasons: a blocklist would put the very names the scrub removed back into
    the repo, and it would only catch the species someone thought to list.
    Requiring every leaf to be a small number leaves nowhere for a name, an ID
    list or an offset to hide.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            assert _KEY_RE.match(key), f"{path}: suspicious key {key!r}"
            _assert_clean(child, f"{path}.{key}")
        return

    # An array is the natural shape of a species-ID list; bands never need one.
    assert not isinstance(value, list), f"{path}: unexpected list"

    if isinstance(value, str):
        # Only descriptive `_comment` fields may hold free text.
        assert path.rsplit(".", 1)[-1].startswith("_"), f"{path}: unexpected text"
        assert not _HEX_RE.search(value), f"{path}: looks like a ROM offset"
        assert not _ID_LIST_RE.search(value), f"{path}: looks like an ID list"
        return

    assert isinstance(value, int) and not isinstance(value, bool), (
        f"{path}: unexpected value {value!r}"
    )
    assert 0 <= value <= _MAX_PLAUSIBLE_NUMBER, f"{path}: implausible number {value}"


def test_fixtures_directory_is_not_empty():
    """Guards the check below from passing vacuously if the dir is renamed."""
    assert list(_FIXTURES.iterdir())


def test_fixtures_carry_no_species_data():
    """No species name, ID list or ROM offset may reach `tests/fixtures/`.

    `test_bst_targets.py` makes the same guarantee for the one fixture it
    reads; this covers the whole directory, so a *newly added* fixture — or a
    regenerated one that skipped the scrub — is caught too.
    """
    for path in sorted(_FIXTURES.rglob("*")):
        if path.is_dir():
            continue
        assert path.suffix == ".json", f"{path.name}: only JSON fixtures allowed"
        _assert_clean(
            json.loads(path.read_text(encoding="utf-8")), path.name
        )
