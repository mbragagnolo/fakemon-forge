import pytest
import sys
from pathlib import Path
from unittest.mock import patch

from fakemon_forge.cli import parse_args, validate_args


# --- parse_args ---

def test_parse_description_only():
    args = parse_args(["--description", "breathes fire"])
    assert args.description == "breathes fire"
    assert args.image is None
    assert args.mode == "single"


def test_parse_image_only(tmp_path):
    img = tmp_path / "creature.png"
    img.touch()
    args = parse_args(["--image", str(img)])
    assert args.image == str(img)
    assert args.description is None


def test_parse_mode_line():
    args = parse_args(["--description", "fluffy cloud beast", "--mode", "line"])
    assert args.mode == "line"


def test_parse_mode_defaults_to_single():
    args = parse_args(["--description", "rock turtle"])
    assert args.mode == "single"


def test_parse_both_inputs(tmp_path):
    img = tmp_path / "drawing.jpg"
    img.touch()
    args = parse_args(["--image", str(img), "--description", "three tails"])
    assert args.image == str(img)
    assert args.description == "three tails"


# --- validate_args ---

def test_validate_raises_if_no_inputs():
    args = parse_args(["--description", "placeholder"])
    args.description = None
    args.image = None
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1


def test_validate_raises_if_image_path_missing(tmp_path):
    args = parse_args(["--description", "placeholder"])
    args.image = str(tmp_path / "nonexistent.png")
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1


def test_validate_raises_if_image_not_an_image(tmp_path):
    bad_file = tmp_path / "drawing.txt"
    bad_file.touch()
    args = parse_args(["--description", "placeholder"])
    args.image = str(bad_file)
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1


def test_validate_passes_with_description_only():
    args = parse_args(["--description", "ice lizard"])
    validate_args(args)  # should not raise


def test_validate_passes_with_image_only(tmp_path):
    img = tmp_path / "creature.png"
    img.touch()
    args = parse_args(["--image", str(img)])
    validate_args(args)  # should not raise


def test_validate_passes_with_both(tmp_path):
    img = tmp_path / "creature.jpg"
    img.touch()
    args = parse_args(["--image", str(img), "--description", "spiky"])
    validate_args(args)  # should not raise


def test_validate_accepts_jpeg_extension(tmp_path):
    img = tmp_path / "creature.jpeg"
    img.touch()
    args = parse_args(["--description", "placeholder"])
    args.image = str(img)
    validate_args(args)  # should not raise


def test_validate_invalid_mode():
    with pytest.raises(SystemExit):
        parse_args(["--description", "blob", "--mode", "duo"])


# --- --tier ---

def test_parse_tier_defaults_to_standard():
    args = parse_args(["--description", "fire lizard"])
    assert args.tier == "standard"


def test_parse_tier_pseudo():
    args = parse_args(["--description", "fire lizard", "--tier", "pseudo"])
    assert args.tier == "pseudo"


def test_parse_tier_legendary():
    args = parse_args(["--description", "fire lizard", "--tier", "legendary"])
    assert args.tier == "legendary"


def test_parse_tier_mythical():
    args = parse_args(["--description", "fire lizard", "--tier", "mythical"])
    assert args.tier == "mythical"


def test_parse_tier_invalid():
    with pytest.raises(SystemExit):
        parse_args(["--description", "blob", "--tier", "uber"])


def test_validate_legendary_with_line_exits(capsys):
    args = parse_args(["--description", "fire lizard", "--tier", "legendary", "--mode", "line"])
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1


def test_validate_mythical_with_line_exits(capsys):
    args = parse_args(["--description", "fire lizard", "--tier", "mythical", "--mode", "line"])
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1


def test_validate_legendary_with_single_passes():
    args = parse_args(["--description", "fire lizard", "--tier", "legendary"])
    validate_args(args)  # should not raise


def test_validate_pseudo_with_line_passes():
    args = parse_args(["--description", "fire lizard", "--tier", "pseudo", "--mode", "line"])
    validate_args(args)  # should not raise


# --- --stages (#59) ---

def _args(*extra):
    return parse_args(["--description", "fire lizard", *extra])


def test_parse_stages_defaults_to_three():
    """The default is what keeps an existing `--mode line` invocation
    unchanged."""
    assert _args("--mode", "line").stages == 3


def test_parse_stages_default_is_an_int():
    assert isinstance(_args("--mode", "line").stages, int)


@pytest.mark.parametrize("value", [2, 3])
def test_parse_stages_accepts_two_and_three(value):
    args = _args("--mode", "line", "--stages", str(value))
    assert args.stages == value
    assert isinstance(args.stages, int)


@pytest.mark.parametrize("value", ["1", "4", "0", "-1", "two", ""])
def test_parse_stages_rejects_everything_else(value):
    """`--stages 1` is deliberately NOT a synonym for `--mode single` --
    two ways to say one thing invites drift."""
    with pytest.raises(SystemExit):
        _args("--mode", "line", "--stages", value)


def test_mode_choices_are_unchanged():
    """`--stages` is additive; shape values must not appear on `--mode`, or
    every existing `--mode line` invocation would break."""
    assert _args("--mode", "single").mode == "single"
    assert _args("--mode", "line").mode == "line"
    for bad in ("line2", "line3", "branched"):
        with pytest.raises(SystemExit):
            _args("--mode", bad)


# --- rejection: --stages only applies to --mode line -------------------------

@pytest.mark.parametrize("value", ["2", "3"])
def test_validate_stages_with_single_exits(capsys, value):
    """Rejected because the flag was *given*, not because of its value -- an
    explicit `--stages 3` with single mode is just as contradictory."""
    args = _args("--mode", "single", "--stages", value)
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--stages" in err
    assert "--mode line" in err


def test_validate_single_without_stages_passes():
    """The default must not trip the rule it would otherwise always fire on."""
    validate_args(_args("--mode", "single"))


def test_validate_default_mode_without_stages_passes():
    validate_args(_args())


# --- rejection: pseudo is always a 3-stage line ------------------------------

def test_validate_pseudo_with_two_stages_exits(capsys):
    args = _args("--tier", "pseudo", "--mode", "line", "--stages", "2")
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "pseudo" in err.lower()
    assert "3" in err or "three" in err.lower()


def test_validate_pseudo_with_explicit_three_stages_passes():
    """The guard must not over-reject."""
    validate_args(_args("--tier", "pseudo", "--mode", "line", "--stages", "3"))


def test_validate_pseudo_line_default_stages_passes():
    validate_args(_args("--tier", "pseudo", "--mode", "line"))


def test_validate_pseudo_with_single_exits(capsys):
    """README already documents pseudo as line-only, and `--tier`'s own help
    calls it a "pseudo-legendary line", but nothing enforced it. Left alone it
    produces a standalone form carrying a juvenile's BST -- the exact
    inconsistency #59 fixes for the standard tier."""
    args = _args("--tier", "pseudo", "--mode", "single")
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    assert "pseudo" in capsys.readouterr().err.lower()


# --- combinations that must still pass ---------------------------------------

@pytest.mark.parametrize("extra", [
    ("--mode", "line", "--stages", "2"),
    ("--mode", "line", "--stages", "3"),
    ("--mode", "line"),
    ("--mode", "single"),
])
def test_validate_standard_tier_accepts_every_shape(extra):
    validate_args(_args(*extra))


# --- the existing rejection is untouched -------------------------------------

@pytest.mark.parametrize("tier", ["legendary", "mythical"])
def test_validate_legendary_and_mythical_with_line_still_exit(capsys, tier):
    args = _args("--tier", tier, "--mode", "line")
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    assert f"--tier {tier}" in capsys.readouterr().err


@pytest.mark.parametrize("tier", ["legendary", "mythical"])
def test_validate_legendary_and_mythical_with_single_still_pass(tier):
    validate_args(_args("--tier", tier, "--mode", "single"))


# --- validation order is deliberate, not incidental --------------------------

def test_only_one_message_when_several_rules_could_fire(capsys):
    """`--mode single --tier pseudo --stages 2` trips three rules. The
    mode/flag contradiction is reported, and only it -- a wall of errors makes
    the actual mistake harder to find."""
    args = _args("--mode", "single", "--tier", "pseudo", "--stages", "2")
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert "--stages" in err


def test_missing_input_is_reported_before_shape_rules(capsys):
    """A run with no description and no image is unusable whatever its shape."""
    args = _args("--mode", "single", "--stages", "2")
    args.description = None
    args.image = None
    with pytest.raises(SystemExit) as exc:
        validate_args(args)
    assert exc.value.code == 1
    assert "--image" in capsys.readouterr().err
