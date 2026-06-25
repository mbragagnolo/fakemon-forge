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
