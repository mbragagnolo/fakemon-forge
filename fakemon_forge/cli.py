import argparse
import sys
from pathlib import Path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_STAGE_CHOICES = [2, 3]
_DEFAULT_STAGES = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a Fakemon from a drawing and/or description."
    )
    parser.add_argument("--image", help="Path to a scan or photo of the creature (jpg/png)")
    parser.add_argument("--description", help="Free-text description of the creature")
    parser.add_argument(
        "--mode",
        choices=["single", "line"],
        default="single",
        help="Generate one form ('single') or an evolutionary line ('line'); "
             "see --stages for the line's length",
    )
    parser.add_argument(
        "--tier",
        choices=["standard", "pseudo", "legendary", "mythical"],
        default="standard",
        help="Power tier: standard, pseudo (pseudo-legendary line), legendary, or mythical",
    )
    parser.add_argument(
        "--stages",
        type=int,
        choices=_STAGE_CHOICES,
        default=None,
        help=f"Number of stages in an evolutionary line, with --mode line "
             f"(default: {_DEFAULT_STAGES})",
    )

    args = parser.parse_args(argv)

    # Record whether the flag was actually given *before* filling in the
    # default. The single-mode rejection below is about supplying a flag that
    # does not apply to the chosen mode, so an explicit `--stages 3` is exactly
    # as contradictory as `--stages 2` — and the default must never trip it.
    args.stages_given = args.stages is not None
    if args.stages is None:
        args.stages = _DEFAULT_STAGES
    return args


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def validate_args(args):
    # Order is deliberate: the first rule that fires is the only one reported.
    # An unusable run is reported before any shape rule, and a flag that does
    # not apply to the chosen mode is reported before rules about its value —
    # a wall of errors makes the actual mistake harder to find.
    if not args.image and not args.description:
        _fail("at least one of --image or --description must be provided.")

    tier = getattr(args, "tier", "standard")
    stages = getattr(args, "stages", _DEFAULT_STAGES)

    if getattr(args, "stages_given", False) and args.mode == "single":
        _fail("--stages applies only to --mode line; a single form is one stage.")

    if tier in ("legendary", "mythical") and args.mode == "line":
        _fail(f"--tier {tier} is always a single form; use --mode single.")

    # A pseudo-legendary is defined by its line -- the tier's whole meaning is a
    # final form that rivals a legendary at the end of a three-stage climb.
    # Allowing it as a single form yields a standalone species carrying a
    # juvenile's stat budget, which is the inconsistency #59 exists to remove.
    if tier == "pseudo" and args.mode == "single":
        _fail("--tier pseudo is always a 3-stage line; use --mode line.")

    if tier == "pseudo" and stages != 3:
        _fail(f"--tier pseudo is always a 3-stage line; --stages {stages} is not valid.")

    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"Error: image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            print(
                f"Error: unsupported image type '{path.suffix}'. "
                f"Expected one of: {', '.join(sorted(_IMAGE_EXTENSIONS))}",
                file=sys.stderr,
            )
            sys.exit(1)
