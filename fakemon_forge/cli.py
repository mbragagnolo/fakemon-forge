import argparse
import sys
from pathlib import Path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
        help="Generate one form ('single') or a 3-stage evolutionary line ('line')",
    )
    parser.add_argument(
        "--tier",
        choices=["standard", "pseudo", "legendary", "mythical"],
        default="standard",
        help="Power tier: standard, pseudo (pseudo-legendary line), legendary, or mythical",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not args.image and not args.description:
        print("Error: at least one of --image or --description must be provided.", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "tier", "standard") in ("legendary", "mythical") and args.mode == "line":
        print(
            f"Error: --tier {args.tier} is always a single form; use --mode single.",
            file=sys.stderr,
        )
        sys.exit(1)

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
