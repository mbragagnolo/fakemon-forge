import os
import sys

from mistralai.client import Mistral

from fakemon_forge.cli import parse_args, validate_args
from fakemon_forge.vision import describe_image
from fakemon_forge.generator import generate_fakemon
from fakemon_forge.sprites import (
    generate_sprite,
    generate_sprite_img2img,
    load_txt2img_pipeline,
    load_img2img_pipeline,
)
from fakemon_forge.writer import write_output


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print(
            "Error: MISTRAL_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = Mistral(api_key=api_key)

    vision_desc = ""
    if args.image:
        vision_desc = describe_image(args.image, client=client)

    parts = [p for p in [vision_desc, args.description] if p]
    combined = "\n\n".join(parts)

    stages = generate_fakemon(combined, args.mode, tier=args.tier, client=client)

    if args.image:
        pipeline = load_img2img_pipeline()
    else:
        pipeline = load_txt2img_pipeline()

    stage_dirs = write_output(stages)

    for stage, stage_dir in zip(stages, stage_dirs):
        sprite_path = str(stage_dir / "sprite.png")
        try:
            if args.image:
                generate_sprite_img2img(
                    stage["sprite_prompt"], args.image, sprite_path, pipeline=pipeline
                )
            else:
                generate_sprite(stage["sprite_prompt"], sprite_path, pipeline=pipeline)
        except Exception as exc:
            print(
                f"Warning: sprite generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

    print(f"Done! Output written to output/{stages[0]['name']}/")


if __name__ == "__main__":
    main()
