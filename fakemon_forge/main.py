import os
import random
import sys

from mistralai.client import Mistral

from fakemon_forge.cli import parse_args, validate_args
from fakemon_forge.vision import describe_image
from fakemon_forge.generator import generate_fakemon
from fakemon_forge.sprites import (
    generate_sprite,
    generate_sprite_img2img,
    generate_frame2,
    generate_shiny,
    stitch_spritesheet,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    make_img2img_pipeline,
)
from fakemon_forge.writer import write_output
from fakemon_forge.export_ini import export_ini
from fakemon_forge.cries import generate_cry


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
        img2img_pipeline = pipeline
    else:
        pipeline = load_txt2img_pipeline()
        img2img_pipeline = make_img2img_pipeline(pipeline)

    stage_dirs = write_output(stages)

    for stage, stage_dir in zip(stages, stage_dirs):
        seed = random.randint(0, 2**32 - 1)

        # Audio does not depend on the sprites — generate it before the sprite
        # block so a sprite failure (which `continue`s) can't skip the cry.
        try:
            generate_cry(
                stages[0]["name"],   # line_name — stage 1's name, shared by the whole line
                stage["stage"],
                stage["types"],
                str(stage_dir / "cry.wav"),
            )
        except Exception as exc:
            print(
                f"Warning: cry generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        sprite_path = str(stage_dir / "sprite.png")
        try:
            if args.image:
                generate_sprite_img2img(
                    stage["sprite_prompt"], stage["types"], args.image, sprite_path,
                    pipeline=pipeline, seed=seed,
                )
            else:
                generate_sprite(args.description, stage["types"], sprite_path, pipeline=pipeline, seed=seed)
        except Exception as exc:
            print(
                f"Warning: sprite generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )
            continue

        back_path = str(stage_dir / "sprite_back.png")
        try:
            # Always chain the back view from the generated front sprite — a
            # user drawing is a front view and holds no backside information.
            generate_sprite_img2img(
                stage["sprite_prompt"], stage["types"], sprite_path, back_path,
                pipeline=img2img_pipeline, extra_tags=["backside"], seed=seed,
                strength=0.65, reference_path=sprite_path,
            )
        except Exception as exc:
            print(
                f"Warning: back sprite generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        frame2_path = str(stage_dir / "sprite_frame2.png")
        try:
            generate_frame2(
                stage["sprite_prompt"], stage["types"], sprite_path, frame2_path,
                pipeline=img2img_pipeline, seed=seed,
            )
        except Exception as exc:
            print(
                f"Warning: frame 2 generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        frame2_shiny_path = str(stage_dir / "sprite_frame2_shiny.png")
        try:
            generate_shiny(frame2_path, stage["name"], frame2_shiny_path)
        except Exception as exc:
            print(
                f"Warning: frame 2 shiny generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        shiny_path = str(stage_dir / "sprite_shiny.png")
        try:
            generate_shiny(sprite_path, stage["name"], shiny_path)
        except Exception as exc:
            print(
                f"Warning: shiny generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        back_shiny_path = str(stage_dir / "sprite_back_shiny.png")
        try:
            generate_shiny(back_path, stage["name"], back_shiny_path)
        except Exception as exc:
            print(
                f"Warning: back shiny generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

        try:
            stitch_spritesheet(stage_dir, str(stage_dir / "spritesheet.png"))
        except Exception as exc:
            print(
                f"Warning: spritesheet stitching failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

    for stage_dir in stage_dirs:
        export_ini(stage_dir)

    print(f"Done! Output written to {stage_dirs[0].parent}/")


if __name__ == "__main__":
    main()
