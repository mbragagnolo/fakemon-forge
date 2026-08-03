import os
import random
import sys

from mistralai.client import Mistral

from fakemon_forge.cli import parse_args, validate_args
from fakemon_forge.vision import describe_image
from fakemon_forge.generator import generate_fakemon
from fakemon_forge.sprites import (
    generate_sprite_pair,
    generate_sprite_img2img,
    generate_frame2,
    generate_shiny,
    stitch_spritesheet,
    load_txt2img_pipeline,
    load_img2img_pipeline,
    make_img2img_pipeline,
)
from fakemon_forge.icon import generate_icon
from fakemon_forge.writer import write_output
from fakemon_forge.footprint import generate_footprint
from fakemon_forge.export_ini import export_ini
from fakemon_forge.cries import generate_cry


# Chibi caricature tags for the party-menu icon's img2img pass. Prototype
# tunable: needs a GPU spike to confirm the LoRA actually produces caricature
# proportions (big head / small body) from img2img before committing to these.
_CHIBI_TAGS = ["chibi", "big head", "small body"]

# Footprint size by line length -> stage number. A 2-stage line reuses the
# 3-stage line's endpoints; any length not listed (a single form) falls through
# to the full-size footprint.
_FULL_FOOTPRINT = 0.9
_FOOTPRINT_FRACTIONS = {
    2: {1: 0.6, 2: 0.9},
    3: {1: 0.6, 2: 0.75, 3: 0.9},
}


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

    # `forms` holds the stage dicts that came back; `args.stages` is how many
    # were asked for. Naming the local `stages` would shadow the count — the
    # same collision that silently broke the generator before it was caught.
    forms = generate_fakemon(
        combined, args.mode, tier=args.tier, stages=args.stages, client=client
    )

    # Always the txt2img pipeline, even for --image runs (issue #69): the
    # drawing's content reaches the sprite renderer only via describe_image's
    # vision output feeding into sprite_prompt (see the `combined` construction
    # above), never via img2img on the raw pixels — img2img seeded with a single
    # front-facing drawing has no mechanism to turn it into a genuine back view
    # for the right half of the front+back canvas.
    # `load_img2img_pipeline` is consequently no longer called from here; it
    # stays imported so the "--image runs never load it" regression test has a
    # live binding to assert against.
    pipeline = load_txt2img_pipeline()
    img2img_pipeline = make_img2img_pipeline(pipeline)

    stage_dirs = write_output(forms)

    for stage, stage_dir in zip(forms, stage_dirs):
        seed = random.randint(0, 2**32 - 1)

        # Audio does not depend on the sprites — generate it before the sprite
        # block so a sprite failure (which `continue`s) can't skip the cry.
        try:
            generate_cry(
                forms[0]["name"],   # line_name — stage 1's name, shared by the whole line
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
        back_path = str(stage_dir / "sprite_back.png")
        try:
            generate_sprite_pair(
                stage["sprite_prompt"], stage["types"], sprite_path, back_path,
                pipeline=pipeline, seed=seed,
            )
        except Exception as exc:
            print(
                f"Warning: sprite generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )
            continue

        small_path = str(stage_dir / "sprite_small.png")
        chibi_path = str(stage_dir / "sprite_chibi.png")
        try:
            try:
                # Chibi caricature enhancement: render a big-head/small-body
                # variant of the front sprite, then downscale THAT into the
                # party-menu icon so it reads like a Gen-3 caricature.
                generate_sprite_img2img(
                    stage["sprite_prompt"], stage["types"], sprite_path, chibi_path,
                    pipeline=img2img_pipeline, extra_tags=_CHIBI_TAGS, seed=seed,
                )
            except Exception:
                # Enhancement is optional: fall back to the plain downscale of
                # sprite.png (exactly today's behavior). No warning.
                icon_source = sprite_path
            else:
                icon_source = chibi_path
            generate_icon(icon_source, small_path)
        except Exception as exc:
            print(
                f"Warning: icon generation failed for {stage['name']}: {exc}",
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

        # Footprint size scales with the stage's position in its line; single
        # forms (any tier) always use the full-size footprint. A 2-stage line
        # takes the first and last fractions and skips the middle, mirroring
        # its height/weight defaults, where the final form takes the stage-3
        # row — otherwise a 2-stage juvenile would print a full-size footprint.
        size_fraction = _FOOTPRINT_FRACTIONS.get(len(forms), {}).get(
            stage["stage"], _FULL_FOOTPRINT
        )

        footprint_path = str(stage_dir / "footprint.png")
        try:
            generate_footprint(
                sprite_path,
                footprint_path,
                types=stage["types"],
                size_fraction=size_fraction,
                blank=stage.get("levitates", False),
            )
        except Exception as exc:
            print(
                f"Warning: footprint generation failed for {stage['name']}: {exc}",
                file=sys.stderr,
            )

    for stage_dir in stage_dirs:
        export_ini(stage_dir)

    print(f"Done! Output written to {stage_dirs[0].parent}/")


if __name__ == "__main__":
    main()
