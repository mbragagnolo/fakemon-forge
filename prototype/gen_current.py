# Spike: generate the 5 test creatures with the CURRENT production stack
# (dreamshaper-8 + pksp768 LoRA), mirroring main.py's front + back flow.
# Output: prototype/out/current/<name>_front.png / _back.png (P-mode 768, as prod saves them)
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fakemon_forge.sprites import (
    load_txt2img_pipeline, make_img2img_pipeline,
    generate_sprite, generate_sprite_img2img,
)
from spike_prompts import CREATURES, SEED

OUT = Path(__file__).parent / "out" / "current"
OUT.mkdir(parents=True, exist_ok=True)

pipe = load_txt2img_pipeline()
img2img = make_img2img_pipeline(pipe)

for name, types, desc in CREATURES:
    front = str(OUT / f"{name}_front.png")
    back = str(OUT / f"{name}_back.png")
    t0 = time.time()
    generate_sprite(desc, types, front, pipeline=pipe, seed=SEED)
    t1 = time.time()
    # Mirrors main.py:141 — back view chained from the front sprite.
    generate_sprite_img2img(
        desc, types, front, back, pipeline=img2img,
        extra_tags=["backside"], seed=SEED, strength=0.65, reference_path=front,
    )
    t2 = time.time()
    print(f"{name}: front {t1-t0:.0f}s, back {t2-t1:.0f}s", flush=True)

print("current stack done")
