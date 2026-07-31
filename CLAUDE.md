# fakemon-forge

CLI tool that generates Fakemon (name, types, stats, Pokédex entry, sprites)
from a text description or reference image, using the Mistral API for text and
Stable Diffusion (diffusers + a pixel-art LoRA) for sprites.

## Commands

- `pytest` — run the test suite
- Package layout is flat (`fakemon_forge/` at the repo root); run pytest from
  the repo root so the package is importable without installing.

## Test suite slicing

Every test is fully mocked — none makes a real API call or generates a real
image. But tests marked `ml` (see `tests/test_sprites_ml.py`) still need
torch installed, because `sprites.py` uses function-local `import torch` that
runs even with a mocked pipeline.

`tests/conftest.py` auto-skips `ml` tests when torch is not installed. In the
keep sandbox container (slim image: pytest, mistralai, Pillow only) that means
~21 tests report as skipped — **this is expected and correct**. Do NOT
`pip install` torch/diffusers/transformers to "fix" skipped tests, and do not
treat those skips as failures. The full suite, including `ml` tests, runs on
the host where the ML stack and GPU exist.

When adding tests: anything that calls `generate_sprite`,
`generate_sprite_img2img`, or otherwise triggers a real `import torch` /
`from diffusers import ...` belongs in `test_sprites_ml.py` (or carries
`@pytest.mark.ml`). Tests that fake those modules via `sys.modules` injection
or touch no ML code go in the regular files.
