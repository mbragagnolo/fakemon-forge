# Bug: img2img back sprite inits from the drawing, not the front sprite

> Tracking: #10

## Summary
On `--image` runs, front and back sprites look like two different creatures.
The back sprite is generated as an independent img2img draw from the user's
drawing instead of from the generated front sprite, so the two views never
share a design — only a seed and (since PR #9) a palette.

## Repro steps
1. `python -m fakemon_forge.main --image <any drawing>` (observed with
   `F:\Projects\Projet_Pokemon\pokemon_arthur_1.png` → Duocurve, branch
   `keep/957b27ba`, also present on `main`).
2. Compare `sprite.png` and `sprite_back.png` in the stage dir.

## Expected vs. actual
- Expected: the back view reads as the same creature from behind (7bb7566's
  stated behavior: "Back sprites now use the front sprite as the init image
  ... so colours and body shape carry over between views").
- Actual: back sprite differs wildly in body shape/design from the front.

## Root cause  (confirmed)
`fakemon_forge/main.py`, back-sprite block:

```python
init_image = args.image if args.image else sprite_path
```

Only the txt2img path chains the back from the generated front sprite. With
`--image`, the init is the user's drawing — a *front* view — prompted with
`backside` at strength 0.65: an independent interpretation that never sees
the front sprite. Introduced by 7bb7566, which implemented its
front-as-init intent for the txt2img path only.

The user's initial hypothesis — that the input image is bypassed entirely —
is **falsified**: with seed and prompt fixed and only the init image varied,
outputs are bit-identical for the same init (diff 0.0000) and 64.9%
different for a different init. The image conditions generation strongly.

## Affected files
- `fakemon_forge/main.py` — `init_image` selection in the back-sprite block
- `tests/test_main.py` — img2img-path back-sprite expectations assert the
  drawing as init (they encode the buggy behavior)

## Regression info
- Introduced by: 7bb7566 ("fix: use img2img for back sprites to improve
  front/back consistency", 2026-06-29) — intent/implementation mismatch.

## Proposed fix approach
Always init the back sprite from `sprite_path` (the generated front sprite),
in both paths — the drawing holds no backside information, and the front
sprite is the canonical design. Update the img2img-path tests accordingly.
Land after the PR #6–#9 stack merges (the stack edits the same block:
`reference_path=sprite_path` was added there), then rebase this branch.
