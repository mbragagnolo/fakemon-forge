# Bug: quantizer key bleed + frame-2 gate false positive

> Tracking: #90

## Summary
Two post-processing defects, both confirmed across all 31 mons of the
2026-08-04 generation round: (1) `_quantize_gen3` nearest-maps creature pixels
onto the transparency key, punching holes through the body; (2) `build_frame2`'s
acceptance band reads palette-index churn as motion, so a color-flickering,
essentially static candidate is accepted as frame 2.

## Repro steps
1. Run a normal generation (e.g. `fakemon-forge "<any description>"`) on the
   host with the ML stack; or take any `output/<Mon>/stage1_<Mon>/` from the
   2026-08-04 round.
2. Open `sprite.png`, recolor palette-index-0 pixels not connected to the
   border to magenta (script kept in the session scratchpad:
   `visualize_bleed.py`).
3. Compare `sprite.png` and `sprite_frame2.png` as an A/B flip, and decompose
   their index diff into silhouette change (key <-> non-key) vs same-silhouette
   index change (`diagnose.py` in the scratchpad).

## Expected vs. actual
- Expected (1): only background — the outer field and see-through pockets —
  carries palette index 0; the creature body has no transparent pixels.
- Actual (1): every mon of the round has enclosed key speckle inside the body
  (57–5,805 px per sprite; e.g. Windle 5,805, Tinelle 5,487, Pebblid 4,101),
  clustered on light beige/gray regions and along the anti-aliased seams
  between the 768-res faux "pixels". Downstream keys by color value, so each
  speckle is a hole in-game.
- Expected (2): an accepted frame 2 differs from frame 1 by visible motion
  (pose change), and a candidate that is frame 1 plus noise is rejected.
- Actual (2): 25/31 candidates accepted; in every one, same-silhouette color
  churn dwarfs silhouette change (Slidot: 12.2% churn vs 0.9% silhouette;
  round-wide churn 5.3–24.7% vs silhouette 0.9–8.3%). Visually the pair reads
  as shading shimmer on a static pose — the false positive that survived #67.

## Root cause

**(1) — confirmed.** `_quantize_gen3` (`fakemon_forge/sprites.py:584`) maps
every pixel of the flattened image against a palette whose index 0 is
`_KEY_COLOR` `(200, 200, 168)`. `_nudge_off_key` only guarantees no *centroid*
lands within `_KEY_COLLISION_DISTANCE` (12) of the key; the *pixel* mapping is
nearest-entry over the whole palette, and the key sits mid-range in the
light-warm-gray/beige region of RGB space. Any creature pixel closer to the key
than to any of the ≤13 surviving centroids — typically AA blends between faux
pixels, and light midtones the 13-color budget gave no centroid (Kitewk's
belly `(242, 212, 183)` is 46 from the key; Windle's light grays sit between
body gray `(134, 137, 137)` and the key) — resolves to index 0. Verified by
synthetic reproduction: a beige-bodied creature on a white background run
through `_quantize_gen3` alone gets 917 body pixels keyed. The flatten stage is
*not* leaking: keyed speckle sits in regions far outside `_KEY_TOLERANCE` of
the detected background, so it can only have been produced by the quantize.
`quantize_to_reference` (`sprites.py:279`) shares the defect — same
key-in-palette mapping — which is why back sprites and locked frame-2
candidates show it too.

**(2) — confirmed.** `build_frame2` (`sprites.py:858`) accepts a candidate when
`difference_ratio` ∈ `[low=0.02, high=0.30]`. `difference_ratio` counts *any*
palette-index inequality, but the img2img re-render (strength 0.30) plus
re-quantization onto frame 1's palette flips 5–25% of pixels as baseline noise
— shading redistribution and quantization-boundary flips with no pose change.
Baseline churn alone therefore lands mid-band, so the gate cannot fail: it
accepts flicker as motion. The 6 squash fallbacks of the round fired only when
churn happened to exceed 0.30 — same metric, same blindness. #67's squash-init
change put a real pose change into the *init image*, but nothing verifies the
candidate *kept* it; the low-strength img2img largely denoises back to the
frame-1 pose while re-rolling the shading.

## Affected files
- `fakemon_forge/sprites.py` — `_quantize_gen3` (key participates in
  nearest-color mapping; flatten mask discarded before quantize),
  `quantize_to_reference` (same), `build_frame2` / `difference_ratio`
  (churn-blind acceptance metric).
- `fakemon_forge/icon.py` — *not* affected the same way: `_build_frame1`
  already keeps a background mask and force-pastes index 0 from it, which is
  the shape of the fix for (1).

## Regression info
Not a regression in the strict sense: both defects shipped with the SDXL
retooling (#61 line). (1) became *visible* at 768 because `_SPRITE_SIZE = 768`
keeps the native render, whose AA faux-pixel seams a downscale used to merge
away. (2) is the unresolved remainder of #67.

## Proposed fix approach
Per the user's direction (2026-08-04):

1. **Mask-based keying.** In `_quantize_gen3` (and `quantize_to_reference`),
   derive a background mask from `_flatten_background_to_key` (which pixels it
   actually keyed), quantize the creature pixels against a palette *without*
   the key entry, then force index 0 from the mask — mirroring
   `icon.py:_build_frame1`'s existing paste-from-mask pattern. A creature pixel
   can then never fall onto the key regardless of its color.
2. **Squash default, strict structural gate.** Keep `procedural_squash` as the
   default frame 2. Accept an img2img candidate only on a motion-sensitive
   measure — e.g. silhouette (key/non-key mask) difference, or an index diff
   computed after a structural downscale — with thresholds that reject
   flicker-only candidates like this round's 25 false positives (silhouette
   diff of the accepted set ran 0.9–8.3%, so the bar must sit above the churn
   floor, not above zero).
