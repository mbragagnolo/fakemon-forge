# Bugfix Verification: quantizer key bleed + frame-2 gate false positive

> Tracking: #90 — fix commit `1c40c3d` on `fix/90-key-bleed-frame2-gate`

Run on the host (full ML stack). Generate at least two fresh mons: one with a
light/beige/tan description (the tones that bled worst — e.g. "a sandy tan
armadillo with a cream belly") and one dark-bodied.

## Original repro (now fixed)

- [ ] Generate the light/tan mon and open `sprite.png` over a contrasting
      viewer background (or recolor index-0 pixels magenta with the session's
      `visualize_bleed.py`) → no key-colored speckle/cracks inside the body;
      the only transparent regions are the outer field and genuine
      see-through gaps.
- [ ] A/B-flip `sprite.png` against `sprite_frame2.png` → the pair reads as
      motion (usually the clean bottom-anchored squash; occasionally a
      genuinely-moved accepted candidate). No whole-body color shimmer on a
      static pose.

## Adjacent behavior

- [ ] `sprite_back.png` for the same mons → background keys out fully, body
      has no transparency holes, and its palette is identical to the front's
      (shared 16-color palette).
- [ ] `sprite_shiny.png` / `sprite_frame2_shiny.png` → background still keys
      (index 0 unrotated), creature colors rotated consistently across views.
- [ ] A mon with real see-through gaps (legs apart, a ring/handle shape) →
      those pockets are still transparent, not filled in by the mask-based
      keying.
- [ ] `spritesheet.png` → all six cells key against the sheet's uniform key
      background; empty cells solid key.

## Fix boundary cases

- [ ] The tan mon's body tones that sit *near* the key color stay opaque
      creature pixels (nudged off the key), while the background around them
      still keys — i.e. the fix separated "near-key creature" from
      "background" rather than trading one bleed for another.
- [ ] Across a handful of mons, frame 2 is mostly the squash fallback (the
      strict gate rejecting flicker) → confirm the squash itself looks like
      breathing at GBA scale, since it is now the default animation.
- [ ] The dark-bodied mon → no new speckle (dark tones are far from the key;
      this guards the no-key mapping against off-by-one palette-index shifts).
