# Manual QA — `icon.py` party-menu icon (`sprite_small.png`)

- [ ] Run `generate_icon` on a real 768x768 pipeline sprite and confirm the saved `sprite_small.png` is 32x64 PNG with two stacked 32x32 frames.
- [ ] Open the icon in an image viewer: background is opaque teal-green (96, 152, 128) and dominates; the creature is recognizable at 32px.
- [ ] Confirm the file has no transparency (fully opaque) and its palette has teal at index 0 with <= 16 distinct colours.
- [ ] Diff the two frames visually: frame 2 (bottom) is frame 1 (top) nudged down 1px, top row teal, bottom row of frame 1 dropped.
- [ ] Feed a non-`P`-mode (e.g. RGB) image and confirm it raises `ValueError` naming the mode.
- [ ] Feed an all-background sprite and confirm the icon is entirely teal, still valid 32x64 PNG.
