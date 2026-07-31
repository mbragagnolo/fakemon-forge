# Manual QA — pure-PIL frame-2 assembly (#3)

- [ ] Run `pytest tests/test_sprites.py` and confirm the new `procedural_squash`, `recenter_to_anchor`, `difference_ratio`, `build_frame2`, and `_background_index` tests pass without torch installed.
- [ ] Generate a real 96x96 P-mode front sprite, call `procedural_squash(frame1)`, save both, and eyeball that the second frame reads as a bottom-anchored breathing/bounce (feet planted, top compressed) — not a shifted or shrunken blob.
- [ ] Confirm `procedural_squash(frame1).getpalette() == frame1.getpalette()` and the output is 96x96 mode `P` (no palette drift / no new colours).
- [ ] Feed `build_frame2` a near-identical candidate and a wildly-different candidate; confirm both fall back to the squash, and an in-band candidate is returned palette-locked and recentred.
- [ ] Feed `recenter_to_anchor` a deliberately shifted candidate and confirm its non-background bbox realigns to frame 1's bottom-centre (no visible jitter between frames).
- [ ] Pass a non-`P`-mode `frame1` to `procedural_squash` / `recenter_to_anchor` / `build_frame2` and confirm a `ValueError` mentioning "palette-mode"; pass mismatched sizes to `difference_ratio` and confirm it raises.
