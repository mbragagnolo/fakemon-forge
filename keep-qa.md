# Manual QA — `quantize_to_reference` (palette-lock helper)

- [ ] In a REPL, build a reference with `postprocess(<some RGB image>)`, run `quantize_to_reference(<different RGB image>, reference)`, and confirm the output is a 96×96 `P`-mode image.
- [ ] Confirm `out.getpalette() == reference.getpalette()` (byte-for-byte identical palette, not an adaptive one).
- [ ] Quantize two visibly different inputs against the same reference and confirm both outputs share the reference's palette (the core shared-palette guarantee).
- [ ] Pass a non-`P` reference (e.g. a plain RGB image) and confirm it raises `ValueError` with a clear "palette-mode" message rather than a cryptic Pillow error.
- [ ] Confirm the original `image` and `reference` are unchanged after the call (size/mode/palette intact) — no in-place mutation.
- [ ] Confirm `postprocess` still behaves as before (adaptive path untouched) and `pytest tests/test_sprites.py` is green without torch installed.
