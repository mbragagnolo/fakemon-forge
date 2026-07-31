# Manual QA — footprint.py (16×16 Pokédex footprint renderer)

- [ ] Render a real stage `sprite.png` (P-mode, 768) and open the output: it is exactly 16×16 and shows a single foot (tapered pad + toe marks) as opaque black on transparency.
- [ ] Open the output in an editor and confirm every pixel is either fully opaque black or fully transparent — no greys, no anti-aliased edges.
- [ ] Vary the primary type: `["Dragon"]`/`["Fire"]` → 3 claw wedges, `["Normal"]`/`["Ground"]` → 4 round toes, `["Flying"]` → 3 prongs, `["Water"]`/empty/unknown → plain pad; toes read as distinct marks with a clear gap above the pad.
- [ ] Call with `blank=True` and a bogus `sprite_path` → an all-transparent 16×16 PNG is written and no file-read error occurs.
- [ ] Feed a very thin/tall leg sprite → a small tall "hoof" oval (not a crash), still within the colour contract.
- [ ] Confirm `import fakemon_forge.footprint` does not pull torch/diffusers into `sys.modules`, and `pytest tests/test_footprint.py` passes (not skips) in the slim container.
