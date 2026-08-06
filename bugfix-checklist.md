# Bugfix Verification: k_centroid emits colours absent from the source tile (#92)

Fix commit: `8420c11` on `fix/k-centroid-tile-colours` — `k_centroid` now emits
the dominant cluster's most frequent *actual* colour instead of the cluster
centroid.

Local baseline before the fix (measured 2026-08-06): **68 / 68** stage sheets
under `output/` exceed 16 colours; worst is `stage3_Tempestris` at **6875**.

## Original repro (now fixed)

- [ ] Re-stitch every existing stage sheet with the fixed code (no sprite
      regeneration needed — the `sprite*.png` views are already correct):

      ```powershell
      python -c "
      from pathlib import Path
      from fakemon_forge.sprites import stitch_spritesheet
      for p in sorted(Path('output').glob('*/stage*')):
          if (p / 'spritesheet.png').exists():
              stitch_spritesheet(str(p), str(p / 'spritesheet.png'))
      "
      ```

      → completes without errors.
- [ ] Re-run the colour audit:

      ```powershell
      python -c "
      from PIL import Image
      from pathlib import Path
      counts = sorted((len(set(Image.open(p).convert('RGB').get_flattened_data())), p.parent.name)
                      for p in Path('output').glob('*/stage*/spritesheet.png'))
      print('worst:', counts[-3:])
      print('over 32:', sum(1 for n, _ in counts if n > 32), '/', len(counts))
      "
      ```

      → **0 / 68** sheets over 32 colours (each sheet holds one normal
      16-colour palette plus one shiny palette; the issue's reference batch
      measured 28 on its worst sheet). Nothing anywhere near the old
      thousands.

## Adjacent behavior

- [ ] Open a few re-stitched `spritesheet.png` files (e.g. Tempestris,
      Fortressk, Axolitt) → cells still read as clean sprites: outline
      restoration intact, no visible speckle/quality regression from the
      dominant-colour choice changing on boundary tiles.
- [ ] Icons share the same downscale (`_build_frame1` → `k_centroid`
      768→32): regenerate one icon (or eyeball an existing one against a
      regenerated one) → still a recognisable, ≤16-colour icon.

## Fix boundary cases

- [ ] A 64px view stitches without any downscale (the `cell.size !=
      (cell_size, cell_size)` guard): `stitch_spritesheet(dir,
      out, cell_size=96)` on a 96px-view stage, or trust
      `test_spritesheet_cell_size_override` → cells land verbatim.
- [ ] Upscale fallback unchanged (`k_centroid` to a larger size falls back
      to `NEAREST`): covered by `test_k_centroid_upscale_falls_back_to_nearest`
      in the green suite — no manual step needed.
