# Manual QA — `cries.py` procedural cry synthesis

- [ ] `pytest tests/test_cries.py` passes from the repo root (no ML stack needed).
- [ ] Generate a cry (`generate_cry("Pikaclone", 1, ["Electric"], "cry.wav")`) and play `cry.wav` in an audio player — it should sound like a short Gen 3 creature cry.
- [ ] Open the WAV in an audio tool and confirm it is mono, 8-bit unsigned, 10512 Hz.
- [ ] Generate the same line at stages 1→3 and confirm each cry is audibly longer and lower-pitched than the last.
- [ ] Generate a Fairy line vs a Fire line and confirm Fairy sounds noticeably higher-pitched.
- [ ] Call twice with identical args and confirm the two files are byte-identical.
- [ ] Call with `types=[]`, an unknown type (e.g. `["Cosmic"]`), and an empty `line_name` — each writes a valid WAV without crashing.
- [ ] Spot-check a Fairy line (low duration multiplier) actually lands within 0.35–1.55 s.
