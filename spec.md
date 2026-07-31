# Spec: `cries.py` — procedural GBA-style `cry.wav` synthesis

## Summary

`fakemon-forge` generates a Fakemon line (name, types, stats, Pokédex entry,
sprites) from a description or reference image. This slice adds a **standalone,
fully-tested library module** `fakemon_forge/cries.py` that synthesizes a
Gen 3-style creature **cry** as a `cry.wav` for one Fakemon stage — procedurally,
with **no ML, no GPU, standard library only** (`math`, `random`, `hashlib`,
`wave`, and optionally `struct`/`array`).

The module exposes a single public function:

```python
def generate_cry(line_name: str, stage: int, types: list[str], output_path: str) -> None
```

It writes a WAV file to `output_path`: **mono, 8-bit unsigned PCM, 10512 Hz**,
duration in **0.35–1.55 s**, peak-normalized so the loudest sample deviates from
128 by ≈120 (reference Gen 3 cries peak at 120 in u8 scale). The cry is a
deterministic function of `(line_name, stage, types)`: identical arguments
produce a **byte-identical** WAV every call, and a whole evolution line (sharing
`line_name`, which is stage 1's name) shares one seeded "voice" and motif, with
per-stage transforms making later stages longer and lower-pitched.

This is Slice 1/2 of #22. A **later** slice will wire `generate_cry` into the
per-stage generation loop in `main.py`.

### Explicitly out of scope

- **No changes to `main.py`.** Wiring the cry into the generation loop is the
  next slice; this module must stand alone.
- **No changes to `pyproject.toml` dependencies** or any other module. Stdlib
  only; no new packages.
- No ML, no GPU, no `torch`/`diffusers`, no network, no time/OS entropy.
- No playback, format conversion (only `.wav`), or ROM packaging.

## Inputs

`generate_cry(line_name, stage, types, output_path)`:

- **`line_name: str`** — stage 1's name, shared across the whole evolution line.
  It is the **sole seed source**: `hashlib.sha256(line_name.encode())` seeds a
  local `random.Random` instance so the whole line shares one voice + motif.
  (Assumption: encode with default UTF-8; empty string is a valid seed and must
  not crash.)
- **`stage: int`** — evolution stage, `>= 1`. Stage 1 applies no stage transform;
  stages `> 1` apply the corpus-derived stage transform (longer, lower, harder).
  Single-form mons are treated as `stage == 1`. Any `stage >= 1` is valid; the
  final duration is always capped at 1.5 s.
- **`types: list[str]`** — the stage's types. **`types[0]`** (the primary type)
  selects the type profile; an empty list, a missing/unknown primary type falls
  back to the default profile row. Secondary types **may** optionally influence
  timbre, but only deterministically. Type-name matching is exact and
  case-sensitive against the profile keys below (matching how `sprites.py`'s
  `_TYPE_TAGS` keys off exact capitalized names like `"Fire"`, `"Fairy"`).
- **`output_path: str`** — filesystem path to write the WAV to. (Assumption: the
  caller/tests pass a full path including filename, e.g. `tmp_path / "cry.wav"`;
  the task text says "writes a `cry.wav` to `output_path`". The function writes
  exactly to `output_path` and does not create parent directories or resolve a
  subpath — callers pass the final file path, mirroring how `sprites.py` writers
  take an `output_path` and call `.save(output_path)` directly.)

Return value: **`None`**. The function's effect is the written file.

## Outputs

A single WAV file at `output_path` with these guaranteed properties (all
assertable by reading it back with the `wave` module):

- **Channels**: `getnchannels() == 1` (mono).
- **Sample width**: `getsampwidth() == 1` (8-bit).
- **Sample encoding**: unsigned PCM; every sample byte is a valid u8 in `[0, 255]`
  (the `wave` module's contract for 8-bit).
- **Frame rate**: `getframerate() == 10512` Hz.
- **Duration**: `getnframes() / 10512` lands in **`[0.35, 1.55]`** seconds. (The
  synthesis targets 0.35–1.5 s pre-cap; the 1.55 upper bound is the test
  tolerance band.)
- **Peak level**: the maximum absolute deviation of any sample from 128 is
  **≈120** (target `120/127` of full scale), asserted within a small tolerance
  (±3) to absorb u8 integer-rounding.
- **Determinism**: two calls with identical `(line_name, stage, types)` produce
  **byte-identical** files.

## Behavior

All randomness comes from one `random.Random` seeded from
`sha256(line_name.encode())`; the global `random` module state is never touched,
and no time/OS entropy is used. Given that, the synthesis follows the validated
model below. **All numeric table values are audition-validated defaults / tunable
starting points, not contracts** — they may be tweaked by ear, but the tests in
this spec must still pass.

### 1. Seed
Derive a seed integer from `hashlib.sha256(line_name.encode())` and construct a
local `rng = random.Random(seed)`. Every subsequent random draw (voice, motif,
envelope) uses `rng`.

### 2. Type profile
Select the profile row from the primary type `types[0]`; fall back to the
**default row** when `types` is empty or `types[0]` is unrecognized. Each row
provides: register band (Hz), syllable-count weights (for counts 1–4), noise
base, AM base, and duration multiplier.

| Type | Band (Hz) | Syll. weights | Noise | AM | Dur mult |
|---|---|---|---|---|---|
| Grass | 130–420 | 3,3,1,0 | 0.10 | 0.45 | 1.0 |
| Dragon | 100–300 | 4,2,0,0 | 0.16 | 0.50 | 1.0 |
| Fighting | 140–380 | 2,3,2,0 | 0.12 | 0.45 | 1.0 |
| Rock | 110–320 | 3,2,1,0 | 0.18 | 0.50 | 1.0 |
| Ground | 110–340 | 3,2,1,0 | 0.14 | 0.45 | 1.0 |
| Normal | 450–1800 | 1,3,3,1 | 0.03 | 0.30 | 0.85 |
| Fairy | 900–2400 | 1,2,3,2 | 0.02 | 0.30 | 0.75 |
| Flying | 700–2200 | 1,3,2,1 | 0.04 | 0.35 | 0.80 |
| Electric | 220–700 | 2,3,1,0 | 0.10 | 0.60 | 0.90 |
| Bug | 260–800 | 2,2,2,1 | 0.08 | 0.65 | 0.90 |
| Psychic | 350–1100 | 3,2,1,0 | 0.03 | 0.50 | 1.25 |
| Ghost | 300–900 | 3,2,0,0 | 0.05 | 0.55 | 1.10 |
| Poison | 500–1500 | 3,2,1,0 | 0.15 | 0.50 | 1.05 |
| Water | 250–850 | 2,3,1,0 | 0.03 | 0.35 | 1.35 |
| Ice | 400–1300 | 2,3,1,0 | 0.04 | 0.35 | 1.20 |
| Fire | 100–320 | 3,2,1,0 | 0.25 | 0.40 | 1.0 |
| Steel | 180–550 | 2,2,2,0 | 0.14 | 0.55 | 1.0 |
| Dark | 120–360 | 3,2,1,0 | 0.12 | 0.50 | 1.0 |
| **(default)** | **300–900** | **2,3,1,0** | **0.08** | **0.40** | **1.0** |

### 3. Voice (per line, from `rng` within the primary type's profile)
- **Register**: log-uniform within the type's Hz band.
- **Sound source**: one of `pwm` (duty 0.12–0.5), `saw`, `triangle`,
  `fm` (ratio ∈ {1.5, 2.0, 2.77, 3.51}, index 0.8–3.0),
  `ring` (detune 1.002–1.03).
- **Noise**: type noise base × 0.5–1.6; color either white or sample-hold
  (gritty digital) at ~SR/900 or ~SR/300 update rates.
- **Articulation**: AM rate 18–90 Hz (depth type-scaled off the AM base);
  vibrato 4.5–11 Hz; trill 11–22 Hz.

### 4. Motif (per line)
1–4 syllables (count drawn using the profile's syllable weights). Each syllable:
- an interval from the scale set {1.0, 1.19, 1.34, 1.5, 1.78, 2.0, 0.84, 0.67};
- a contour: fall / rise / bend / flat+vibrato / trill;
- a length weight (last syllable × 1.6);
- inter-syllable gaps 15–70 ms;
- contour depth 0.25–0.6;
- per-syllable envelope: fast attack (~6% of the syllable), slight decay.

### 5. Stage transform (applied for `stage > 1`; no-op at `stage == 1`)
- **Duration** × `(1 + 0.20 * (stage - 1))`, then the final duration is capped at
  1.5 s.
- **Pitch** × `0.90 ** (stage - 1)` (each stage lower than the last).
- **Hardening** `+0.25 * (stage - 1)`: mix in an octave-up rough partial and/or
  raise the FM index so the attack is brighter/harsher per stage.

### 6. Global envelope and normalization
- Base duration 0.45–1.0 s (drawn from `rng`) × type duration multiplier ×
  stage stretch, then capped at 1.5 s.
- Linear fade over the **final 10%** of samples.
- **Peak-normalize** so the loudest sample sits at `120/127` of full u8 scale
  (peak absolute deviation from 128 ≈ 120).

### 7. Write
Emit mono / 8-bit unsigned / 10512 Hz PCM via the `wave` module to `output_path`.

## Edge cases (must not crash)

- **Unknown / missing primary type** → default profile row; valid WAV.
- **`types == []`** → default profile row; valid WAV; never crashes.
- **Single-form mons** → treated as `stage == 1` (stage transform is a no-op).
- **Any `stage >= 1`** is valid; even large stage values keep duration capped at
  1.5 s, so the frame count never exceeds the 1.55 s test bound.
- **Empty `line_name`** → still a valid seed via `sha256(b"")`; must not crash.
  (Assumption: the module does not special-case or reject empty names.)

## Errors

- The module raises no custom exceptions and validates no inputs beyond what the
  synthesis naturally requires; it is written so the documented edge cases do not
  raise. (Assumption: matching the project's writer/sprite modules, which do not
  add input-validation layers for the happy path.)
- I/O errors from `wave.open(output_path, "wb")` (e.g. a non-existent parent
  directory or an unwritable path) propagate naturally as the underlying
  `OSError`; the module does not catch or wrap them. Tests always write into a
  pytest `tmp_path`, so this path is not exercised. (Assumption.)
- `stage < 1` is documented as out of contract (the spec states `stage >= 1` is
  valid). The function is not required to guard against it, and tests will not
  pass `stage < 1`. (Assumption.)

## Constraints & dependencies

- **Standard library only**: `math`, `random`, `hashlib`, `wave`, optionally
  `struct` / `array`. No new third-party dependencies, no changes to
  `pyproject.toml`.
- **No ML / GPU**: no `torch`, `diffusers`, `transformers`, or any import that
  triggers them. Because the module is pure stdlib, its tests are **regular
  (non-`ml`) tests** and run everywhere, including the keep sandbox container.
- **Determinism**: seeded solely from `line_name` via `sha256` → a local
  `random.Random`; the global `random` state and all time/OS entropy are
  untouched. Same inputs → byte-identical output.
- **Fixed WAV format**: mono, 8-bit unsigned PCM, 10512 Hz.
- **Test file placement**: add `tests/test_cries.py` as a **regular** test file
  (imports no ML code, so **not** `@pytest.mark.ml` and **not** in
  `test_sprites_ml.py`, per `CLAUDE.md`'s test-slicing rules). It imports
  `fakemon_forge.cries`, writes WAVs into a pytest `tmp_path` (never into the
  repo tree), and reads them back with the `wave` module. `pytest` runs from the
  repo root (flat package layout).

### Tests to add (`tests/test_cries.py`)

Reading each WAV back with `wave`, assert:

1. **Format**: `getnchannels() == 1`, `getsampwidth() == 1`,
   `getframerate() == 10512`; `getnframes() / 10512` in `[0.35, 1.55]`; every
   byte a valid u8 in `[0, 255]`; peak absolute deviation from 128 ≈ 120 within
   ±3.
2. **Determinism**: two `generate_cry` calls with identical args → byte-identical
   files.
3. **Line motif / stage growth**: same `line_name` at stages 1, 2, 3 → durations
   **strictly increasing**.
4. **Type register**: a low-band type (e.g. `Fire`) vs a high-band type
   (e.g. `Fairy`) land in their respective registers — measured via
   **zero-crossing rate** (ZCR) over the sustained portion (higher band → higher
   ZCR).
5. **Stage pitch**: for the same line, stage 3's ZCR median is **below** stage
   1's (lower pitch per stage).
6. **Edge cases**: an empty `types` list and an unknown type name each produce a
   valid WAV without raising.

## Assumptions

Each item below is a default chosen for this headless spec, **not** confirmed by
existing code, tests, or docs — except where it restates the task's stated
approach.

- **[Restates task]** Seed source is `sha256(line_name.encode())` fed into a
  local `random.Random`; the global `random` module is never touched.
- **[Restates task]** The default profile row uses the spec's `(unknown)`
  values: band 300–900 Hz, syllable weights 2,3,1,0, noise 0.08, AM 0.40, dur
  mult 1.0.
- **[Restates task]** Profile selection keys off `types[0]`; any secondary-type
  timbre influence is optional and left to implementer discretion so long as it
  stays deterministic.
- **[Default]** Type-name matching is exact and case-sensitive against the
  capitalized profile keys (`"Fire"`, `"Fairy"`, …), consistent with
  `sprites.py`'s `_TYPE_TAGS`. A differently-cased or unknown name falls to the
  default row.
- **[Default]** Peak-normalization target 120 is validated within ±3 to absorb
  u8 integer-rounding.
- **[Default]** `output_path` is the full destination path (including filename);
  the function writes exactly there and does not create parent directories or
  append a `cry.wav` subpath — callers pass the final path, matching the
  `sprites.py`/`writer.py` `output_path` convention.
- **[Default]** Empty `line_name` is a valid seed (`sha256(b"")`) and does not
  crash; it is not special-cased.
- **[Default]** No input validation / custom exceptions beyond what synthesis
  requires; the documented edge cases are handled so they don't raise, and
  underlying `OSError`s from `wave.open` propagate unwrapped.
- **[Default]** `stage < 1` is out of contract (spec says `stage >= 1`); not
  guarded and not tested.
- **[Default]** The numeric synthesis tables are audition-validated tunable
  defaults; the implementer may adjust them by ear provided the tests above still
  pass. Concrete draw distributions within the stated ranges (e.g. exact
  weighting math, which source/contour is chosen) are implementation detail so
  long as output stays deterministic and within all asserted bounds.
- **[Default, scoping]** This is a self-contained single-slice module; no other
  file is touched. The `main.py` wiring is explicitly deferred to slice 2/2 of
  #22.
