# Spec: Chibi img2img pass for `sprite_small.png` (party-menu icon), falling back to the plain downscale

## Summary

`fakemon-forge` writes, per stage, a party-menu icon `sprite_small.png` — the
32×64 two-frame, opaque-teal-background Gen-3 icon — by downscaling the front
sprite `sprite.png` through `generate_icon` (pure Pillow, no torch). Official
Gen-3 party icons are **chibi caricatures** (big head, small body), not literal
miniatures of the battle sprite. This slice (3/3 of #21, issue #28) inserts an
SD img2img pass **before** the downscale: it renders a chibi-proportioned 768px
variant of the front sprite (`sprite_chibi.png`) and feeds *that* into
`generate_icon`, so the icon reads as a proper caricature.

The chibi pass is an **enhancement, not a dependency**. If the img2img render
raises, the flow falls back to feeding `sprite.png` straight into
`generate_icon` — exactly today's behavior. An icon must always exist whenever
`sprite.png` exists.

The entire change is confined to `fakemon_forge/main.py`, inside the existing
per-stage loop, and to test files. No change to `sprites.py`, `icon.py`,
`writer.py`, or `export_ini.py`.

### Explicitly out of scope

- **`sprites.py` / `icon.py` code** — both are reused as-is.
  `generate_sprite_img2img` already renders an img2img variant and saves a
  `P`-mode PNG; `generate_icon` already downscales a `P`-mode source to the
  icon. No signature or behaviour change to either.
- **The spritesheet** — `sprite_chibi.png` is an intermediate render, **not**
  added to `_SHEET_LAYOUT` / `spritesheet.png` (just like `sprite_small.png`
  is already excluded).
- **`.ini` / writer changes** — neither references sprite filenames.
- **Tuning the chibi look** — the tag set and `strength` are prototype tunables
  pending a GPU spike (see Assumptions); this slice wires the pass and its
  fallback, it does not verify the LoRA actually produces caricature
  proportions.

## Inputs

No new CLI arguments; no changed function signatures. Inside the per-stage
`for stage, stage_dir in zip(stages, stage_dirs)` loop, the icon block reuses
values already in scope:

- `seed` — the per-stage `random.randint(0, 2**32 - 1)` already computed at the
  top of the loop; reused so the chibi render is reproducible with the rest of
  the stage's sprites.
- `stage["sprite_prompt"]`, `stage["types"]`, `stage["name"]` — the stage's
  prompt/types/name.
- `sprite_path` — `str(stage_dir / "sprite.png")`, the front sprite written
  earlier in the same iteration (a `P`-mode Gen-3-contract sprite).
- `img2img_pipeline` — the pipeline used for the existing back-sprite / frame-2
  img2img calls.

New module-level constant in `main.py`:

- `_CHIBI_TAGS = ["chibi", "big head", "small body"]` — the `extra_tags` passed
  to the chibi img2img render. A single obvious tunable with a comment that it
  needs a GPU spike to confirm the LoRA produces caricature proportions from
  img2img. **[picked]** — see Assumptions.

New per-stage local paths (derived from `stage_dir`):

- `small_path = str(stage_dir / "sprite_small.png")` — the icon output (renamed
  from today's `icon_path`, or kept as-is; the name is immaterial).
- `chibi_path = str(stage_dir / "sprite_chibi.png")` — the persisted chibi
  render (intermediate; kept on disk for debuggability, matching how every
  other view is persisted).

## Outputs

Per stage, in addition to the files written today:

- **`sprite_chibi.png`** — a 768×768 `P`-mode PNG (adaptive palette) produced by
  `generate_sprite_img2img` from `sprite.png`, written whenever the chibi render
  succeeds. Not part of the spritesheet.
- **`sprite_small.png`** — the 32×64 party-menu icon, now derived from
  `sprite_chibi.png` on the happy path, or from `sprite.png` on the fallback
  path. Same format/contract as today (opaque teal background, ≤16 colours,
  two stacked 32×32 frames).

When the chibi render fails, `sprite_chibi.png` may be absent (or partially
written by the failing call) and `sprite_small.png` is the plain downscale from
`sprite.png` — byte-for-byte today's output. When `generate_icon` itself fails,
`sprite_small.png` may be absent and a warning is printed (unchanged from
today).

## Behavior

The current single-call icon block:

```
icon_path = str(stage_dir / "sprite_small.png")
try:
    generate_icon(sprite_path, icon_path)
except Exception as exc:
    print(f"Warning: icon generation failed for {stage['name']}: {exc}", file=sys.stderr)
```

becomes a chibi-first flow inside the **same** outer warn-and-continue
`try/except`:

```
small_path = str(stage_dir / "sprite_small.png")
chibi_path = str(stage_dir / "sprite_chibi.png")
try:
    try:
        # Chibi caricature enhancement: render a big-head/small-body variant of
        # the front sprite, then downscale THAT into the party-menu icon.
        generate_sprite_img2img(
            stage["sprite_prompt"], stage["types"], sprite_path, chibi_path,
            pipeline=img2img_pipeline, extra_tags=_CHIBI_TAGS, seed=seed,
        )
    except Exception:
        # Enhancement is optional: fall back to the plain downscale of sprite.png.
        icon_source = sprite_path
    else:
        icon_source = chibi_path
    generate_icon(icon_source, small_path)
except Exception as exc:
    print(f"Warning: icon generation failed for {stage['name']}: {exc}", file=sys.stderr)
```

Step by step:

1. **Chibi render** — call `generate_sprite_img2img(stage["sprite_prompt"],
   stage["types"], sprite_path, chibi_path, pipeline=img2img_pipeline,
   extra_tags=_CHIBI_TAGS, seed=seed)`. `reference_path` is **omitted** (defaults
   to `None`), so the chibi render is quantized adaptively via `postprocess` and
   gets its **own** palette — the icon applies its own palette policy downstream,
   so the chibi render needn't share the front sprite's palette. `strength` is
   omitted, i.e. the `generate_sprite_img2img` default of `0.8` (a documented
   tunable — do not invent a tuned value here). This writes `sprite_chibi.png`.
2. **Downscale from chibi** — on success, `icon_source = chibi_path` and
   `generate_icon(chibi_path, small_path)` builds the icon from the chibi render.
3. **Fallback** — if the chibi img2img call (step 1) raises, `icon_source =
   sprite_path`, so `generate_icon(sprite_path, small_path)` runs — the plain
   downscale from the front sprite. This is precisely today's behavior. The stage
   does **not** abort.
4. **`generate_icon` failure** — `generate_icon` runs exactly once per iteration
   (on `chibi_path` or `sprite_path`) and is inside the outer `try`. If it raises
   — whether on the chibi render or on the fallback source — the outer
   warn-and-continue prints `Warning: icon generation failed for {stage['name']}:
   {exc}` and the stage continues to its later blocks (back sprite, frame 2,
   shinies, spritesheet), unchanged from today.

Key scoping detail: the **inner** `try/except` wraps *only* the chibi img2img
call, so a failure there triggers the fallback (not a warning). The **outer**
`try/except` wraps the whole block, so a `generate_icon` failure (on either
source) triggers the warning (not a fallback). This matches the parent issue's
steps 3 and 4 exactly.

Ordering: the icon block stays where it is today — after the front-sprite block
(so `sprite.png` exists) and before the back-sprite block. `sprite_chibi.png`
is persisted alongside the other views but is not referenced by
`stitch_spritesheet` (it is not in `_SHEET_LAYOUT`).

## Edge cases

- **Front sprite generation failed** → the front-sprite block `continue`s to the
  next stage before reaching the icon block, so neither the chibi render nor the
  fallback ever runs against a missing `sprite.png`.
- **Chibi img2img raises** (pipeline crash, torch/OOM, bad init, etc.) → caught
  by the inner `except`; icon is built from `sprite.png` (plain downscale). No
  warning; the stage proceeds. `sprite_chibi.png` may be missing or partial —
  harmless, since it is not consumed further.
- **Chibi render succeeds but `generate_icon(chibi_path, …)` raises** (e.g. the
  chibi render is somehow not `P`-mode, or unreadable) → the outer `except`
  warns; there is **no** second attempt from `sprite.png` (the fallback is scoped
  to the img2img call only, per issue step 4). This is an accepted trade-off: the
  chibi render from `generate_sprite_img2img` is `P`-mode by construction
  (`postprocess`), so `generate_icon`'s `P`-mode precondition holds on the happy
  path.
- **Fallback `generate_icon(sprite_path, …)` raises** → outer `except` warns
  (same as today's single-call behavior when the plain downscale fails).
- **Line mode (3 stages)** → the icon block is inside the per-stage loop; each
  stage renders its own `sprite_chibi.png` from its own `sprite.png` with its own
  `seed`, and independently falls back per stage.
- **`sprite_chibi.png` is not in the spritesheet** → `_SHEET_LAYOUT` is unchanged
  (still 6 entries); the intermediate render never leaks into `spritesheet.png`.

## Errors

- `generate_sprite_img2img` surfaces exceptions to the caller; here they are
  swallowed by the **inner** `except` and converted into the fallback (no message
  printed — a failed enhancement is expected/tolerable, not a user-facing
  warning).
- `generate_icon` raises `ValueError` if its source is not `P`-mode, and may
  raise on I/O errors; these hit the **outer** `except` and warn
  `Warning: icon generation failed for {stage['name']}: {exc}` — unchanged
  wording and structure.
- No new `sys.exit` paths. Pipeline-load failure paths are unchanged.

## Constraints & dependencies

- The change lives entirely in `fakemon_forge/main.py` (the per-stage icon block
  plus one module-level `_CHIBI_TAGS` constant) and in the test files. It reuses
  the existing `generate_sprite_img2img` (already imported into `main.py`) and
  `generate_icon` (imported as the bare name `generate_icon`, patched in tests as
  `fakemon_forge.main.generate_icon`). Nothing is duplicated.
- `generate_sprite_img2img` performs a function-local `import torch` (via
  `_run_img2img` → `_make_generator`), so **any test that actually calls it is an
  `ml` test** and belongs in `tests/test_sprites_ml.py` (or carries
  `@pytest.mark.ml`), per `CLAUDE.md`. In `tests/test_main.py`,
  `generate_sprite_img2img` is **mocked**, so the `main`-level wiring for the
  chibi pass is torch-free and testable in the slim sandbox.
- `_CHIBI_TAGS` is the single, obvious place to tune the caricature tags; keep
  the GPU-spike comment next to it.
- Backward compatibility: `generate_sprite_img2img` and `generate_icon`
  signatures are untouched; the chibi call omits `reference_path` (default
  `None` = adaptive palette) and omits `strength` (default `0.8`).

## Tests

Per `CLAUDE.md`'s test slicing:

### `tests/test_main.py` (torch-free — collaborators mocked)

The chibi img2img call is exercised through the already-mocked
`generate_sprite_img2img`, so all of this is torch-free.

- **Happy path (chibi feeds the icon)**: on `--description …`, assert
  `generate_sprite_img2img` is called to produce `sprite_chibi.png` — i.e. a call
  whose `output_path` (positional arg index 3) is
  `str(stage_dir / "sprite_chibi.png")`, whose init `image_path` (positional arg
  index 2) is `str(stage_dir / "sprite.png")`, whose `extra_tags` equals
  `_CHIBI_TAGS` (`["chibi", "big head", "small body"]`), and whose `seed` is
  passed. Assert `generate_icon` is called with
  `(str(stage_dir / "sprite_chibi.png"), str(stage_dir / "sprite_small.png"))`.
- **Fallback (chibi render raises)**: give the chibi `generate_sprite_img2img`
  call a `side_effect` that raises (scoped to the chibi call — e.g. a
  side-effect function that raises when `output_path` ends in `sprite_chibi.png`
  and otherwise returns a `MagicMock`, since `generate_sprite_img2img` is also
  used for the back sprite). Assert `generate_icon` is still called with the
  **fallback** source `(str(stage_dir / "sprite.png"),
  str(stage_dir / "sprite_small.png"))`, that the stage does not abort (later
  blocks such as the spritesheet still run), and that **no** icon warning is
  printed for the fallback case.
- **`generate_icon` failure still warns**: the existing
  `test_icon_failure_warns_but_does_not_exit` remains valid — a `generate_icon`
  side-effect raise must still produce `Warning: … Flamburr …` and not skip the
  spritesheet stitch.

### Existing `tests/test_main.py` assertions that MUST be updated

There is now an **extra** `generate_sprite_img2img` call per stage (the chibi
render), so raw call-count assertions and `reference_path`-based filters change:

- `test_txt2img_path_calls_generate_sprite` — `ctx["sprite_i2i"]` is no longer
  called once; the txt2img path now has **2** img2img calls (back + chibi).
  Replace `assert_called_once()` / `call_args` with a filter for the back call by
  `extra_tags == ["backside"]` (rather than by raw count), and optionally assert
  a chibi call with `extra_tags == _CHIBI_TAGS` exists.
- `test_img2img_path_calls_generate_sprite_img2img` — `call_count` is now **3**
  (front + back + chibi), not 2. Update the count (or, better, filter by the
  distinguishing kwargs below).
- `test_img2img_back_sprite_inits_from_front_sprite` and
  `test_txt2img_back_sprite_reference_is_frame1` — these filter
  `call_args_list` by `reference_path`. The chibi call also has
  `reference_path is None` (like the front call), so `reference_path is None` no
  longer uniquely identifies the front call. Distinguish the three calls by
  `extra_tags`:
  - **front** (img2img path only): `extra_tags` absent/`None`;
  - **chibi**: `extra_tags == _CHIBI_TAGS` (init `sprite.png`, `reference_path`
    `None`);
  - **back**: `extra_tags == ["backside"]` (`reference_path == sprite.png`,
    `strength == 0.65`).
  Rewrite the front/back filters to select by `extra_tags` accordingly, keeping
  the existing `strength == 0.65`, `reference_path == sprite.png`, and
  init-image assertions on the back call.
- `test_icon_generated_once_per_stage` — the icon's source path changes from
  `sprite.png` to `sprite_chibi.png` on the happy path. Update the expected
  `generate_icon` args to `(str(stage_dir / "sprite_chibi.png"),
  str(stage_dir / "sprite_small.png"))` (icon is still generated exactly once
  per stage).
- `test_icon_generated_three_times_in_line_mode` — still 3 icon calls; no change
  beyond the source path being `sprite_chibi.png` if that test inspects args
  (it currently only counts, so likely unchanged).

### `tests/test_sprites_ml.py` (auto-skipped without torch) — optional

The chibi img2img primitive is `generate_sprite_img2img` with an adaptive
palette (`reference_path=None`), which is **already** covered by existing ml
tests (front-sprite img2img). No new ml test is strictly required, since this
slice adds no new `sprites.py` code. Optionally add an ml test asserting the
chibi-tagged call (`extra_tags=_CHIBI_TAGS`, no `reference_path`) writes a
`P`-mode 768×768 PNG, to document the chibi render's contract — but this only
duplicates the existing adaptive-`postprocess` coverage.

## Assumptions

Items marked **[picked]** are defaults chosen here (not confirmed by existing
code/tests/docs); **[confirmed]** items are grounded in the codebase.

- **[picked]** **Chibi tag set** = `_CHIBI_TAGS = ["chibi", "big head",
  "small body"]`, defined as a module-level constant in `main.py` with a comment
  that it is a prototype tunable needing a GPU spike to confirm the LoRA produces
  caricature proportions from img2img. Placed in `main.py` (where the chibi call
  lives) rather than `sprites.py`, since it is a pipeline-wiring choice, not a
  sprite-generation primitive. The issue permits either location.
- **[picked]** **img2img `strength`** = the `generate_sprite_img2img` default of
  `0.8` (parameter omitted at the call site). A documented tunable; no tuned
  value is invented. A GPU spike may lower it later.
- **[picked]** **No `reference_path`** on the chibi call, so the chibi render
  gets its **own** adaptive palette via `postprocess`. The icon applies its own
  palette policy in `generate_icon` regardless of the source palette, so sharing
  the front sprite's palette would add nothing. (The issue explicitly directs
  omitting `reference_path`.)
- **[picked]** **`sprite_chibi.png` is persisted** to the stage dir (not written
  to a temp file), matching how the pipeline keeps every intermediate view for
  debuggability. It is **not** added to `_SHEET_LAYOUT`.
- **[picked]** **Fallback is scoped to the chibi img2img call only** via a nested
  `try/except`; a `generate_icon` failure (on either source) is handled by the
  outer warn-and-continue, with **no** re-attempt from `sprite.png`. This is the
  literal reading of the issue's steps 3–4. An alternative (retry the plain
  downscale if the chibi-sourced `generate_icon` also fails) was rejected as
  scope creep — the chibi render is `P`-mode by construction, so
  `generate_icon`'s precondition holds on the happy path.
- **[picked]** **No warning is printed when the chibi render fails** (the inner
  `except` is silent, only setting the fallback source). The parent issue frames
  the chibi pass as a silent enhancement ("silently falls back"), so a failed
  enhancement is not surfaced; only an outright icon-generation failure warns.
- **[confirmed]** The chibi render **reuses the stage's front-sprite `seed`**
  (already computed at the top of the loop) for reproducibility.
- **[confirmed]** `generate_sprite_img2img` writes a `P`-mode PNG (`postprocess`
  when `reference_path` is `None`), so the chibi render satisfies
  `generate_icon`'s `P`-mode precondition on success.
- **[confirmed]** `main.py` imports `generate_sprite_img2img` and `generate_icon`
  directly (bare names), patched in `test_main.py` as
  `fakemon_forge.main.generate_sprite_img2img` /
  `fakemon_forge.main.generate_icon`; so the chibi wiring is fully mockable
  without torch.
- **[confirmed]** `img2img_pipeline` is in scope in the per-stage loop and is the
  same pipeline used for the back-sprite / frame-2 img2img calls.
- **[confirmed]** Adding the chibi call increases the per-stage
  `generate_sprite_img2img` count by one (txt2img path: 1→2; img2img path: 2→3),
  which is why the existing call-count/`reference_path`-filter assertions in
  `test_main.py` must be updated to filter by `extra_tags`.
- **[confirmed]** `sprite_chibi.png` is excluded from `spritesheet.png` because
  `_SHEET_LAYOUT` (in `sprites.py`) lists only the six shipped views and is not
  modified by this slice.
