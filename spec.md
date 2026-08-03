# Spec: Update README and sweep for leftover SD1.5 references

## Summary

Final slice (9/9) of the #61 sprite-generation retooling. All code for the
NoobAI-XL + back&front LoRA backend, k-centroid downscaling, connectivity-based
keying, content-aware front/back split, squash-init frame 2, ported chibi icon,
and `--image` mode is already merged (verified in `fakemon_forge/sprites.py`
and elsewhere — see Assumptions). This slice is documentation-only: bring
`README.md`'s "Sprite generation" section, its Dependencies table, and its
"first run downloads" note in line with what's actually shipped, and confirm
(via grep) that no leftover SD1.5-era identifiers survive anywhere the earlier
slices should have removed them from.

"Done and correct" means: `README.md` describes the NoobAI-XL 1.1 + back&front
LoRA pipeline end-to-end (model, one-call front+back generation, k-centroid
downscale, manual LoRA download step, license note), the Dependencies table
matches `pyproject.toml`, and a repo-wide grep for the old-backend identifiers
listed in the issue turns up nothing outside git history and the one
intentionally-historical research document.

## Inputs

- Current `README.md` (root).
- Current `pyproject.toml` `[project.dependencies]`.
- Current `fakemon_forge/sprites.py` (source of truth for what the shipped
  backend actually does — model id, LoRA path/scale, generation entry points).
- `.gitignore` (confirms `models/` — and therefore the LoRA weight file — is
  never committed).
- The issue's list of grep targets: `dreamshaper`, `pksp768`, `compel`,
  `_TYPE_TAGS`, `_encode_prompt`, `lambdalabs/sd-pokemon-diffusers`,
  `DPMSolverMultistepScheduler`, and the literal path
  `models/loras/pksp768_V2-1.safetensors`.

## Outputs

- `README.md`, edited in place:
  - "How it works" step 3 ("Sprite generation") rewritten to describe the
    shipped backend.
  - A new subsection (or paragraph under Installation) documenting the manual
    LoRA download step.
  - A new license note about NoobAI-XL's Fair-AI/no-commercialisation clause.
  - The "Dependencies" table's `diffusers` row reworded if "Stable Diffusion"
    is misleading; `compel` row removal (see Edge cases — already absent).
  - The "first run downloads ~1.7GB" note updated to the real model identity
    and size.
- No other files are modified by this slice — the grep sweep is a
  verification pass, expected to be a no-op against `fakemon_forge/`,
  `tests/`, and `pyproject.toml` (see Behavior). If it is *not* a no-op
  against actual code, that finding is in-scope to fix here (see Edge cases).

## Behavior

### 1. Rewrite "Sprite generation" (README.md step 3, currently line 9)

Replace the `lambdalabs/sd-pokemon-diffusers` / "downsampled to 96×96" description
with a description matching `fakemon_forge/sprites.py`'s actual pipeline:

- Base model: NoobAI-XL 1.1 (`Laxhar/noobai-XL-1.1` — `_BASE_MODEL_ID`).
- LoRA: the Pokemon Sprite XL PixelArt **back&front** LoRA, fused at scale 0.7
  (`_LORA_SCALE`), loaded from `models/loras/pkspbf_nb_v1.safetensors`.
- One `txt2img` call renders a single `1536×768` canvas (`_PAIR_WIDTH` ×
  `_GEN_SIZE`) with the front sprite on the left half and the back sprite on
  the right half (per `generate_sprite_pair`), rather than the old separate
  front-generation + img2img-backside chain.
- The canvas is split content-aware (`split_front_back_canvas`, with a reroll
  + naive-midline fallback), then each half is downscaled to the final sprite
  size via **k-centroid** downscaling (`k_centroid`, not nearest-neighbour),
  which is what actually produces the Gen-3-style palette/pixel-grid look —
  not "downsampled to 96×96" as the current text implies for the whole
  pipeline (the native render/working size is 768; only `stitch_spritesheet`'s
  64px sheet cells are an actual downscale in the current code).
- Palette-quantised to 16 colours via the Gen-3 contract (`_quantize_gen3`) —
  this part of the existing description is still accurate and should be kept.

### 2. Document the LoRA manual-download step

State, near Installation (or as its own subsection under "Installation"):

- The LoRA weight file is **not** auto-downloaded by `pip install` or by the
  tool at runtime, and it is **never committed** to the repo (`models/` is
  gitignored — cite `.gitignore` line `models/`).
- It must be downloaded manually from Civitai, model id **378602** ("back&front
  Noob v1"), which requires a Civitai login.
- It must be placed at `models/loras/pkspbf_nb_v1.safetensors` (relative to
  the repo root — matches `_LORA_PATH = Path(__file__).parent.parent / "models"
  / "loras" / "pkspbf_nb_v1.safetensors"`) before running the tool.
- Running without the file present fails at pipeline-load time (`_apply_lora`
  calls `StableDiffusionXLLoraLoaderMixin.lora_state_dict` against a
  nonexistent path) — worth one sentence so the failure mode is legible to a
  first-time reader, but no new error handling is being added in this slice.

### 3. Add a license note

Plainly state:

- NoobAI-XL is distributed under a Fair-AI license carrying a
  no-commercialisation clause.
- This is acceptable for this project because fakemon-forge is a
  non-monetised public portfolio project.
- If that clause is ever a problem, the documented fallback is the SDXL-base
  LoRA variant published on the same Civitai model page (378602) — no code
  change is implied or required by mentioning this; it's a documented escape
  hatch, not a supported `--flag`.

Do not claim GPL/MIT/OpenRAIL-equivalent freedom, and do not imply the tool
itself is licensed any differently — this note is scoped to the model/LoRA
weights only. Keep the project's own `LICENSE` section untouched.

### 4. Update "Dependencies" table and "~1.7GB" download note

- Re-verify against `pyproject.toml`: current dependencies are `mistralai`,
  `diffusers`, `transformers`, `accelerate`, `Pillow`, `torch` — no `compel`
  entry exists in `pyproject.toml` or in the current README table (see Edge
  cases: this part of the issue is already satisfied, not a new removal).
  Leave the row-removal a no-op; do not delete any row that's already gone.
- Reword the `diffusers` row's "Purpose" cell (currently "Stable Diffusion
  sprite generation") to name the actual pipeline, e.g. something like "SDXL
  sprite generation (NoobAI-XL + LoRA)" — exact wording is an implementation
  choice for whoever writes the prose, not fixed by this spec.
- Update the "first run downloads ~1.7 GB" sentence (currently referencing
  `lambdalabs/sd-pokemon-diffusers`) to name `Laxhar/noobai-XL-1.1` and its
  real download size. The exact GB figure for an SDXL base checkpoint is not
  independently confirmed by anything in this repo (no fixture pins it) — see
  Assumptions for the placeholder figure to use and how to hedge it.

### 5. Grep sweep (verification, not a rewrite)

Grep the whole repo (tracked files) for each of:
`dreamshaper`, `pksp768`, `compel`, `_TYPE_TAGS`, `_encode_prompt`,
`lambdalabs/sd-pokemon-diffusers`, `DPMSolverMultistepScheduler`, and the
literal string `models/loras/pksp768_V2-1.safetensors`.

Already confirmed by investigation for this spec (re-run as part of
implementation to catch drift, but no further code fix is expected):

- **Zero hits** in `fakemon_forge/*.py`, `tests/*.py`, and `pyproject.toml`
  for every one of the above identifiers. The old backend is fully gone from
  code.
- **Two hits in `README.md`** for `lambdalabs/sd-pokemon-diffusers` (the two
  lines this slice rewrites — see Behavior §1 and §4).
- **Hits in `research-sprite-generation.md`** for `dreamshaper` and
  `pksp768` (as `pksp768_V2-1`, `pksp768`) — this is the dated discovery
  document that *kicked off* issue #61 by describing the old stack as
  historical context at time of writing (2026-08-03). It is not code, README,
  or `pyproject.toml`, and rewriting a dated research artifact to describe a
  stack it predates would falsify the historical record. Per the issue's own
  scope ("code, comments, README, or pyproject.toml"), this file is out of
  scope — see Assumptions.
- **One hit in `spec.md`** itself (this file, prior slice's leftover content,
  referencing `_stub_encode_prompt` — a *test helper* naming pattern, not the
  real `_encode_prompt` the issue asks about, and this file is about to be
  fully overwritten by this slice's own spec anyway).

If a fresh run of the sweep at implementation time turns up any hit inside
`fakemon_forge/` (an actual code path, not docs/comments) that this
investigation missed, fix it in that same slice and call it out explicitly in
the PR body, per the issue's instructions — do not silently leave it.

## Edge cases

- **`compel` row already absent from both files.** The issue says "remove the
  `compel` row if one still exists" — it doesn't, in either `pyproject.toml`
  or `README.md`'s current Dependencies table. Treat this as already done;
  no edit needed for it specifically (distinct from the `diffusers` row
  reword, which *is* still needed).
- **`models/loras/pksp768_V2-1.safetensors` path.** Confirmed absent from
  every tracked file (only the bare model name `pksp768_V2-1`, no path,
  appears in `research-sprite-generation.md`). The issue frames this as "a
  check for stray path references, not a file deletion" — the check passes;
  nothing to change.
- **`research-sprite-generation.md`'s historical references.** See Behavior
  §5 — treated as intentionally out of scope. Flagged explicitly rather than
  silently skipped, per the "Assumptions" instruction below.
- **Public-repo wording constraint.** The new README prose must not imply
  real Pokémon sprites/assets are shipped, referenced as training data
  provenance for the user, or bundled — stay within the existing "GBA-style" /
  "Pokémon-like" framing already used elsewhere in the README. The LoRA name
  ("Pokemon Sprite XL PixelArt") and Civitai model name are third-party
  proper nouns being cited for attribution/download purposes, not a claim
  about this repo's own outputs — write around them carefully (e.g. don't
  say "generates Pokémon sprites"; keep saying it generates original
  creatures in a Pokémon-*like* GBA style).

## Errors

Not applicable — this slice changes no runtime code path, so no new error
conditions are introduced. (The existing `_apply_lora` failure when the LoRA
file is missing already exists in shipped code from an earlier slice; this
slice only documents that precondition in prose, per Behavior §2.)

## Constraints & dependencies

- No implementation code changes are expected. If the grep sweep at
  implementation time finds a genuine leftover code path, fixing it is
  in-scope for *that* slice's implementation phase, but is not anticipated by
  this investigation (see Behavior §5).
- Must not alter `LICENSE` (project license) — only add a note about the
  *model weights'* license inside the README body.
- Must not change any test file; no test in the suite parses or asserts on
  README content (confirmed: the two `README` mentions in `tests/` are
  descriptive comments only, not assertions against the file).
- Table/prose wording choices (exact GB figure, exact "Purpose" cell text)
  are left to implementation-time judgement within the constraints stated
  above — this spec fixes the *facts* that must appear, not the exact prose.

## Assumptions

- **Slices 1-8 are fully merged into the working tree as given.** Verified by
  reading `fakemon_forge/sprites.py` directly rather than trusting the issue
  text: `_BASE_MODEL_ID = "Laxhar/noobai-XL-1.1"`, `_LORA_PATH` pointing at
  `pkspbf_nb_v1.safetensors`, `_LORA_SCALE = 0.7`, `k_centroid`,
  `generate_sprite_pair` / `split_front_back_canvas`, `procedural_squash` /
  `build_frame2`, and `_flatten_background_to_key` (connectivity-based keying)
  are all present and match the issue's description of what should already be
  shipped.
- **`research-sprite-generation.md` is out of scope for editing.** Assumption,
  not confirmed by the issue text (which doesn't mention this file at all).
  Chosen because the file is explicitly dated/historical ("discovery,
  2026-08-03") and predates the retooling it describes; treating dated
  research notes as a live-documentation surface would be inconsistent with
  the file's own stated purpose ("Pure discovery, no spec/code"). If this
  assumption is wrong, the fix is a one-line scope addition, not a design
  change.
- **Exact "~X GB" download-size figure for `Laxhar/noobai-XL-1.1`.** Nothing
  in this repo (fixtures, tests, comments) pins the real download size of the
  NoobAI-XL 1.1 checkpoint. Default: state it as "several GB" or carry over
  a close approximate figure for a standard SDXL-family checkpoint (typically
  ~6.5-7 GB for the base UNet+VAE+encoders in fp32, less if only fp16 shards
  download) rather than inventing false precision. Whoever implements this
  slice should independently confirm the real Hugging Face repo size if
  precision matters more than this spec assumes; using a hedge word ("about")
  is an acceptable default rather than blocking the slice on an unverifiable
  number.
- **"Sprite generation" step numbering/heading stays at position 3** in "How
  it works" — this slice only rewrites step 3's body text, not the surrounding
  structure (steps 1, 2, 4 are untouched and don't reference the sprite
  backend).
- **New LoRA-download and license-note content are placed under
  "Installation"**, either as new paragraphs or a small new subsection (e.g.
  "### LoRA weights" / "### License note on model weights"), rather than
  inline in "How it works" step 3 — keeps step 3 focused on describing the
  pipeline and keeps the actionable "you must do this before running" content
  next to the rest of the setup instructions. This is a structural default,
  not dictated by the issue text.
- **This is a single coherent slice**, not something to split further — it's
  a documentation-only change plus a verification grep with no code-path
  fallout found, well within one focused change.
