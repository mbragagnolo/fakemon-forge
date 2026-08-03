# QA Checklist: BST hardening + 2-stage evolution lines

> Tracking: #59 · Spec: `bst-hardening-spec.md` · Plan: `tasks/PLAN.md`
> Branch: `feature/59-bst-hardening-2stage` (commits `6a63624`..`8101005`)

**Verified programmatically on 2026-08-03 — 25/25 checks passed.** Every box
below was executed rather than walked through by hand, at the user's request.
Two execution styles were used:

- **subprocess** — the real CLI (`python -m fakemon_forge.main`), real argparse,
  real exit codes. Used for everything that fails before the network is touched,
  so nothing is faked at all.
- **in-process** — `main()` with only the Mistral client and the sprite/audio
  calls faked, writing to a real working directory. The artifacts inspected are
  the ones a real run puts on disk.

The harness is not committed; the durable coverage is the 660-test suite. What
it adds over that suite is the subprocess layer — real process exit codes and
`--help` output, which the in-process tests cannot observe.

## Setup

- [x] Repo root is the working dir, so the flat `fakemon_forge/` package imports
      without installing → `pytest` and `python -m fakemon_forge.main` both work.
- [x] `MISTRAL_API_KEY` set for the runs that reach the client; unset
      deliberately for X7.
- [x] No GPU or network needed: the ML entry points are patched in `main`'s
      namespace, so `sprites.py`'s function-local `import torch` never runs.

## Happy path

- [x] **H1** `--mode line --stages 2` → exactly two directories,
      `stage1_<Name>` / `stage2_<Name2>`, each with `stats.json`, `entry.md`
      and a `<Name>.ini` beginning `[Pokemon]`.
- [x] **H2** `--mode line` with no `--stages` → three directories. *The default
      is preserved; no existing invocation changes shape.*
- [x] **H3** `--mode single` → one directory.
- [x] **H4** `stats.json` carries exactly the ten injector-contract keys —
      no key added, renamed or dropped by this change.
- [x] **H5** Every stage's `.ini` carries `PokemonName`, `BaseStats`,
      `PokedexDescription`, and the `Hght`/`Wght` from `stats.json`.

## Prompt correctness — the actual point of #59

- [x] **P1** 3-stage prompt reads `BST targets: stage 1 ~295, stage 2 ~405,
      stage 3 ~518.` and contains none of the stale 300/420/520.
- [x] **P2** 2-stage prompt reads `stage 1 ~305, stage 2 ~468.`, says "two
      evolutionary stages (stages 1 and 2)", and describes **no** adolescent
      form → a 2-stage line is never asked for a middle stage it won't generate.
- [x] **P3** `--mode single` prompts `BST target: ~430.` — the standalone
      value, not the juvenile 300 it used to send. *This is the user-visible
      correction: single forms were being built on a stage-1 stat budget.*
- [x] **P4** `--tier pseudo --mode line` still prompts 300/420/600 with the
      pseudo-legendary lore note → validated as already correct, not changed.

## Edge cases

- [x] **E1** A 2-stage final with no model-supplied size defaults to
      17 dm / 600 hg — the stage-3 row, not the 3-stage middle row.
- [x] **E2** A 2-stage line's footprints scale `[0.6, 0.9]`, skipping the 0.75
      middle → the final form doesn't print a juvenile's footprint.
- [x] **E3** A model returning **3** stages when 2 were requested → three
      directories written and exactly **one** API call. Accepted, not retried
      or truncated. *Deliberate non-goal; pinned so it isn't "fixed" later.*
- [x] **E4** `tests/fixtures/gen3_bst_bands.json` holds aggregate numbers only
      — every leaf an int, and legendary + mythical + box_legendary summing to
      `n=21`, the placeholder-filter check (an unfiltered regeneration reads 46).
- [x] **E5** Every `standard` target sits inside its band's `[p10, p90]` *and*
      equals the band median exactly.

## Error paths

Each verified as a real subprocess: exit code, stderr text, and no traceback.

- [x] **X1** `--mode single --stages 2` → exit 1,
      `Error: --stages applies only to --mode line; a single form is one stage.`
- [x] **X2** `--tier pseudo --stages 2` → exit 1,
      `Error: --tier pseudo is always a 3-stage line; --stages 2 is not valid.`
- [x] **X3** `--stages 1`, `--stages 4`, `--stages 0` → exit 2, argparse
      `invalid choice`. *Confirms `--stages 1` is not a second way to say
      `--mode single`.*
- [x] **X4** `--tier legendary|mythical --mode line` → exit 1, unchanged.
- [x] **X5** `--tier pseudo --mode single` → exit 1, unchanged.
- [x] **X6** Neither `--image` nor `--description` → exit 1, reported before
      any shape rule.
- [x] **X7** `MISTRAL_API_KEY` unset → exit 1, before any API call.
- [x] **X8** A rejected run leaves **no** `output/` directory behind.

## Regression spot-checks

- [x] **R1** `--help` lists `--stages` with its `{2,3}` choices and default.
- [x] **R2** `tests/test_stages_e2e.py` passes with `torch` never entering
      `sys.modules` → runs in the keep sandbox, per `CLAUDE.md`.
- [x] **R3** Full suite green from the repo root: **660 passed** (648 before
      this issue's final task, +12 new e2e tests).

## Not covered here

- **Real sprite/cry generation.** Needs the GPU and the ML stack; out of scope
  for this issue, which changes prompt text and constant lookups only. Worth one
  manual `--mode line --stages 2` run on the host before merge, to confirm a
  2-stage line renders two sprite sets.
- **Actual model output quality** — whether the corrected BST hints really move
  the returned stat spreads. Only observable against the live API, and it is a
  judgement call rather than a pass/fail check.
