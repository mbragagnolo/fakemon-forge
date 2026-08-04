# Spec: run.json input manifest

## Summary

Every `fakemon-forge` invocation writes its stage outputs (`stats.json`,
`entry.md`, sprites, `.ini`, ...) under a per-run folder
(`output/<Name>[_N]/`), but nothing records what produced that folder. The
`--description` text, the `--image` path and its vision-derived description,
the `--mode`/`--tier`/`--stages` flags, and each stage's Mistral-generated
`sprite_prompt` are computed and then discarded once the run finishes.
Recovering that context today means reverse-engineering it from the sprite,
the typing, and the Pokédex entry — which is what `output/PROMPT-ARCHAEOLOGY.md`
had to do by hand across 84 existing runs.

This slice adds a `run.json` manifest, written once per run at the run
folder's root (sibling to `stage1_<Name>/`, `stage2_<Name>/`, ...), capturing
the run's *inputs*: what was asked for, what the vision step saw, what
produced it, and how to ask for it again. It is written as soon as the run
folder exists — before any stage subfolder, sprite, or other stage artifact
is created — so a run that dies partway through sprite generation still
leaves behind a manifest explaining what it was trying to do.

"Done and correct" means: for any completed or partially-completed run
(including one that died during sprite generation), `output/<Name>/run.json`
exists, is valid JSON, and alone is enough to (a) tell a human what inputs
produced this folder and (b) hand them a command to reproduce that same
request.

## Inputs

Values `write_output` needs in order to build `run.json`, all already
computed by `main.py` before `write_output` is called today:

| Field | Source in `main.py` today |
|---|---|
| description | `args.description` (may be `None`) |
| image path | `args.image` (may be `None`) |
| vision-derived text | `vision_desc` (`""` when no `--image`) |
| mode | `args.mode` (`"single"` \| `"line"`) |
| tier | `args.tier` (`"standard"` \| `"pseudo"` \| `"legendary"` \| `"mythical"`) |
| requested stage count | `args.stages`, only meaningful when `mode == "line"` (see `generate_fakemon`'s docstring: ignored for `"single"`) |
| generated stages | `forms`, the list of stage dicts already passed into `write_output` — each carries `stage`, `name`, and `sprite_prompt` |

Two more values are *not* passed in by `main.py` — they're read directly by
`write_output` at write time:

- the installed package version
- the current git commit sha of the fakemon-forge checkout

## Outputs

A single file, `run.json`, written to the run folder root:
`output/<Name>[_N]/run.json` (using the same collision-resolved directory
`_resolve_dir` already picks for the fakemon as a whole — the same directory
that becomes the parent of `stage1_<Name>/` etc.).

Shape:

```json
{
  "description": "a fire lizard with blue flames",
  "image": null,
  "vision_description": "",
  "mode": "line",
  "tier": "standard",
  "requested_stages": 3,
  "timestamp": "2026-08-04T18:22:31+00:00",
  "package_version": "0.1.0",
  "git_sha": "481da94",
  "rerun_command": "fakemon-forge --description \"a fire lizard with blue flames\" --mode line --tier standard --stages 3",
  "generated_stages": [
    {"stage": 1, "name": "Flamburr",  "sprite_prompt": "A small fire lizard, GBA pixel art, white background"},
    {"stage": 2, "name": "Flamburro", "sprite_prompt": "A medium fire lizard, more muscular, GBA pixel art"},
    {"stage": 3, "name": "Flamburron","sprite_prompt": "A large fire dragon, imposing, GBA pixel art"}
  ]
}
```

Field notes:

- `description` / `image`: exactly the CLI values, `null` when not given
  (not `""`) so "not provided" is distinguishable from "provided as empty".
- `vision_description`: the vision model's plain-English output when
  `--image` was given; `""` when it wasn't (mirrors `vision_desc`'s existing
  default in `main.py`).
- `requested_stages`: the `--stages` value in effect for the CLI invocation
  (explicit or the parser's default of 3), recorded only for
  `mode == "line"`; `null` for `mode == "single"`, where the flag is parsed
  but has no effect. The *actual* number of stages produced is not
  duplicated here as a second integer — it's always `len(generated_stages)`.
- `generated_stages`: one entry per stage actually returned by
  `generate_fakemon`, in stage order, each carrying that stage's
  `sprite_prompt` — the field this whole manifest exists to stop discarding.
- No outcome data: no sprite paths, no success/failure flags, no per-stage
  warnings. `run.json` describes what was asked for, not what happened
  during generation — matching the "Inputs only" instruction in the issue.

## Behavior

1. `write_output` creates the run folder (`fakemon_dir`, via `_resolve_dir` +
   `mkdir()`) exactly as it does today.
2. Immediately after that `mkdir()` — before creating any
   `stageN_<Name>/` subfolder — `write_output` builds and writes
   `run.json` into the run folder root.
3. `write_output` then proceeds with today's per-stage loop
   (`stats.json`, `entry.md`) unchanged.
4. `main.py`'s call site changes only in what it passes to `write_output`:
   alongside the existing `forms` list, it now also passes the run's inputs
   (description, image, vision_description, mode, tier, requested stage
   count) gathered from `args` and `vision_desc`, which it already has in
   scope at the point it calls `write_output` today.
5. `package_version` is resolved once per write: try
   `importlib.metadata.version("fakemon-forge")` first (the package's
   documented install path is `pip install -e .`, per `README.md`); if that
   raises `PackageNotFoundError` (e.g. running from an uninstalled
   checkout), fall back to reading the `version = "..."` line out of
   `pyproject.toml` directly, resolved relative to the package the same way
   `generator.py`/`export_ini.py` already locate `resources/`
   (`Path(__file__).parent.parent`); if that also fails, use `"unknown"`.
6. `git_sha` is resolved once per write by running
   `git rev-parse --short HEAD` with its working directory set to the repo
   root; on any failure (git not installed, not a git checkout, any
   non-zero exit) fall back to `"unknown"`.
7. `rerun_command` is assembled as a single-line, POSIX-shell-quoted string
   (`shlex.quote`-style, applied uniformly to every value so the output is
   predictable) invoking the documented `fakemon-forge` console-script
   entry point, with flags emitted in the same order `cli.py` defines them
   (`--image`, `--description`, `--mode`, `--tier`, `--stages`), omitting
   any flag that doesn't apply (no `--image` if none was given, no
   `--stages` for `mode == "single"`). `--stages` is always included
   explicitly for `mode == "line"` — even when the user didn't pass it and
   it came from the parser default — so the recorded command stays
   reproducible if the default ever changes in a later version.
8. `timestamp` is captured at the moment `run.json` is written (step 2
   above), as ISO 8601 with a UTC offset.

## Edge cases

- **`--image` given, no `--description`**: `description` is `null`,
  `vision_description` holds the vision model's text, `rerun_command`
  includes only `--image`.
- **`--description` given, no `--image`**: `image` is `null`,
  `vision_description` is `""`, `rerun_command` includes only
  `--description`.
- **Both given**: both are recorded independently; `rerun_command` includes
  both flags. (`main.py` already concatenates them into `combined` for the
  LLM call — `run.json` keeps them separate rather than only storing the
  merged text, so the original two inputs stay individually recoverable.)
- **`mode == "single"`**: `requested_stages` is `null`; `rerun_command`
  omits `--stages` entirely (matches `cli.py`'s own rejection of
  `--stages` with `--mode single`).
- **Run folder name collision** (`Flamburr` already exists → `Flamburr_2`):
  `run.json` is written into the resolved `Flamburr_2/`, not `Flamburr/` —
  it always lands in the same directory `write_output` actually creates and
  returns stage dirs under.
- **Sprite generation fails or is interrupted after `write_output`
  returns**: `run.json` already exists and is complete — this is the
  scenario the "written up front" requirement exists for. No partial/empty
  `run.json` is ever left behind, since it's written whole in one call.
- **`generate_fakemon` returns fewer/more stages than `requested_stages`**
  (an existing possible divergence, unrelated to this feature):
  `generated_stages` reflects reality (whatever `forms` actually contains);
  `requested_stages` and `rerun_command`'s `--stages` still reflect what was
  asked for. The two are allowed to disagree — that disagreement is itself
  informative, not a bug in the manifest.
- **Description or image path contains characters needing shell escaping**
  (quotes, spaces, `$`, backticks, ...): `rerun_command` quotes every value,
  so it stays paste-safe.
- **Non-ASCII text** in `description` or the vision output (e.g. `é`, `♂`,
  `♀`, which `generator.py` already allows in names): serialized like any
  other `write_output` JSON output today (see Constraints below) — valid,
  just `\uXXXX`-escaped rather than written as literal UTF-8 characters.
- **Repo checked out without `.git`** (e.g. a tarball/wheel install) or
  `git` not on `PATH`: `git_sha` is `"unknown"`, not a crash.
- **Package not installed** (running straight from a checkout without
  `pip install -e .`, as the test suite does per `CLAUDE.md`): falls back to
  reading `pyproject.toml`'s version directly; per-repo this always
  resolves given `pyproject.toml`'s presence, so `"unknown"` here would only
  occur if reading that file's `version` key ever fails, which is not the
  normal case in this repo's checkout.

## Errors

- `run.json` failing to write (disk full, permission denied, etc.) is left
  **unguarded** — it propagates as an uncaught exception, exactly like
  today's `stats.json`/`entry.md` writes in `writer.py`, which are not
  wrapped in `try`/`except` either. This is a strict improvement over today:
  the run now fails before any sprite work starts, instead of after
  partway through it.
- Provenance lookups (`package_version`, `git_sha`) never raise — every
  failure mode for each falls back to `"unknown"` as described above, since
  neither is essential to the run succeeding and a provenance-lookup
  failure should not block writing the manifest that's the whole point of
  this feature.
- No new validation is introduced for `--description`/`--image`/`--mode`/
  `--tier`/`--stages` — those are already validated by `cli.validate_args`
  before `main.py` ever reaches `write_output`.

## Constraints & dependencies

- No new third-party dependency. `subprocess` (for `git rev-parse`) and
  `importlib.metadata` (for the package version) are both stdlib.
- `git` must be present on `PATH` for `git_sha` to resolve to a real sha;
  its absence is handled (falls back to `"unknown"`), not required.
- Serialization follows `writer.py`'s existing convention exactly:
  `json.dumps(data, indent=2)` (default `ensure_ascii=True`, i.e. no
  `ensure_ascii=False` override) written via `.write_text(..., encoding="utf-8")`
  — introducing a different JSON style for one file in the same module
  would be inconsistent for no benefit.
- `write_output`'s existing signature and behavior for `stats.json`/
  `entry.md`/directory layout/name-collision handling are unchanged; this
  is additive.
- `output/` is already git-ignored (`.gitignore`), so `run.json` files are
  never committed — no interaction with version control beyond reading the
  current sha.

## Assumptions

- **Scope**: this spec covers only the `run.json` manifest write. The
  issue's problem statement also mentions "the PNGs carry no metadata," but
  the concrete requirement is specifically "write a run.json manifest" —
  embedding provenance into PNG metadata (e.g. `sprite_prompt` as a PNG text
  chunk) is treated as a separate, not-yet-scoped follow-up, not part of
  this slice.
- **No backfill**: existing `output/` folders (including the 84 covered by
  `PROMPT-ARCHAEOLOGY.md`) do not get a `run.json` retroactively — stated
  explicitly in the issue, restated here as an explicit non-goal so it
  isn't accidentally picked up later.
- **Parameter shape into `write_output`**: the new inputs are assumed to
  arrive as one bundled mapping (a `run_info`-shaped dict), not as several
  new flat keyword arguments and not as the raw `argparse.Namespace`. This
  keeps `writer.py` free of any dependency on `argparse`/`cli.py` (it has
  none today) and avoids a name collision with `write_output`'s existing
  first positional parameter, which is already named `stages` (the list of
  per-stage dicts) — the CLI's `--stages` integer must be reachable under a
  different name.
- **`requested_stages` semantics**: recorded as `null` for
  `mode == "single"` rather than echoing the parser's ignored default —
  chosen because persisting a number that `generate_fakemon` documents as
  ignored would misleadingly suggest it did something.
- **Version resolution order**: `importlib.metadata.version(...)` first,
  `pyproject.toml` parsing as fallback, `"unknown"` as last resort — chosen
  so this works both for an installed (`pip install -e .`) run and for a
  bare checkout (how this repo's own test suite runs it, per `CLAUDE.md`),
  without ever making version capture a reason the whole run fails.
- **No "dirty" flag**: `git_sha` records `HEAD` only, with no indication of
  uncommitted local changes at generation time. A dirty-tree flag would be
  a reasonable enhancement but isn't required by the issue and is left out
  of this slice.
- **Timestamp format**: ISO 8601 with an explicit UTC offset (e.g.
  `"2026-08-04T18:22:31+00:00"`) — not a Unix epoch integer and not a naive
  local-time string — chosen for human-readability directly in the JSON
  file, consistent with this being a manifest meant to be read by a person,
  not just machine-parsed.
- **`rerun_command` targets a POSIX shell**: uniformly `shlex.quote`-style
  quoted. `README.md` documents both Windows and macOS/Linux install steps,
  but a single canonical quoting convention is the only reasonable default
  for one string field; Windows `cmd`/PowerShell users would need to adapt
  quoting themselves.
- **Model identifiers are out of scope**: `run.json` does not record the
  Mistral model names (`generator._MODEL`, `vision._VISION_MODEL`) or the
  sprite pipeline's model/LoRA identifiers. The issue's ask is the
  description/flags/vision-text/sprite_prompt plus version+sha for
  code-level provenance — model identifiers are effectively pinned by
  `git_sha` already (they're constants in the versioned source), so
  recording them separately would be redundant for this slice.
- **This is a small enough change to do as one slice**: it touches
  `writer.py` (new write step) and `main.py` (new call-site arguments) only;
  no other module needs to change. Scoped as a single coherent change
  rather than split further.
