import json
import sys

from mistralai.client import Mistral

_MODEL = "mistral-large-latest"

_MAX_NAME_LEN = 10

_ALLOWED_NAME_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " é♂♀"
    ".,'-…!?/()\":;"
)

_BST_TARGETS = {
    "standard": {"stage1": 300, "stage2": 420, "stage3": 520},
    "pseudo":   {"stage1": 300, "stage2": 420, "stage3": 600},
    "legendary":{"stage1": 580},
    "mythical": {"stage1": 600},
}

_SYSTEM_PROMPT = """\
You are a Pokémon game designer. Generate Fakemon data as a JSON array.
Each element represents one evolutionary stage and must have exactly these fields:
  name          – portmanteau-style name (string); max 10 characters, using only
    letters, digits, spaces, é, ♂, ♀ and the punctuation . , ' - … ! ? / ( ) " : ;
  stage         – stage number as an integer (1, 2, or 3)
  types         – list of 1 or 2 type strings, e.g. ["Fire"] or ["Water", "Flying"]
  ability       – one ability name (string)
  base_stats    – object with integer values for: hp, attack, defense, sp_atk, sp_def, speed
  pokedex_entry – 2 sentence flavour text (string)
  sprite_prompt – visual description for pixel-art sprite generation; max 75 words, lead with the creature's most distinctive shape and colour features (string)
  levitates     – boolean; true only if the creature levitates, is bodiless/gaseous/amorphous, or otherwise never touches the ground (e.g. floating orbs, ghosts, cloud/gas creatures); otherwise false
  height_dm – height in decimetres (integer).
  weight_hg – weight in hectograms (integer).
    For scale: a small rodent is ~3 dm / 35 hg, a mid-size quadruped
    ~10 dm / 300 hg, a large final-stage dragon ~20 dm / 2100 hg.
    Values must grow across an evolutionary line.

All stage names and sprite prompts must share a clear thematic throughline.
Return ONLY the JSON array. No markdown fences, no explanation, no extra keys.\
"""

_EVO_PROGRESSION = """\

Evolutionary progression — each stage must look and feel visually distinct:
  Stage 1: juvenile/child form — small and simple, cute or curious expression, \
limited limbs or features, undeveloped power.
  Stage 2: adolescent/teenage form — noticeably larger, silhouette more defined, \
signature features emerging, power becoming apparent.
  Stage 3: adult/final form — fully developed, imposing presence, complex design \
with a different silhouette from stage 1, design complexity at its peak.\
"""

_TIER_NOTES = {
    "pseudo":    "\nThis is a pseudo-legendary line: the final form should rival legendary "
                 "Pokémon in visual impact and raw power.",
    "legendary": "\nThis is a legendary Pokémon: unique, awe-inspiring, and lore-significant. "
                 "It should feel like a force of nature.",
    "mythical":  "\nThis is a mythical Pokémon: mysterious, rarely seen, tied to ancient legend.",
}


def _user_prompt(description: str, mode: str, tier: str) -> str:
    targets = _BST_TARGETS[tier]

    if mode == "single":
        count = "one stage (stage 1 only)"
        bst_hint = f"BST target: ~{targets['stage1']}."
        evo_text = ""
    else:
        count = "three evolutionary stages (stages 1, 2, and 3)"
        bst_hint = (
            f"BST targets: stage 1 ~{targets['stage1']}, "
            f"stage 2 ~{targets['stage2']}, "
            f"stage 3 ~{targets['stage3']}."
        )
        evo_text = _EVO_PROGRESSION

    tier_note = _TIER_NOTES.get(tier, "")

    return (
        f"Generate {count} for a Fakemon based on this description:\n\n"
        f"{description}\n\n"
        f"{bst_hint}"
        f"{evo_text}"
        f"{tier_note}"
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _name_violations(stages: list[dict]) -> tuple[list[str], list[str]]:
    """Return (too_long_names, illegal_char_names) across all stages."""
    too_long = []
    illegal = []
    for stage in stages:
        # A present-but-non-string name is repaired via str(), not raised on —
        # so it has to be measured the same way here or the two disagree.
        name = str(stage["name"])
        if len(name) > _MAX_NAME_LEN:
            too_long.append(name)
        if any(ch not in _ALLOWED_NAME_CHARS for ch in name):
            illegal.append(name)
    return too_long, illegal


def _corrective_message(too_long: list[str], illegal: list[str]) -> str:
    parts = []
    if too_long:
        parts.append(
            "These names exceed 10 characters: " + ", ".join(too_long) +
            ". Return the full array again with shorter names."
        )
    if illegal:
        parts.append(
            "These names contain characters that can't be used: " + ", ".join(illegal) +
            ". Return the full array again using only letters, numbers, spaces, "
            "é, ♂, ♀ and the punctuation . , ' - … ! ? / ( ) \" : ;"
        )
    return " ".join(parts)


def _repair_name(name) -> str:
    cleaned = "".join(ch for ch in str(name) if ch in _ALLOWED_NAME_CHARS)
    return cleaned[:_MAX_NAME_LEN]


_SIZE_DEFAULTS_BY_LINE_STAGE = {
    1: (5, 30),
    2: (10, 150),
    3: (17, 600),
}

_SIZE_DEFAULTS_BY_TIER = {
    "standard": (10, 150),
    "pseudo": (17, 600),
    "legendary": (17, 600),
    "mythical": (17, 600),
}


def _size_defaults(stage: dict, mode: str, tier: str) -> tuple[int, int]:
    """Stage/tier-scaled (height_dm, weight_hg) fallbacks.

    An off-spec stage number — missing, out of range, or a JSON string — falls
    through to the tier table rather than raising KeyError, matching how
    ``main.py`` already reads the same field for its sprite size fraction.
    """
    if mode == "line":
        try:
            return _SIZE_DEFAULTS_BY_LINE_STAGE[int(stage.get("stage"))]
        except (TypeError, ValueError, KeyError):
            pass
    return _SIZE_DEFAULTS_BY_TIER.get(tier, _SIZE_DEFAULTS_BY_TIER["standard"])


def _clamp_dimension(value, upper: int, fallback: int) -> int:
    """Coerce to int and clamp to [1, upper]; unusable values take the default.

    Both fields are 2-byte unsigned downstream, so a float would be as
    unencodable as a string — the int() coercion is part of the contract, not
    just defensive typing.
    """
    if isinstance(value, bool):
        return fallback
    try:
        value = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(1, min(upper, value))


def _normalize(stages: list[dict], mode: str, tier: str) -> list[dict]:
    """Post-parse cleanup pass: enforces the name contract, and defaults/clamps
    height_dm and weight_hg.

    Repair is idempotent: a name already inside the Gen 3 contract comes out
    of ``_repair_name`` unchanged, so valid names pass through untouched.
    """
    for stage in stages:
        stage["name"] = _repair_name(stage["name"])

        height_default, weight_default = _size_defaults(stage, mode, tier)
        stage["height_dm"] = _clamp_dimension(stage.get("height_dm"), 999, height_default)
        stage["weight_hg"] = _clamp_dimension(stage.get("weight_hg"), 9999, weight_default)
    return stages


def generate_fakemon(
    description: str,
    mode: str,
    tier: str = "standard",
    *,
    client=None,
    api_key: str = None,
) -> list[dict]:
    if client is None:
        client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(description, mode, tier)},
    ]

    raw = None
    for attempt in range(2):
        try:
            response = client.chat.complete(model=_MODEL, messages=messages)
            raw = response.choices[0].message.content
            stages = json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            if attempt == 1:
                print(
                    f"Error: LLM returned malformed JSON after 2 attempts.\n"
                    f"Raw response:\n{raw}",
                    file=sys.stderr,
                )
                sys.exit(1)
            continue
        except Exception as exc:
            print(
                f"Error: Mistral API call failed ({exc}). "
                "Check that MISTRAL_API_KEY is set and valid.",
                file=sys.stderr,
            )
            sys.exit(1)

        too_long, illegal = _name_violations(stages)
        if (too_long or illegal) and attempt == 0:
            # The offending array has to be in the conversation for "return the
            # full array again" to mean anything — without it the model rebuilds
            # the line from scratch and the already-valid sibling names change.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": _corrective_message(too_long, illegal)})
            continue

        return _normalize(stages, mode, tier)
