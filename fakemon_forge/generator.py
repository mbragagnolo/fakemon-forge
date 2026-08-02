import json
import sys
from pathlib import Path

from mistralai.client import Mistral

_MODEL = "mistral-large-latest"

_RESOURCES = Path(__file__).parent.parent / "resources"


def _normalize_ability_name(name: str) -> str:
    return "".join(name.split()).lower()


_ABILITIES_BY_INDEX: dict[str, str] = json.loads(
    (_RESOURCES / "gen3_abilities.json").read_text(encoding="utf-8")
)
_ABILITY_POOL = [
    name for idx, name in _ABILITIES_BY_INDEX.items() if idx not in ("0", "76")
]
_ABILITY_LOOKUP = {_normalize_ability_name(name): name for name in _ABILITY_POOL}

_MAX_NAME_LEN = 10
_MAX_CATEGORY_LEN = 11

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

_SYSTEM_PROMPT = f"""\
You are a Pokémon game designer. Generate Fakemon data as a JSON array.
Each element represents one evolutionary stage and must have exactly these fields:
  name          – portmanteau-style name (string)
  stage         – stage number as an integer (1, 2, or 3)
  types         – list of 1 or 2 type strings, e.g. ["Fire"] or ["Water", "Flying"]
  category      – Pokédex category noun in caps, max 11 characters, e.g. "SEED",
    "MOUSE", "TINY TURTLE". Describes what the creature *is*, not its type —
    never "FIRE" or "WATER". No trailing "POKEMON".
  ability       – one ability name (string)
  abilities_gen3 – list of 1 or 2 distinct real Gen 3 ability names, chosen only from this list:
    {", ".join(_ABILITY_POOL)}
    Prefer two abilities (authentic Gen 3 species are roughly half dual-ability, and variety
    is preferred); one is acceptable for a single signature ability.
    In an evolutionary line, all stages should share the same abilities_gen3; the final stage
    may add one more.
    The free-text ability above should express the same concept as the chosen abilities_gen3
    entries.
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
        name = stage["name"]
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
            "and standard punctuation."
        )
    return " ".join(parts)


def _repair_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch in _ALLOWED_NAME_CHARS)
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


def _normalize_abilities_gen3(raw: list) -> list[str]:
    """Drop entries outside the Gen 3 pool, canonicalize spelling, collapse
    duplicates (by normalized form), then cap at 2 — in that order, so a
    dedup-worthy duplicate can't crowd out a later distinct valid entry."""
    result = []
    seen = set()
    for entry in raw:
        key = _normalize_ability_name(entry)
        canonical = _ABILITY_LOOKUP.get(key)
        if canonical is None or key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result[:2]


def _normalize_category(raw, types: list[str]) -> str:
    """Uppercase/truncate/strip-trailing-"POKEMON" for the Pokédex category
    noun; falls back to the primary type word when raw is missing, empty,
    non-str, or cleans away to nothing. Truncation is immediate — unlike
    ``name``, an over-long category never triggers a retry.

    Whitespace is trimmed on the way in (rstrip only, so the leading space of
    a bare " POKEMON" still reads as the suffix token) and again after
    truncation, which would otherwise emit a dangling "GIANT SEED ".
    """
    if not isinstance(raw, str):
        return types[0].upper()
    cleaned = "".join(ch for ch in raw if ch in _ALLOWED_NAME_CHARS).rstrip()
    if cleaned.upper().endswith(" POKEMON"):
        cleaned = cleaned[: -len(" POKEMON")]
    result = cleaned.upper().strip()[:_MAX_CATEGORY_LEN].strip()
    return result or types[0].upper()


def _normalize(stages: list[dict], mode: str, tier: str) -> list[dict]:
    """Post-parse cleanup pass: enforces the name contract, defaults/clamps
    height_dm and weight_hg, and filters abilities_gen3 to the Gen 3 pool.

    Repair is idempotent: a name already inside the Gen 3 contract comes out
    of ``_repair_name`` unchanged, so valid names pass through untouched.
    """
    for stage in stages:
        stage["name"] = _repair_name(stage["name"])
        stage["abilities_gen3"] = _normalize_abilities_gen3(stage.get("abilities_gen3", []))
        stage["category"] = _normalize_category(stage.get("category"), stage["types"])

        if "height_dm" not in stage or "weight_hg" not in stage:
            if mode == "line":
                height_default, weight_default = _SIZE_DEFAULTS_BY_LINE_STAGE[stage["stage"]]
            else:
                height_default, weight_default = _SIZE_DEFAULTS_BY_TIER[tier]
            if "height_dm" not in stage:
                stage["height_dm"] = height_default
            if "weight_hg" not in stage:
                stage["weight_hg"] = weight_default

        stage["height_dm"] = max(1, min(999, stage["height_dm"]))
        stage["weight_hg"] = max(1, min(9999, stage["weight_hg"]))
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
            messages.append({"role": "user", "content": _corrective_message(too_long, illegal)})
            continue

        return _normalize(stages, mode, tier)
