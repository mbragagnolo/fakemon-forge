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

# Base-stat-total targets, keyed tier -> stage count -> per-stage targets.
# A stage count of 1 is a standalone species, not a juvenile: `standard` 1 is
# deliberately far above `standard` 3's opening stage (issue #48 settled the
# same reading for height/weight).
#
# Every value is the median of its observed Gen 3 band -- one rule, no
# exceptions, so any number here can be re-derived and checked. The bands live
# in tests/fixtures/gen3_bst_bands.json and tests/test_bst_targets.py enforces
# the correspondence; nothing at runtime reads that file.
#
# `pseudo` has no 2-stage row on purpose -- every pseudo-legendary line is three
# stages, and the CLI rejects the combination.
_BST_TARGETS = {
    "standard":  {1: (430,), 2: (305, 468), 3: (295, 405, 518)},
    "pseudo":    {3: (300, 420, 600)},
    "legendary": {1: (580,)},
    "mythical":  {1: (600,)},
}

_SYSTEM_PROMPT = f"""\
You are a Pokémon game designer. Generate Fakemon data as a JSON array.
Each element represents one evolutionary stage and must have exactly these fields:
  name          – portmanteau-style name (string); max 10 characters, using only
    letters, digits, spaces, é, ♂, ♀ and the punctuation . , ' - … ! ? / ( ) " : ;
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
  pokedex_entry – 2 sentence flavour text (string); at most 130 characters, and
    use only straight quotes and hyphens (' " -), never curly quotes or dashes.
    The display window fits 4 lines of 40 characters — anything past that is cut
    mid-sentence, so keep it comfortably short.
  sprite_prompt – comma-separated visual TAGS for pixel-art sprite generation (string).
    Tags, never sentences. Write "small round bird, slate blue feathers, yellow
    lightning-bolt crest, stubby wings, hooked beak" — not "a small round bird
    whose feathers are slate blue, and it has a crest that arcs with lightning."
    No verbs, no clauses, no full stops.
    Hard limit: at most 18 tags and 35 words. The sprite model's text encoder
    truncates at 77 tokens and silently discards everything past it — and the
    styling tags the pipeline appends AFTER this string are what get discarded
    first. Going long does not add detail, it deletes the background and style
    instructions.
    Order: overall body shape first, then main colours, then distinctive features.
    Never use framing or absolute-size words here: no "large", "huge", "giant",
    "massive", "towering", "imposing", "close-up", "dramatic", "epic". The sprite
    model reads those as instructions about how much of the picture to fill, not
    as facts about the creature — "large" makes it render a cropped close-up that
    runs off the edges instead of a whole creature. A later evolution is conveyed
    by having MORE and BIGGER FEATURES than the earlier one — extra limbs, longer
    horns, heavier armour plating, a more complex silhouette — never by calling
    the creature large. The size difference between stages belongs in height_dm
    and weight_hg, which is where it is actually recorded.
    It must also show the creature's types through what it looks like — flames or
    embers for Fire, fins or droplets for Water, wings or feathers for Flying, and
    so on for every type you assigned. The sprite model is given this string and
    nothing else: it never sees the types field, so a type you do not describe here
    cannot reach the sprite.
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

# A 2-stage line goes juvenile -> adult with no middle form. Describing an
# adolescent stage here would ask for a form that is never generated.
_EVO_PROGRESSION_2 = """\

Evolutionary progression — each stage must look and feel visually distinct:
  Stage 1: juvenile/child form — small and simple, cute or curious expression, \
limited limbs or features, undeveloped power.
  Stage 2: adult/final form — fully developed, imposing presence, complex design \
with a different silhouette from stage 1, design complexity at its peak.\
"""

_EVO_PROGRESSION_BY_COUNT = {2: _EVO_PROGRESSION_2, 3: _EVO_PROGRESSION}

_STAGE_COUNT_WORDING = {
    2: "two evolutionary stages (stages 1 and 2)",
    3: "three evolutionary stages (stages 1, 2, and 3)",
}

_TIER_NOTES = {
    "pseudo":    "\nThis is a pseudo-legendary line: the final form should rival legendary "
                 "Pokémon in visual impact and raw power.",
    "legendary": "\nThis is a legendary Pokémon: unique, awe-inspiring, and lore-significant. "
                 "It should feel like a force of nature.",
    "mythical":  "\nThis is a mythical Pokémon: mysterious, rarely seen, tied to ancient legend.",
}


def _bst_row(tier: str, stage_count: int) -> tuple[int, ...]:
    """Per-stage BST targets for one tier and stage count.

    A tier without a row for the requested count falls back to the one row it
    does have, trimmed to the count asked for. That keeps
    ``--tier pseudo --mode single`` prompting the value it prompted before #59:
    pseudo has only a 3-stage row, and single mode reads the first entry. The
    combination is already documented as line-only and the CLI rejects it;
    until then it must not raise.
    """
    rows = _BST_TARGETS[tier]
    row = rows.get(stage_count)
    if row is None:
        row = next(iter(rows.values()))[:stage_count]
    return row


def _user_prompt(description: str, mode: str, tier: str, stage_count: int = 3) -> str:
    # Single mode is one stage by definition; any requested count is ignored
    # rather than allowed to contradict the mode.
    if mode == "single":
        stage_count = 1

    row = _bst_row(tier, stage_count)

    if stage_count == 1:
        count = "one stage (stage 1 only)"
        bst_hint = f"BST target: ~{row[0]}."
        evo_text = ""
    else:
        count = _STAGE_COUNT_WORDING[stage_count]
        targets = ", ".join(f"stage {i} ~{v}" for i, v in enumerate(row, start=1))
        bst_hint = f"BST targets: {targets}."
        evo_text = _EVO_PROGRESSION_BY_COUNT[stage_count]

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


#: Typographic characters a language model reaches for that fall outside the
#: Gen 3 text contract, mapped to the equivalents that are inside it. Folding
#: beats dropping: "It's" reads correctly, "Its" does not.
_PUNCTUATION_FOLD = {
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "",
}

#: Gen 3 renders flavour text in a fixed window: this many lines of this many
#: characters, greedily word-wrapped. Text past the window is not shown at all,
#: so an entry that overruns is cut mid-sentence rather than scrolled.
_ENTRY_LINE_WIDTH = 40
_ENTRY_MAX_LINES = 4


def _wrap_entry(entry: str) -> list[str]:
    """Greedy word-wrap ``entry`` into ``_ENTRY_LINE_WIDTH``-char lines."""
    lines: list[str] = []
    current = ""
    for word in entry.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= _ENTRY_LINE_WIDTH:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _entry_fits_budget(entry: str) -> bool:
    """Whether ``entry`` fits the flavour-text window without being cut."""
    return len(_wrap_entry(entry)) <= _ENTRY_MAX_LINES


def _repair_entry(entry) -> str:
    """Bring flavour text inside the Gen 3 text contract.

    Folds typographic punctuation to its in-contract equivalent, drops
    characters the contract has no glyph for, and trims to the display window
    on a word boundary so the result reads as a finished sentence rather than
    stopping mid-word.
    """
    folded = "".join(_PUNCTUATION_FOLD.get(ch, ch) for ch in str(entry))
    cleaned = "".join(ch for ch in folded if ch in _ALLOWED_NAME_CHARS)
    cleaned = " ".join(cleaned.split())
    if _entry_fits_budget(cleaned):
        return cleaned

    # Prefer dropping whole sentences: an entry that stops at a full stop still
    # reads as written, where one cut at a word boundary trails off on a
    # fragment ("...disrupting nearby electronics. Often.").
    sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
    while len(sentences) > 1:
        sentences.pop()
        candidate = ". ".join(sentences) + "."
        if _entry_fits_budget(candidate):
            return candidate

    # One sentence, still too long: fall back to dropping whole words.
    words = cleaned.split()
    while words and not _entry_fits_budget(" ".join(words)):
        words.pop()
    trimmed = " ".join(words).rstrip(" ,;:-")
    if trimmed and not trimmed.endswith(".") and _entry_fits_budget(trimmed + "."):
        trimmed += "."
    return trimmed


# Per-stage size fallbacks, keyed stage count -> stage number. A 2-stage line's
# stage 2 is a *final* form, so it takes the same row a 3-stage final does —
# reusing the 3-stage middle row would under-size it.
_SIZE_DEFAULTS_BY_LINE = {
    2: {1: (5, 30), 2: (17, 600)},
    3: {1: (5, 30), 2: (10, 150), 3: (17, 600)},
}

_SIZE_DEFAULTS_BY_TIER = {
    "standard": (10, 150),
    "pseudo": (17, 600),
    "legendary": (17, 600),
    "mythical": (17, 600),
}


def _normalize_abilities_gen3(raw) -> list[str]:
    """Drop entries outside the Gen 3 pool, canonicalize spelling, collapse
    duplicates (by normalized form), then cap at 2 — in that order, so a
    dedup-worthy duplicate can't crowd out a later distinct valid entry.

    A non-list value, or a non-string entry inside the list, is treated as
    absent rather than raised on — the same reading ``export_ini`` applies to
    the persisted field, so both halves agree on what malformed looks like.
    """
    if not isinstance(raw, list):
        return []
    result = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        key = _normalize_ability_name(entry)
        canonical = _ABILITY_LOOKUP.get(key)
        if canonical is None or key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result[:2]


_POKEMON_SUFFIX = " POKEMON"


def _gen3_upper(text: str) -> str:
    """Uppercase within the Gen 3 text set.

    The set carries lowercase é but no uppercase É, so a plain ``.upper()``
    turns a legal character into an unencodable one. É folds back to é.
    """
    return text.upper().replace("É", "é")


def _strip_pokemon_suffix(text: str) -> str:
    """Drop a trailing " POKEMON", however the model spelled it.

    The prompt writes "Pokémon" with the accent throughout, so that is the
    likelier echo. Both spellings are 8 characters, so the fold is only used
    to locate the suffix — the slice comes off the original text.
    """
    if text.upper().replace("É", "E").endswith(_POKEMON_SUFFIX):
        return text[: -len(_POKEMON_SUFFIX)]
    return text


def _type_word(types) -> str:
    """Primary type word for the category fallback.

    An absent or empty ``types`` degrades to NORMAL rather than raising:
    category is cosmetic, and _normalize's contract is that it repairs in
    process. A genuinely typeless stage still fails later in export_ini,
    where the type bytes actually matter.
    """
    if isinstance(types, list) and types and isinstance(types[0], str):
        return types[0].upper()
    return "NORMAL"


def _normalize_category(raw, types) -> str:
    """Uppercase/truncate/strip-trailing-"POKEMON" for the Pokédex category
    noun; falls back to the primary type word when raw is missing, empty,
    non-str, or cleans away to nothing. Truncation is immediate — unlike
    ``name``, an over-long category never triggers a retry.

    Whitespace is trimmed on the way in (rstrip only, so the leading space of
    a bare " POKEMON" still reads as the suffix token) and again after
    truncation, which would otherwise emit a dangling "GIANT SEED ".

    The suffix comes off before charset filtering, not after: É is outside the
    allowed set, so filtering first turned "SEED POKÉMON" into "SEED POKMON"
    and the suffix stopped matching anything.
    """
    if not isinstance(raw, str):
        return _type_word(types)
    cleaned = _strip_pokemon_suffix(raw.rstrip())
    cleaned = "".join(ch for ch in _gen3_upper(cleaned) if ch in _ALLOWED_NAME_CHARS)
    result = cleaned.strip()[:_MAX_CATEGORY_LEN].strip()
    return result or _type_word(types)


def _size_defaults(
    stage: dict, mode: str, tier: str, stage_count: int = 3
) -> tuple[int, int]:
    """Stage/tier-scaled (height_dm, weight_hg) fallbacks.

    An off-spec stage number — missing, out of range, or a JSON string — falls
    through to the tier table rather than raising KeyError, matching how
    ``main.py`` already reads the same field for its sprite size fraction. An
    unrecognised stage count degrades to the 3-stage rows the same way.
    """
    if mode == "line":
        rows = _SIZE_DEFAULTS_BY_LINE.get(stage_count, _SIZE_DEFAULTS_BY_LINE[3])
        try:
            return rows[int(stage.get("stage"))]
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


def _normalize(
    stages: list[dict], mode: str, tier: str, stage_count: int = 3
) -> list[dict]:
    """Post-parse cleanup pass: enforces the name contract, defaults/clamps
    height_dm and weight_hg, and filters abilities_gen3 to the Gen 3 pool.

    Repair is idempotent: a name already inside the Gen 3 contract comes out
    of ``_repair_name`` unchanged, so valid names pass through untouched.

    ``stages`` is the parsed list of stage dicts; ``stage_count`` is how many
    were *requested*. They are deliberately different things — the model may
    return a different number than was asked for, which is accepted.
    """
    for stage in stages:
        stage["name"] = _repair_name(stage["name"])
        if "pokedex_entry" in stage:
            stage["pokedex_entry"] = _repair_entry(stage["pokedex_entry"])
        stage["abilities_gen3"] = _normalize_abilities_gen3(stage.get("abilities_gen3", []))
        stage["category"] = _normalize_category(stage.get("category"), stage.get("types"))

        height_default, weight_default = _size_defaults(stage, mode, tier, stage_count)
        stage["height_dm"] = _clamp_dimension(stage.get("height_dm"), 999, height_default)
        stage["weight_hg"] = _clamp_dimension(stage.get("weight_hg"), 9999, weight_default)
    return stages


def generate_fakemon(
    description: str,
    mode: str,
    tier: str = "standard",
    *,
    stages: int = 3,
    client=None,
    api_key: str = None,
) -> list[dict]:
    """Generate one Fakemon's stages.

    ``stages`` is how many evolutionary stages to ask for; it applies to
    ``mode="line"`` and is ignored for ``mode="single"``. The default of 3
    is what keeps an existing ``--mode line`` invocation unchanged.
    """
    if client is None:
        client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(description, mode, tier, stages)},
    ]

    raw = None
    for attempt in range(2):
        try:
            response = client.chat.complete(model=_MODEL, messages=messages)
            raw = response.choices[0].message.content
            # Named `parsed`, not `stages`: `stages` is the requested *count*
            # parameter, and reusing the name would shadow it before
            # `_normalize` below can be told how many stages were asked for.
            parsed = json.loads(_strip_fences(raw))
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

        too_long, illegal = _name_violations(parsed)
        if (too_long or illegal) and attempt == 0:
            # The offending array has to be in the conversation for "return the
            # full array again" to mean anything — without it the model rebuilds
            # the line from scratch and the already-valid sibling names change.
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": _corrective_message(too_long, illegal)})
            continue

        return _normalize(parsed, mode, tier, stages)
