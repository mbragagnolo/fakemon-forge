import hashlib
import json
import sys
from pathlib import Path

from mistralai.client import Mistral

_MODEL = "mistral-large-latest"

_RESOURCES = Path(__file__).parent.parent / "resources"


def _lookup_key(name: str) -> str:
    """Fold a name to its case- and space-insensitive lookup key.

    Shared by the ability and type pools: both canonicalize whatever spelling
    the model returned back onto the one form the ROM tables expect.
    """
    return "".join(name.split()).lower()


_ABILITIES_BY_INDEX: dict[str, str] = json.loads(
    (_RESOURCES / "gen3_abilities.json").read_text(encoding="utf-8")
)
_ABILITY_POOL = [
    name for idx, name in _ABILITIES_BY_INDEX.items() if idx not in ("0", "76")
]
_ABILITY_LOOKUP = {_lookup_key(name): name for name in _ABILITY_POOL}

# The 17 Gen 3 types, shared with ``export_ini`` through one resource file so the
# pool the model may pick from and the pool that can be encoded cannot drift
# apart. Fairy is deliberately absent: it has no Gen 3 byte and export folds it
# to Normal, so offering it would only invite a typing that cannot survive.
_TYPES_BY_INDEX: dict[str, str] = json.loads(
    (_RESOURCES / "gen3_types.json").read_text(encoding="utf-8")
)
_TYPE_POOL = list(_TYPES_BY_INDEX.values())
_TYPE_LOOKUP = {_lookup_key(name): name for name in _TYPE_POOL}
_DEFAULT_TYPE = "Normal"

_MAX_NAME_LEN = 10
_MAX_CATEGORY_LEN = 11

_ALLOWED_NAME_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " é♂♀"
    ".,'-…!?/()\":;"
)

#: Index of each statistic inside a `_BST_TARGETS` band.
_P10, _MEDIAN, _P90 = 0, 1, 2

# Base-stat-total bands, keyed tier -> stage count -> per-stage
# `(p10, median, p90)` triples.
#
# A stage count of 1 is a standalone species, not a juvenile: `standard` 1 is
# deliberately far above `standard` 3's opening stage (issue #48 settled the
# same reading for height/weight).
#
# Every triple is its observed Gen 3 band's 10th percentile, median and 90th --
# one rule, no exceptions, so any number here can be re-derived and checked. The
# bands live in tests/fixtures/gen3_bst_bands.json and tests/test_bst_targets.py
# enforces the correspondence; nothing at runtime reads that file.
#
# The two numbers do different jobs. The median is what the prompt asks the
# model for; the p10..p90 span is what `_bst_target` picks the enforced total
# inside. Several bands are flat because the observed data is: Gen 3 gives every
# legendary 580 and every mythical 600, and the four pseudo-legendary lines
# agree on 300 and 600 for their outer stages. A flat band makes the pick a
# no-op and the enforced total exactly the number the prompt names.
#
# `pseudo` is the one bucket that is not a whole observed population but a named
# subset of the 3-stage one, and its n is 4. Its row is the #59 hand-picked one
# re-derived from the source data: the medians came back identical, and the only
# correction was a middle-stage floor of 410, since one of the four lines sits
# 10 under the other three.
#
# `pseudo` has no 2-stage row on purpose -- every pseudo-legendary line is three
# stages, and the CLI rejects the combination.
_BST_TARGETS = {
    "standard": {
        1: ((336, 430, 500),),
        2: ((240, 305, 360), (410, 468, 515)),
        3: ((205, 295, 314), (278, 405, 420), (450, 518, 600)),
    },
    "pseudo":    {3: ((300, 300, 300), (410, 420, 420), (600, 600, 600))},
    "legendary": {1: ((580, 580, 580),)},
    "mythical":  {1: ((600, 600, 600),)},
}

_SYSTEM_PROMPT = f"""\
You are a Pokémon game designer. Generate Fakemon data as a JSON array.
Each element represents one evolutionary stage and must have exactly these fields:
  name          – portmanteau-style name (string); max 10 characters, using only
    letters, digits, spaces, é, ♂, ♀ and the punctuation . , ' - … ! ? / ( ) " : ;
  stage         – stage number as an integer (1, 2, or 3)
  types         – list of 1 or 2 distinct type strings, e.g. ["Fire"] or ["Water", "Flying"],
    chosen only from this list:
    {", ".join(_TYPE_POOL)}
    These 17 are the only types that exist. Do not invent one (there is no Sound,
    Light, Cosmic or Crystal type) and do not use Fairy — express the concept
    through the closest listed type instead, and through the sprite and the
    Pokédex entry.
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
    Every tag must describe the creature's own body. Never name anything that
    encloses or sits behind it: no "porthole", "frame", "border", "ring",
    "roundel", "emblem", "badge", "medallion", "vignette", "scene", "sky",
    "clouds", "sea", "background". The sprite model draws whatever it is given,
    so an enclosing shape becomes a frame around the picture and a setting
    becomes painted scenery — and both destroy the flat backdrop the sprite
    splitter and the background keyer need in order to work at all.
    A feature that is normally part of a vehicle or a building must be attached
    to the body to read as anatomy: write "portholes set into its flank", not
    "porthole".
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


def _bst_row(tier: str, stage_count: int) -> tuple[tuple[int, int, int], ...]:
    """Per-stage BST bands for one tier and stage count.

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
    """The user turn: description, BST hint, evolution and tier notes.

    The BST hint is deliberately **not** load-bearing. Issue #85 measured the
    model anchoring ~90 points under whatever number this string names, across
    every run, so `_normalize_base_stats` imposes the total afterwards and the
    hint only steers the raw stat line towards roughly the right scale. Removing
    it would cost nothing measurable; removing the enforcement would restore the
    bug. Anyone pruning the redundancy should prune this half.
    """
    # Single mode is one stage by definition; any requested count is ignored
    # rather than allowed to contradict the mode.
    if mode == "single":
        stage_count = 1

    row = _bst_row(tier, stage_count)

    if stage_count == 1:
        count = "one stage (stage 1 only)"
        bst_hint = f"BST target: ~{row[0][_MEDIAN]}."
        evo_text = ""
    else:
        count = _STAGE_COUNT_WORDING[stage_count]
        targets = ", ".join(
            f"stage {i} ~{band[_MEDIAN]}" for i, band in enumerate(row, start=1)
        )
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
        key = _lookup_key(entry)
        canonical = _ABILITY_LOOKUP.get(key)
        if canonical is None or key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result[:2]


def _normalize_types(raw) -> list[str]:
    """Drop entries outside the 17 Gen 3 types, canonicalize spelling, collapse
    duplicates, then cap at 2 — the same order, and for the same reason, as
    ``_normalize_abilities_gen3``.

    Never returns empty: anything that cleans away to nothing degrades to
    ``_DEFAULT_TYPE`` rather than raising, because every stage must carry a
    primary type for ``export_ini`` to encode. The prompt already constrains the
    model to the pool; this is what makes an invented type ("Sound") a repaired
    field instead of a ``KeyError`` at the very end of the run, after every
    sprite and cry in the line has already been generated.
    """
    if not isinstance(raw, list):
        return [_DEFAULT_TYPE]
    result = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        key = _lookup_key(entry)
        canonical = _TYPE_LOOKUP.get(key)
        if canonical is None or key in seen:
            continue
        seen.add(key)
        result.append(canonical)
    return result[:2] or [_DEFAULT_TYPE]


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
    process. Within ``_normalize`` this fallback is now unreachable — types are
    repaired first, and ``_normalize_types`` never returns empty — but it is kept
    for callers that reach here with a raw, unrepaired stage dict.
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


_STAT_KEYS = ("hp", "attack", "defense", "sp_atk", "sp_def", "speed")

#: Gen 3 stores each base stat in one unsigned byte, and a stat of 0 makes the
#: damage formula degenerate, so every value has to land inside this range.
_STAT_MIN = 1
_STAT_MAX = 255

#: Resolution of the deterministic position drawn from a name. Only needs to be
#: far wider than the widest band (164 points, the standalone 336-500) so that
#: no total inside a band is unreachable; a round number keeps `_bst_target`
#: easy to reason about.
_HASH_SPACE = 1_000_000


def _name_position(name: str) -> int:
    """A stable ``0 .. _HASH_SPACE - 1`` position drawn from ``name``.

    md5 is the determinism convention already used by ``export_ini._dex_number``
    and the shiny palette rotation, so a species keeps the same BST across runs
    for the same reason it keeps the same dex slot and the same shiny hue.
    """
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % _HASH_SPACE


def _bst_target(band: tuple[int, int, int], position: int) -> int:
    """The BST a species at ``position`` takes inside ``band``'s p10..p90.

    Not the flat median: standalone species genuinely spread 336-500 in Gen 3,
    and pinning every standard single form to 430 would be *less* faithful than
    the scatter it replaces.

    ``position`` is a quantile shared by every stage of a line, not a per-stage
    draw. Both p10 and p90 rise from stage to stage, so the same quantile in
    each rises too — that is what makes the totals ascend across an evolutionary
    line. Drawing independently per stage would let a stage 2 land near its p10,
    below a stage 1 near its p90, which is one of the failures this path exists
    to remove.
    """
    low, _, high = band
    if high <= low:
        return low
    return low + (position * (high - low)) // (_HASH_SPACE - 1)


def _apportion(weights: list[int], target: int) -> list[int]:
    """Split ``target`` across ``weights`` in proportion, summing to it exactly.

    Largest-remainder apportionment: floor every share, then hand the leftover
    out one point at a time to the largest remainders. Rounding each share
    independently would land on 429 or 431 about as often as on 430.

    Integer arithmetic throughout, so the leftover is exactly
    ``target - sum(shares)`` with no float error to defend against. Ties go to
    the earlier stat, keeping the result a function of the inputs alone.

    All-zero weights carry no proportion worth preserving, so they split evenly.
    """
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        weights = [1] * len(weights)
        total = len(weights)

    shares = [w * target // total for w in weights]
    remainders = [w * target % total for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
    for i in order[: target - sum(shares)]:
        shares[i] += 1
    return shares


def _apportion_clamped(weights: list[int], target: int) -> list[int]:
    """``_apportion`` with every share held inside [_STAT_MIN, _STAT_MAX].

    Mostly a safety net: no band target can push a stat near 255 unless the
    model returns something degenerate, like five 1s and a 250.

    Clamping breaks the total, so whatever the clamp gained or cost is moved
    back across the stats that still have room, in proportion to how much room
    each has. Redistributing by *headroom* rather than by weight is what makes
    this converge: the stats that absorb the leftover are exactly the ones that
    can, and a stat sitting on a bound is skipped rather than pushed through it.

    One pass always closes the gap. ``target`` is bounded to
    [_STAT_MIN * n, _STAT_MAX * n] by the caller, which makes the available room
    strictly larger than the shortfall, so `_apportion` hands out the whole of
    it. The loop is written as a loop anyway so that a caller which skipped that
    bound degrades to a best effort instead of returning a wrong total silently.
    """
    values = [min(_STAT_MAX, max(_STAT_MIN, s)) for s in _apportion(weights, target)]
    while True:
        deficit = target - sum(values)
        if deficit == 0:
            return values
        sign = 1 if deficit > 0 else -1
        room = [_STAT_MAX - v if sign > 0 else v - _STAT_MIN for v in values]
        movable = [i for i in range(len(values)) if room[i] > 0]
        if not movable:
            return values
        moved = _apportion(
            [room[i] for i in movable],
            min(abs(deficit), sum(room)),
        )
        for i, amount in zip(movable, moved):
            values[i] += sign * amount


def _stat_weights(raw) -> list[int] | None:
    """The six base stats as proportional weights, or None if unusable.

    Unusable means the value carries no design intent to preserve: not a dict, a
    missing stat, a bool (which ``isinstance(x, int)`` would otherwise wave
    through), a non-number, or a negative. One bad stat spoils the whole set
    rather than being read as zero — a zero weight becomes a base stat of 1,
    which is a worse guess than an even split and hides the malformed field.
    """
    if not isinstance(raw, dict):
        return None
    weights = []
    for key in _STAT_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value < 0:
            return None
        weights.append(int(value))
    return weights


def _normalize_base_stats(raw, target: int) -> dict[str, int]:
    """Rescale the model's six stats so they sum to exactly ``target``.

    The model is reliable about the *shape* of a stat line — a wall is bulky and
    slow, a sweeper is fast and frail — and unreliable about its magnitude. It
    anchors low and ignores the total the prompt asks for: eleven ``--tier
    standard`` single forms all prompted with ~430 averaged 337 (issue #85).
    That is systematic bias rather than variance, so re-prompting only re-samples
    it. The six returned stats are read as weights instead, and the total is
    imposed here.

    Ratios survive up to rounding, so a glass cannon stays a glass cannon.
    Unusable input falls back to an even split of the target.
    """
    weights = _stat_weights(raw)
    if weights is None:
        weights = [1] * len(_STAT_KEYS)
    # Bound the target to what six clamped bytes can actually add up to, so the
    # redistribution always has a solution. Every value `_BST_TARGETS` can
    # produce is already far inside this.
    target = max(
        _STAT_MIN * len(_STAT_KEYS),
        min(_STAT_MAX * len(_STAT_KEYS), int(target)),
    )
    return dict(zip(_STAT_KEYS, _apportion_clamped(weights, target)))


def _stage_band(
    stage: dict, mode: str, tier: str, stage_count: int, index: int
) -> tuple[int, int, int]:
    """The BST band one stage draws its target from.

    Single mode is one standalone form whatever count was requested — the same
    reading ``_size_defaults`` applies. In line mode the stage's own declared
    number picks the row; an off-spec number (missing, out of range, a JSON
    string) falls back to the stage's position in the returned list, and a list
    longer than the row falls back to the row's last band.
    """
    if mode != "line":
        return _bst_row(tier, 1)[0]
    row = _bst_row(tier, stage_count)
    try:
        position = int(stage.get("stage")) - 1
    except (TypeError, ValueError):
        position = index
    if not 0 <= position < len(row):
        position = index
    return row[min(position, len(row) - 1)]


def _normalize(
    stages: list[dict], mode: str, tier: str, stage_count: int = 3
) -> list[dict]:
    """Post-parse cleanup pass: enforces the name contract, rescales base_stats
    onto their band target, defaults/clamps height_dm and weight_hg, and filters
    both types and abilities_gen3 to their Gen 3 pools.

    Repair is idempotent: a name already inside the Gen 3 contract comes out
    of ``_repair_name`` unchanged, so valid names pass through untouched.

    ``stages`` is the parsed list of stage dicts; ``stage_count`` is how many
    were *requested*. They are deliberately different things — the model may
    return a different number than was asked for, which is accepted.
    """
    # One position for the whole line, drawn from stage 1's name — the line's
    # name, which is what ``main.py`` already treats as the line identity when
    # seeding cries. Every stage is then placed at that same quantile of its own
    # band, which is what makes the totals ascend. ``_repair_name`` is
    # idempotent, so seeding from it here agrees with what the loop assigns.
    position = _name_position(_repair_name(stages[0]["name"])) if stages else 0

    for index, stage in enumerate(stages):
        stage["name"] = _repair_name(stage["name"])
        if "pokedex_entry" in stage:
            stage["pokedex_entry"] = _repair_entry(stage["pokedex_entry"])
        stage["abilities_gen3"] = _normalize_abilities_gen3(stage.get("abilities_gen3", []))
        # Before category: it falls back to the primary type word, which must be
        # a real type rather than whatever the model invented.
        stage["types"] = _normalize_types(stage.get("types"))
        stage["category"] = _normalize_category(stage.get("category"), stage.get("types"))

        band = _stage_band(stage, mode, tier, stage_count, index)
        stage["base_stats"] = _normalize_base_stats(
            stage.get("base_stats"), _bst_target(band, position)
        )

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
