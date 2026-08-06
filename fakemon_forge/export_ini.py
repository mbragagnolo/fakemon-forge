"""Export a Fakemon stage directory to a Gen 3-style .ini data file."""

import hashlib
import json
import random
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

_RESOURCES = Path(__file__).parent.parent / "resources"

# PokedexType's field budget. One char more than a species name — a different
# field with a different limit, so the two constants stay separate.
_MAX_CATEGORY_LEN = 11

# ── lookup tables ──────────────────────────────────────────────────────────────

# Post-Gen-3 types the generator will not emit (it is constrained to the pool in
# ``gen3_types.json``), kept so a hand-edited or pre-existing stats.json still
# encodes rather than raising. Fairy has no Gen 3 byte, so it folds to Normal.
_TYPE_ALIASES = {"Fairy": "Normal"}

_TYPE_BODY_COLOR = {
    0x00: 8,  # Normal → White
    0x01: 4,  # Fighting → Black
    0x02: 1,  # Flying → Blue
    0x03: 6,  # Poison → Purple
    0x04: 5,  # Ground → Brown
    0x05: 5,  # Rock → Brown
    0x06: 3,  # Bug → Green
    0x07: 6,  # Ghost → Purple
    0x08: 7,  # Steel → Gray
    0x0A: 0,  # Fire → Red
    0x0B: 1,  # Water → Blue
    0x0C: 3,  # Grass → Green
    0x0D: 2,  # Electric → Yellow
    0x0E: 9,  # Psychic → Pink
    0x0F: 8,  # Ice → White
    0x10: 1,  # Dragon → Blue
    0x11: 4,  # Dark → Black
}

# Thematic level-up move pools per type (level, move_id).
#
# Curation rules, enforced by tests/test_move_tables.py:
#   * every entry is the pool's own type — no Muddy Water in Ground;
#   * damaging moves come in non-decreasing power order, so a mon's newest
#     attack is never weaker than one it already knows;
#   * every pool opens with a damaging move at level 1 — a low-level
#     encounter draws only the entries at or below its level, and a status
#     opener would leave it unable to attack;
#   * body-part moves (punches, kicks, fangs, tail, wings, beak, horn,
#     claws) live in `_TRAIT_MOVES`, not here, so a wingless Steel type no
#     longer learns Steel Wing. Two documented exceptions stay: Metal Claw
#     is the only non-anatomical damaging Steel move in Gen 3, and Dragon
#     Claw stays because dragons have claws by definition.
_MOVE_POOL: dict[str, list[tuple[int, int]]] = {
    "Normal":   [(1,98),(13,29),(19,104),(27,34),(35,216),(43,38),(50,63)],
    "Water":    [(1,145),(7,55),(13,352),(19,61),(26,240),(33,127),(41,57),(49,56)],
    "Fire":     [(1,52),(12,172),(19,241),(26,53),(34,257),(43,126),(50,315)],
    "Grass":    [(1,71),(7,22),(13,75),(20,73),(26,202),(33,80),(41,76),(49,338)],
    "Electric": [(1,84),(9,86),(16,351),(24,209),(31,85),(40,192),(48,87)],
    "Ice":      [(1,181),(10,196),(18,62),(25,258),(32,58),(41,59),(50,329)],
    "Fighting": [(1,67),(10,68),(16,233),(24,280),(33,238),(45,276)],
    "Poison":   [(1,40),(8,51),(15,77),(22,124),(30,92),(38,188)],
    "Ground":   [(1,189),(9,28),(16,341),(23,91),(33,89),(44,90)],
    "Flying":   [(1,16),(11,314),(19,332),(28,19),(46,143)],
    "Bug":      [(1,42),(8,81),(15,141),(24,318),(36,324)],
    "Rock":     [(1,205),(9,88),(17,317),(25,246),(36,157)],
    "Ghost":    [(1,310),(9,101),(16,109),(26,247),(35,171),(44,194)],
    "Dragon":   [(1,239),(10,82),(18,225),(27,337),(36,349),(46,200)],
    "Dark":     [(1,228),(9,168),(17,185),(26,269)],
    "Steel":    [(1,232),(12,319),(29,334),(46,353)],
    "Psychic":  [(1,93),(9,95),(17,60),(25,347),(33,248),(45,94)],
}

# Body-part move buckets, keyed by the trait vocabulary in
# ``resources/traits.json`` (which the generator constrains the model to).
# A mon only draws from the buckets whose traits it actually has, which is
# both what keeps Steel Wing off wingless mons and where cross-type coverage
# comes from — elemental punches, kicks and fangs reach outside the mon's
# own types. Same curation rules as ``_MOVE_POOL``: damaging power
# non-decreasing within each bucket.
_TRAIT_MOVES: dict[str, list[tuple[int, int]]] = {
    "fists": [(7,4),(16,325),(22,7),(23,8),(24,9),(31,327),(37,309),(44,223),(50,264)],
    "kicks": [(8,24),(15,27),(23,26),(31,299),(38,136),(47,25)],
    "fangs": [(11,305),(18,44),(30,158),(39,242)],
    "tail":  [(6,39),(19,342),(28,21),(41,231)],
    "wings": [(13,17),(27,211),(36,297)],
    "beak":  [(6,64),(34,65)],
    "horn":  [(10,30),(37,224),(49,32)],
    "claws": [(5,154),(12,10),(23,163),(35,306)],
    "shell": [(5,110),(11,111),(20,229),(38,334)],
}

# Trait-free Normal reach moves used to top a thin moveset up to
# ``_MOVESET_TARGET`` — mono-types and trait-poor bodies would otherwise end
# up with visibly fewer moves than a dual-type.
_FILLER_MOVES: list[tuple[int, int]] = [(18,129),(26,263),(34,161),(42,36)]

# How many level-up moves a species should carry. Base moves (backbone, type
# pools, ability moves) are never trimmed to reach it; trait and filler picks
# stop being added once it is met.
_MOVESET_TARGET = 13

# Normal staples every species learns regardless of typing — real dex entries
# carry these alongside their type moves, and they guarantee an attack from
# level 1 even for mons whose type pools open softly.
_NORMAL_BACKBONE: list[tuple[int, int]] = [(1, 33), (4, 45)]  # Tackle, Growl

# Extra thematic moves injected per custom ability name
_ABILITY_MOVES: dict[str, list[tuple[int, int]]] = {
    "Steam Engine": [(6, 108), (22, 240)],           # Smokescreen, Rain Dance
    "Cheers Up":    [(10, 213), (30, 207)],          # Attract, Swagger
    "Comfy Hide":   [(8, 281), (24, 156), (36, 133), (48, 138)],  # Yawn, Rest, Amnesia, Dream Eater
}

# Fallback mapping for custom ability names not in gen3_abilities.json
_ABILITY_FALLBACK: dict[str, int] = {
    "steam engine": 33,  # → Swift Swim
    "cheers up":    32,  # → Serene Grace
    "comfy hide":   23,  # → Shadow Tag
}

# ── helpers ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _type_index() -> dict[str, int]:
    """Gen 3 type name -> byte index, read once.

    Shares ``gen3_types.json`` with ``generator``, which offers the same pool to
    the model — one list, so the types the model may pick and the types that can
    be encoded cannot drift apart. Lazily, not at import, for the same reason as
    ``_ability_table``.
    """
    table = json.loads((_RESOURCES / "gen3_types.json").read_text(encoding="utf-8"))
    index = {name: int(idx) for idx, name in table.items()}
    index.update({alias: index[target] for alias, target in _TYPE_ALIASES.items()})
    return index


def _resolve_type(name) -> int:
    """Byte index for one type name, degrading to Normal on anything unknown.

    ``generator`` constrains the model to the Gen 3 pool, so this should only
    ever see valid names. It warns and continues rather than raising for the
    sake of a stats.json written before that constraint existed (or edited by
    hand): by the time the export runs, the sprites, cries and footprints of the
    whole line are already on disk, and losing all of it over one bad type field
    is the worse failure.
    """
    index = _type_index()
    if isinstance(name, str) and name in index:
        return index[name]
    print(
        f"warning: _resolve_type got unknown type {name!r}; falling back to Normal",
        file=sys.stderr,
    )
    return index["Normal"]


@lru_cache(maxsize=1)
def _ability_table() -> dict[str, str]:
    """The Gen 3 ability table, read once.

    Lazily, not at import: a missing resource file should fail the export that
    needs it, not the import of the module.
    """
    return json.loads(
        (_RESOURCES / "gen3_abilities.json").read_text(encoding="utf-8")
    )


def _resolve_ability(name: str) -> int:
    abilities = _ability_table()
    lower = name.lower()
    for idx, aname in abilities.items():
        if aname.lower() == lower:
            return int(idx)
    return _ABILITY_FALLBACK.get(lower, 0)


def _ability_indexes(data: dict) -> tuple[int, int]:
    """Resolve the (ability1, ability2) byte indexes for the BaseStats blob.

    Prefers the real Gen 3 names in ``abilities_gen3``; anything malformed
    (absent, empty, or not a list) falls back to the legacy free-text
    ``ability`` with an empty second slot.
    """
    names = data.get("abilities_gen3")
    if not isinstance(names, list) or not names:
        return _resolve_ability(data.get("ability", "")), 0x00

    def idx(pos: int) -> int:
        if pos < len(names) and isinstance(names[pos], str):
            return _resolve_ability(names[pos])
        return 0x00

    return idx(0), idx(1)


def _dimension(data: dict, key: str, legacy: int) -> int:
    """Read a ``Hght``/``Wght`` value, falling back to the legacy literal.

    Keyed on the value's type rather than its truthiness so a legitimate ``0``
    round-trips; a present-but-non-integer value (malformed file) degrades to
    the literal rather than writing a junk token into the .ini. Values are
    already clamped upstream at generation time, so no re-clamping here.
    """
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return legacy


def _pokedex_type(data: dict) -> str:
    """The PokedexType value: the category noun, else the primary type word.

    Upper-cased and clipped to the field budget on both paths. The generator
    already guarantees both for anything it writes, but this function is also
    the last stop for hand-edited and externally produced stats.json — and an
    over-long category would overrun exactly the budget that dropping the
    " POKEMON" suffix was meant to reclaim.
    """
    category = data.get("category")
    if isinstance(category, str) and category.strip():
        return category.upper().strip()[:_MAX_CATEGORY_LEN].strip()
    return data["types"][0].upper()[:_MAX_CATEGORY_LEN]


def _dex_number(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 200 + 1


def _ev_yield(stats: dict) -> int:
    order = [
        ("hp", 0), ("attack", 2), ("defense", 4),
        ("speed", 6), ("sp_atk", 8), ("sp_def", 10),
    ]
    best = max(stats, key=stats.__getitem__)
    for stat, bit in order:
        if stat == best:
            return 1 << bit
    return 1


def _encode_base_stats(data: dict, ability1_idx: int, ability2_idx: int) -> str:
    s = data["base_stats"]
    types = [_resolve_type(t) for t in data["types"]]
    t1 = types[0] if types else _type_index()["Normal"]
    t2 = types[1] if len(types) > 1 else t1
    ev = _ev_yield(s)

    raw = bytes([
        s["hp"], s["attack"], s["defense"], s["speed"], s["sp_atk"], s["sp_def"],
        t1, t2,
        120, 80,                                 # catch rate, base exp
        ev & 0xFF, (ev >> 8) & 0xFF,            # EV yield (packed 2-bit per stat)
        0x00, 0x00, 0x00, 0x00,                  # held items (empty)
        0x7F,                                    # gender: 50/50
        20,                                      # egg cycles
        70,                                      # base happiness
        0,                                       # growth rate: Medium Fast
        11, 11,                                  # egg groups: Amorphous
        ability1_idx & 0xFF, ability2_idx & 0xFF,  # ability1, ability2
        0x00,                                    # safari flee rate
        _TYPE_BODY_COLOR.get(t1, 8),
        0x00, 0x00,                              # padding
    ])
    return raw.hex().upper()


def _moveset_rng(name: str) -> random.Random:
    """Deterministic picker seeded from the species name.

    "Random" trait/filler picks must not reshuffle on re-export, and tests
    must be able to pin them down — the same reasoning that already seeds
    the dex number from a name hash in ``_dex_number``.
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _traits(data: dict) -> list[str]:
    """The stage's trait list, read with the same tolerance as every other
    optional stats.json field: absent, malformed or unknown entries mean
    fewer buckets, never a raise."""
    raw = data.get("traits")
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, str) and t in _TRAIT_MOVES]


def _build_moveset(data: dict) -> list[tuple[int, int]]:
    types = [t if t != "Fairy" else "Normal" for t in data["types"]]
    ability = data.get("ability", "")

    # Deduped by move id, not by level: Gen 3 tables allow several moves at
    # one level, and keying on level silently dropped whichever colliding
    # move arrived second.
    seen: set[int] = set()
    moves: list[tuple[int, int]] = []

    def add(level: int, mid: int) -> None:
        if mid not in seen:
            seen.add(mid)
            moves.append((level, mid))

    def fill_from(pool: list[tuple[int, int]], rng: random.Random) -> None:
        candidates = [(lv, mid) for lv, mid in pool if mid not in seen]
        shortfall = _MOVESET_TARGET - len(moves)
        if shortfall <= 0 or not candidates:
            return
        for lv, mid in rng.sample(candidates, min(shortfall, len(candidates))):
            add(lv, mid)

    for lv, mid in _NORMAL_BACKBONE:
        add(lv, mid)

    for lv, mid in _MOVE_POOL.get(types[0], []):
        add(lv, mid)

    if len(types) > 1 and types[1] != types[0]:
        for lv, mid in _MOVE_POOL.get(types[1], [])[1::2]:
            add(lv, mid)

    for lv, mid in _ABILITY_MOVES.get(ability, []):
        add(lv, mid)

    # Trait buckets, then trait-free filler, top the moveset up to the
    # target — mono-types have more room than dual-types, so they draw more
    # picks and both land at comparable sizes.
    rng = _moveset_rng(data["name"])
    fill_from([m for trait in _traits(data) for m in _TRAIT_MOVES[trait]], rng)
    fill_from(_FILLER_MOVES, rng)

    return sorted(moves)


def _encode_jambo51(moves: list[tuple[int, int]]) -> str:
    out = "".join(f"{mid & 0xFF:02X}{mid >> 8:02X}{lv:02X}" for lv, mid in moves)
    return out + "FF00"


def _encode_original(moves: list[tuple[int, int]]) -> str:
    out = ""
    for lv, mid in moves:
        val = (mid & 0x1FF) | (lv << 9)
        out += f"{val & 0xFF:02X}{(val >> 8) & 0xFF:02X}"
    return out + "FFFF0000"


def _format_entry(text: str) -> str:
    lines = textwrap.wrap(text.strip(), 40)[:4]
    return r"\n".join(lines) + r"\x"


# ── public API ─────────────────────────────────────────────────────────────────

def export_ini(stage_dir: Path) -> Path:
    data = json.loads((stage_dir / "stats.json").read_text(encoding="utf-8"))
    entry = (stage_dir / "entry.md").read_text(encoding="utf-8").strip()

    ability1_idx, ability2_idx = _ability_indexes(data)

    dex = _dex_number(data["name"])
    base_stats = _encode_base_stats(data, ability1_idx, ability2_idx)
    moves = _build_moveset(data)

    height_dm = _dimension(data, "height_dm", 5)
    weight_hg = _dimension(data, "weight_hg", 30)

    dex_type = _pokedex_type(data)

    ini_lines = [
        "[Pokemon]",
        f"PokemonName={data['name'].upper()}",
        f"BaseStats={base_stats}",
        "PlayerY=08",
        "EnemyY=0D",
        "EnemyAlt=08",
        f"EvolutionData={'00' * 40}",
        f"LevelUpAttacksOriginal={_encode_original(moves)}",
        f"LevelUpAttacksJambo51={_encode_jambo51(moves)}",
        "MoveTutorCompatibility=00000000",
        "FrontAnimationTable=0",
        "BackAnimTable=0",
        "AnimDelayTable=0",
        "TMHMCompatibility=0000000000000000",
        f"NationalDexNumber={dex}",
        f"SecondDexNumber={dex}",
        f"Hght={height_dm}",
        f"Wght={weight_hg}",
        "Scale1=256",
        "Scale2=256",
        "Offset_1=0",
        "Offset_2=0",
        f"PokedexDescription={_format_entry(entry)}",
        f"PokedexType={dex_type}",
        "",
    ]

    out = stage_dir / f"{data['name']}.ini"
    out.write_text("\n".join(ini_lines), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export Fakemon stage dirs to .ini")
    parser.add_argument(
        "stage_dirs", nargs="*", metavar="DIR",
        help="Stage directories to export (omit to export all under output/)",
    )
    args = parser.parse_args(argv)

    dirs: list[Path]
    if args.stage_dirs:
        dirs = [Path(d) for d in args.stage_dirs]
    else:
        dirs = sorted(Path("output").glob("*/stage*_*/"))

    exported = 0
    for d in dirs:
        if not (d / "stats.json").exists():
            continue
        path = export_ini(d)
        print(f"Wrote {path}")
        exported += 1

    if exported == 0:
        print("No stage directories found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
