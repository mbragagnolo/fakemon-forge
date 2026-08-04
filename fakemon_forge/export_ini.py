"""Export a Fakemon stage directory to a Gen 3-style .ini data file."""

import hashlib
import json
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

# Thematic level-up move pools per type (level, move_id)
_MOVE_POOL: dict[str, list[tuple[int, int]]] = {
    "Normal":   [(1,33),(6,45),(14,34),(24,216),(36,39),(44,63)],
    "Water":    [(1,145),(8,55),(15,61),(22,240),(30,57),(38,352),(46,56)],
    "Fire":     [(1,52),(8,83),(16,53),(24,126),(32,241),(44,315)],
    "Grass":    [(1,22),(8,71),(16,75),(24,76),(32,202),(44,338)],
    "Electric": [(1,84),(10,86),(20,85),(30,87),(40,351),(50,192)],
    "Ice":      [(1,181),(10,58),(20,196),(30,59),(40,258),(50,329)],
    "Fighting": [(1,67),(10,68),(20,238),(30,280),(40,276),(50,223)],
    "Poison":   [(1,40),(10,51),(20,124),(30,188),(40,92),(50,305)],
    "Ground":   [(1,28),(10,89),(20,189),(30,341),(40,330),(50,284)],
    "Flying":   [(1,16),(10,64),(20,332),(30,314),(40,239),(50,143)],
    "Bug":      [(1,81),(10,42),(20,210),(30,318),(40,224),(50,141)],
    "Rock":     [(1,88),(10,157),(20,317),(30,350),(40,246),(50,307)],
    "Ghost":    [(1,310),(10,247),(20,109),(30,325),(40,171),(50,194)],
    "Dragon":   [(1,82),(10,239),(20,225),(30,337),(40,349),(50,200)],
    "Dark":     [(1,44),(10,168),(20,228),(30,242),(40,247),(50,262)],
    "Steel":    [(1,106),(10,232),(20,211),(30,309),(40,334),(50,231)],
    "Psychic":  [(1,93),(10,95),(20,60),(30,94),(40,347),(50,326)],
}

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


def _build_moveset(data: dict) -> list[tuple[int, int]]:
    types = [t if t != "Fairy" else "Normal" for t in data["types"]]
    ability = data.get("ability", "")

    seen: dict[int, int] = {}  # level → move_id

    def add(level: int, mid: int) -> None:
        if level not in seen:
            seen[level] = mid

    for lv, mid in _MOVE_POOL.get(types[0], []):
        add(lv, mid)

    if len(types) > 1 and types[1] != types[0]:
        for lv, mid in _MOVE_POOL.get(types[1], [])[1::2]:
            add(lv + (2 if lv in seen else 0), mid)

    for lv, mid in _ABILITY_MOVES.get(ability, []):
        add(lv + (2 if lv in seen else 0), mid)

    return sorted(seen.items())


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
