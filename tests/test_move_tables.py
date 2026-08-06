"""Curation rules for the level-up move tables in export_ini.

The tables are hand-curated data; these tests are what keeps them honest:
every entry the type it claims, damaging power non-decreasing with level,
a damaging move at level 1, and anatomy moves confined to trait buckets.

`_MOVE_DATA` mirrors the Gen 3 move list for exactly the ids the tables
use: id -> (name, type, power). Power is an int for standard damage,
"fixed" for fixed/level/weight-based damage and one-hit KOs (all damaging,
none comparable by power), and None for status moves.
"""

import json
from pathlib import Path

import pytest

from fakemon_forge.export_ini import (
    _ABILITY_MOVES,
    _FILLER_MOVES,
    _MOVE_POOL,
    _NORMAL_BACKBONE,
    _TM_BY_TRAIT,
    _TM_BY_TYPE,
    _TM_UNIVERSAL,
    _TRAIT_MOVES,
)

_MOVE_DATA = {
    4:   ("Comet Punch", "Normal", 18),
    7:   ("Fire Punch", "Fire", 75),
    8:   ("Ice Punch", "Ice", 75),
    9:   ("ThunderPunch", "Electric", 75),
    10:  ("Scratch", "Normal", 40),
    16:  ("Gust", "Flying", 40),
    17:  ("Wing Attack", "Flying", 60),
    19:  ("Fly", "Flying", 70),
    21:  ("Slam", "Normal", 80),
    22:  ("Vine Whip", "Grass", 35),
    24:  ("Double Kick", "Fighting", 30),
    25:  ("Mega Kick", "Normal", 120),
    26:  ("Jump Kick", "Fighting", 70),
    27:  ("Rolling Kick", "Fighting", 60),
    28:  ("Sand Attack", "Ground", None),
    29:  ("Headbutt", "Normal", 70),
    30:  ("Horn Attack", "Normal", 65),
    32:  ("Horn Drill", "Normal", "fixed"),
    33:  ("Tackle", "Normal", 35),
    34:  ("Body Slam", "Normal", 85),
    36:  ("Take Down", "Normal", 90),
    38:  ("Double-Edge", "Normal", 120),
    39:  ("Tail Whip", "Normal", None),
    40:  ("Poison Sting", "Poison", 15),
    42:  ("Pin Missile", "Bug", 14),
    44:  ("Bite", "Dark", 60),
    45:  ("Growl", "Normal", None),
    51:  ("Acid", "Poison", 40),
    52:  ("Ember", "Fire", 40),
    53:  ("Flamethrower", "Fire", 95),
    55:  ("Water Gun", "Water", 40),
    56:  ("Hydro Pump", "Water", 120),
    57:  ("Surf", "Water", 95),
    58:  ("Ice Beam", "Ice", 95),
    59:  ("Blizzard", "Ice", 120),
    60:  ("Psybeam", "Psychic", 65),
    61:  ("BubbleBeam", "Water", 65),
    62:  ("Aurora Beam", "Ice", 65),
    63:  ("Hyper Beam", "Normal", 150),
    64:  ("Peck", "Flying", 35),
    65:  ("Drill Peck", "Flying", 80),
    67:  ("Low Kick", "Fighting", "fixed"),
    68:  ("Counter", "Fighting", "fixed"),
    71:  ("Absorb", "Grass", 20),
    73:  ("Leech Seed", "Grass", None),
    75:  ("Razor Leaf", "Grass", 55),
    76:  ("SolarBeam", "Grass", 120),
    77:  ("Poison Powder", "Poison", None),
    80:  ("Petal Dance", "Grass", 70),
    81:  ("String Shot", "Bug", None),
    82:  ("Dragon Rage", "Dragon", "fixed"),
    84:  ("ThunderShock", "Electric", 40),
    85:  ("Thunderbolt", "Electric", 95),
    86:  ("Thunder Wave", "Electric", None),
    87:  ("Thunder", "Electric", 120),
    88:  ("Rock Throw", "Rock", 50),
    89:  ("Earthquake", "Ground", 100),
    90:  ("Fissure", "Ground", "fixed"),
    91:  ("Dig", "Ground", 60),
    92:  ("Toxic", "Poison", None),
    93:  ("Confusion", "Psychic", 50),
    94:  ("Psychic", "Psychic", 90),
    95:  ("Hypnosis", "Psychic", None),
    98:  ("Quick Attack", "Normal", 40),
    101: ("Night Shade", "Ghost", "fixed"),
    104: ("Double Team", "Normal", None),
    108: ("Smokescreen", "Normal", None),
    109: ("Confuse Ray", "Ghost", None),
    110: ("Withdraw", "Water", None),
    111: ("Defense Curl", "Normal", None),
    124: ("Sludge", "Poison", 65),
    126: ("Fire Blast", "Fire", 120),
    127: ("Waterfall", "Water", 80),
    129: ("Swift", "Normal", 60),
    133: ("Amnesia", "Psychic", None),
    136: ("Hi Jump Kick", "Fighting", 85),
    138: ("Dream Eater", "Psychic", 100),
    141: ("Leech Life", "Bug", 20),
    143: ("Sky Attack", "Flying", 140),
    145: ("Bubble", "Water", 20),
    154: ("Fury Swipes", "Normal", 18),
    156: ("Rest", "Psychic", None),
    157: ("Rock Slide", "Rock", 75),
    158: ("Hyper Fang", "Normal", 80),
    161: ("Tri Attack", "Normal", 80),
    163: ("Slash", "Normal", 70),
    168: ("Thief", "Dark", 40),
    171: ("Nightmare", "Ghost", None),
    172: ("Flame Wheel", "Fire", 60),
    181: ("Powder Snow", "Ice", 40),
    185: ("Faint Attack", "Dark", 60),
    188: ("Sludge Bomb", "Poison", 90),
    189: ("Mud-Slap", "Ground", 20),
    192: ("Zap Cannon", "Electric", 100),
    194: ("Destiny Bond", "Ghost", None),
    196: ("Icy Wind", "Ice", 55),
    200: ("Outrage", "Dragon", 90),
    202: ("Giga Drain", "Grass", 60),
    205: ("Rollout", "Rock", 30),
    207: ("Swagger", "Normal", None),
    209: ("Spark", "Electric", 65),
    211: ("Steel Wing", "Steel", 70),
    213: ("Attract", "Normal", None),
    216: ("Return", "Normal", 102),
    223: ("DynamicPunch", "Fighting", 100),
    224: ("Megahorn", "Bug", 120),
    225: ("DragonBreath", "Dragon", 60),
    228: ("Pursuit", "Dark", 40),
    229: ("Rapid Spin", "Normal", 20),
    231: ("Iron Tail", "Steel", 100),
    232: ("Metal Claw", "Steel", 50),
    233: ("Vital Throw", "Fighting", 70),
    238: ("Cross Chop", "Fighting", 100),
    239: ("Twister", "Dragon", 40),
    240: ("Rain Dance", "Water", None),
    241: ("Sunny Day", "Fire", None),
    242: ("Crunch", "Dark", 80),
    246: ("AncientPower", "Rock", 60),
    247: ("Shadow Ball", "Ghost", 80),
    248: ("Future Sight", "Psychic", 80),
    257: ("Heat Wave", "Fire", 100),
    258: ("Hail", "Ice", None),
    263: ("Facade", "Normal", 70),
    264: ("Focus Punch", "Fighting", 150),
    269: ("Taunt", "Dark", None),
    276: ("Superpower", "Fighting", 120),
    280: ("Brick Break", "Fighting", 75),
    281: ("Yawn", "Normal", None),
    297: ("Feather Dance", "Flying", None),
    299: ("Blaze Kick", "Fire", 85),
    305: ("Poison Fang", "Poison", 50),
    306: ("Crush Claw", "Normal", 75),
    309: ("Meteor Mash", "Steel", 100),
    310: ("Astonish", "Ghost", 30),
    314: ("Air Cutter", "Flying", 55),
    315: ("Overheat", "Fire", 140),
    317: ("Rock Tomb", "Rock", 50),
    318: ("Silver Wind", "Bug", 60),
    319: ("Metal Sound", "Steel", None),
    324: ("Signal Beam", "Bug", 75),
    325: ("Shadow Punch", "Ghost", 60),
    327: ("Sky Uppercut", "Fighting", 85),
    329: ("Sheer Cold", "Ice", "fixed"),
    332: ("Aerial Ace", "Flying", 60),
    334: ("Iron Defense", "Steel", None),
    337: ("Dragon Claw", "Dragon", 80),
    338: ("Frenzy Plant", "Grass", 150),
    341: ("Mud Shot", "Ground", 55),
    342: ("Poison Tail", "Poison", 50),
    347: ("Calm Mind", "Psychic", None),
    349: ("Dragon Dance", "Dragon", None),
    351: ("Shock Wave", "Electric", 60),
    352: ("Water Pulse", "Water", 60),
    353: ("Doom Desire", "Steel", 120),
}

# Documented exceptions to the no-anatomy-in-type-pools rule: Metal Claw is
# Gen 3's only non-anatomical damaging Steel option, dragons have claws by
# definition, and Iron Defense is Steel's own defensive staple — the shell
# bucket merely borrows it.
_ANATOMY_EXEMPT = {232, 337, 334}

_ALL_TABLES = {
    **{f"pool:{t}": pool for t, pool in _MOVE_POOL.items()},
    **{f"trait:{t}": pool for t, pool in _TRAIT_MOVES.items()},
    "filler": _FILLER_MOVES,
    "backbone": _NORMAL_BACKBONE,
    **{f"ability:{a}": pool for a, pool in _ABILITY_MOVES.items()},
}

_TRAIT_MOVE_IDS = {mid for pool in _TRAIT_MOVES.values() for _, mid in pool}


def _damaging(pool):
    """(level, power) pairs for the pool's standard damaging moves."""
    return [
        (lv, _MOVE_DATA[mid][2])
        for lv, mid in pool
        if isinstance(_MOVE_DATA[mid][2], int)
    ]


# ---------------------------------------------------------------------------
# Coverage of the fixture itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", _ALL_TABLES)
def test_every_table_entry_has_move_data(table):
    for _, mid in _ALL_TABLES[table]:
        assert mid in _MOVE_DATA, f"move id {mid} missing from _MOVE_DATA"


@pytest.mark.parametrize("table", _ALL_TABLES)
def test_levels_are_valid_and_ascending(table):
    levels = [lv for lv, _ in _ALL_TABLES[table]]
    assert levels == sorted(levels)
    assert all(1 <= lv <= 100 for lv in levels)


# ---------------------------------------------------------------------------
# Type pools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_name", _MOVE_POOL)
def test_pool_entries_match_their_type(type_name):
    """Regression: the old Ground pool ended in Muddy Water (Water) and
    Eruption (Fire), and Rock's capstone was Blast Burn."""
    for _, mid in _MOVE_POOL[type_name]:
        name, move_type, _ = _MOVE_DATA[mid]
        assert move_type == type_name, f"{name} is {move_type}, not {type_name}"


@pytest.mark.parametrize("type_name", _MOVE_POOL)
def test_pool_opens_with_a_damaging_move_at_level_one(type_name):
    lv, mid = _MOVE_POOL[type_name][0]
    assert lv == 1
    assert _MOVE_DATA[mid][2] is not None, f"{_MOVE_DATA[mid][0]} is a status move"


@pytest.mark.parametrize("type_name", _MOVE_POOL)
def test_pool_keeps_anatomy_moves_in_trait_buckets(type_name):
    for _, mid in _MOVE_POOL[type_name]:
        assert mid in _ANATOMY_EXEMPT or mid not in _TRAIT_MOVE_IDS, (
            f"{_MOVE_DATA[mid][0]} belongs to a trait bucket"
        )


# ---------------------------------------------------------------------------
# Power progression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "table",
    [k for k in _ALL_TABLES if not k.startswith("ability:")],
)
def test_damaging_power_never_decreases_with_level(table):
    """Regression: the old Ground pool taught Earthquake (100) at 10 and Mud
    Shot (55) at 30 — levelling up made your newest attack weaker."""
    powers = [p for _, p in _damaging(_ALL_TABLES[table])]
    assert powers == sorted(powers), f"{table}: {powers}"


# ---------------------------------------------------------------------------
# Trait vocabulary stays in sync with the generator's resource file
# ---------------------------------------------------------------------------

def test_no_move_lives_in_two_trait_buckets():
    """A move in two buckets would appear twice in the concatenated pick
    candidates; ``rng.sample`` could then draw both copies and the by-id
    dedup would leave the moveset one short of its target."""
    seen: dict[int, str] = {}
    for trait, pool in _TRAIT_MOVES.items():
        for _, mid in pool:
            assert mid not in seen, (
                f"{_MOVE_DATA[mid][0]} is in both {seen[mid]!r} and {trait!r}"
            )
            seen[mid] = trait


def test_trait_buckets_match_the_shared_vocabulary():
    resource = json.loads(
        (Path(__file__).parent.parent / "resources" / "traits.json")
        .read_text(encoding="utf-8")
    )
    assert set(_TRAIT_MOVES) == set(resource)


# ---------------------------------------------------------------------------
# TM / HM tables
# ---------------------------------------------------------------------------

def test_tm_type_table_covers_exactly_the_move_pool_types():
    assert set(_TM_BY_TYPE) == set(_MOVE_POOL)


def test_tm_trait_table_matches_the_shared_vocabulary():
    resource = json.loads(
        (Path(__file__).parent.parent / "resources" / "traits.json")
        .read_text(encoding="utf-8")
    )
    assert set(_TM_BY_TRAIT) == set(resource)


def test_tm_numbers_stay_inside_the_58_machine_range():
    """1-50 are TMs, 51-58 are HMs; anything else would set a bit the ROM
    record does not have."""
    all_numbers = set(_TM_UNIVERSAL)
    for pool in (*_TM_BY_TYPE.values(), *_TM_BY_TRAIT.values()):
        all_numbers |= pool
    assert all_numbers <= set(range(1, 59))


def test_anatomy_machines_are_not_granted_by_type():
    """Steel Wing (TM47), Cut (HM01) and Iron Tail (TM23) are trait-gated;
    granting them per type would recreate the Steelit-with-Steel-Wing bug
    at the TM layer."""
    for type_name, pool in _TM_BY_TYPE.items():
        assert not pool & {47, 51, 23}, type_name
