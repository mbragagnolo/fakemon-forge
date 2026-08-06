import json
import pytest

from fakemon_forge.export_ini import (
    _TRAIT_MOVES,
    _resolve_type,
    _type_index,
    enrich_line,
    export_ini,
)
from fakemon_forge.generator import _TYPE_POOL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENTRY = "A small fiery creature with a burning tail tip."

_BASE = {
    "name": "Flamburr",
    "stage": 1,
    "types": ["Fire"],
    "ability": "Blaze",
    "base_stats": {
        "hp": 45, "attack": 52, "defense": 43,
        "sp_atk": 60, "sp_def": 50, "speed": 65,
    },
}

# Canonical indexes from resources/gen3_abilities.json
_BLAZE = "42"      # 66
_SAND_VEIL = "08"  # 8
_NONE = "00"


def _write_stage(tmp_path, data, entry=_ENTRY):
    (tmp_path / "stats.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "entry.md").write_text(entry, encoding="utf-8")
    return tmp_path


def _read_ini(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def _export(tmp_path, data, entry=_ENTRY):
    return _read_ini(export_ini(_write_stage(tmp_path, data, entry)))


def _abilities(fields):
    """(ability1, ability2) hex bytes at offsets 22/23 of the BaseStats blob."""
    blob = fields["BaseStats"]
    return blob[44:46], blob[46:48]


# ---------------------------------------------------------------------------
# New-format round trip
# ---------------------------------------------------------------------------

def test_new_format_round_trip(tmp_path):
    fields = _export(tmp_path, {
        **_BASE,
        "height_dm": 7,
        "weight_hg": 120,
        "abilities_gen3": ["Blaze", "Sand Veil"],
        "category": "Flame",
    })

    assert fields["Hght"] == "7"
    assert fields["Wght"] == "120"
    assert fields["PokedexType"] == "FLAME"
    assert _abilities(fields) == (_BLAZE, _SAND_VEIL)


def test_output_path_and_blob_length_unchanged(tmp_path):
    stage_dir = _write_stage(tmp_path, {**_BASE, "abilities_gen3": ["Blaze", "Sand Veil"]})
    out = export_ini(stage_dir)

    assert out == stage_dir / "Flamburr.ini"
    # 28-byte BaseStats blob; the ability2 byte must not have grown the record.
    assert len(_read_ini(out)["BaseStats"]) == 56


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------

def test_legacy_fallback(tmp_path):
    fields = _export(tmp_path, dict(_BASE))

    assert fields["Hght"] == "5"
    assert fields["Wght"] == "30"
    assert _abilities(fields) == (_BLAZE, _NONE)
    # PokedexType keeps the type-word derivation, minus the dropped suffix.
    assert fields["PokedexType"] == "FIRE"
    assert " POKEMON" not in fields["PokedexType"]


def test_legacy_missing_ability_key_resolves_to_zero(tmp_path):
    data = {k: v for k, v in _BASE.items() if k != "ability"}
    assert _abilities(_export(tmp_path, data)) == (_NONE, _NONE)


# ---------------------------------------------------------------------------
# Height / weight
# ---------------------------------------------------------------------------

def test_zero_height_round_trips(tmp_path):
    """0 is falsy but present — it must not fall back to the legacy literal."""
    assert _export(tmp_path, {**_BASE, "height_dm": 0})["Hght"] == "0"


@pytest.mark.parametrize("data, hght, wght", [
    ({"height_dm": 7}, "7", "30"),
    ({"weight_hg": 120}, "5", "120"),
])
def test_height_and_weight_fall_back_independently(tmp_path, data, hght, wght):
    fields = _export(tmp_path, {**_BASE, **data})
    assert (fields["Hght"], fields["Wght"]) == (hght, wght)


@pytest.mark.parametrize("bad", [None, "tall", True, [7], {"dm": 7}])
def test_non_integer_dimensions_fall_back(tmp_path, bad):
    """A malformed value degrades to the legacy literal, never a junk token."""
    fields = _export(tmp_path, {**_BASE, "height_dm": bad, "weight_hg": bad})
    assert (fields["Hght"], fields["Wght"]) == ("5", "30")


# ---------------------------------------------------------------------------
# Ability bytes
# ---------------------------------------------------------------------------

def test_abilities_gen3_single_entry(tmp_path):
    fields = _export(tmp_path, {**_BASE, "abilities_gen3": ["Sand Veil"]})
    assert _abilities(fields) == (_SAND_VEIL, _NONE)


def test_abilities_gen3_takes_precedence_over_free_text_ability(tmp_path):
    """ability1 comes from abilities_gen3, not data["ability"], when present."""
    fields = _export(tmp_path, {**_BASE, "abilities_gen3": ["Sand Veil"]})
    ability1, _ = _abilities(fields)
    assert ability1 == _SAND_VEIL != _BLAZE


def test_unresolvable_abilities_gen3_entry_resolves_to_zero(tmp_path):
    """No canonical match and no _ABILITY_FALLBACK entry → index 0, no raise."""
    fields = _export(tmp_path, {**_BASE, "abilities_gen3": ["Not An Ability", "Blaze"]})
    assert _abilities(fields) == (_NONE, _BLAZE)


@pytest.mark.parametrize("value", [[], None])
def test_absent_or_empty_abilities_gen3_falls_back_to_ability(tmp_path, value):
    data = {**_BASE, "abilities_gen3": value} if value is not None else dict(_BASE)
    assert _abilities(_export(tmp_path, data)) == (_BLAZE, _NONE)


@pytest.mark.parametrize("bad", ["Blaze", 42, {"1": "Blaze"}])
def test_non_list_abilities_gen3_falls_back_to_ability(tmp_path, bad):
    """A malformed abilities_gen3 is treated as absent rather than raising."""
    assert _abilities(_export(tmp_path, {**_BASE, "abilities_gen3": bad})) == (_BLAZE, _NONE)


@pytest.mark.parametrize("entries, expected", [
    ([None, "Sand Veil"], (_NONE, _SAND_VEIL)),
    (["Blaze", 42], (_BLAZE, _NONE)),
])
def test_non_string_abilities_gen3_entries_resolve_to_zero(tmp_path, entries, expected):
    assert _abilities(_export(tmp_path, {**_BASE, "abilities_gen3": entries})) == expected


def test_ability_moves_stay_keyed_on_free_text_ability(tmp_path):
    """_ABILITY_MOVES is not consulted for abilities_gen3 (out of scope per spec)."""
    custom = {**_BASE, "ability": "Steam Engine"}
    legacy = _export(tmp_path, custom)

    with_gen3 = _export(tmp_path, {**custom, "abilities_gen3": ["Sand Veil"]})

    # Same movepool (still driven by "Steam Engine")...
    assert with_gen3["LevelUpAttacksOriginal"] == legacy["LevelUpAttacksOriginal"]
    # ...but the ability byte now comes from abilities_gen3, not the fallback map.
    assert _abilities(with_gen3)[0] == _SAND_VEIL
    assert _abilities(legacy)[0] == f"{33:02X}"  # _ABILITY_FALLBACK["steam engine"]


# ---------------------------------------------------------------------------
# PokedexType
# ---------------------------------------------------------------------------

def test_pokedex_type_from_category(tmp_path):
    fields = _export(tmp_path, {**_BASE, "category": "Flame"})
    assert fields["PokedexType"] == "FLAME"
    assert " POKEMON" not in fields["PokedexType"]


def test_pokedex_type_category_is_upper_cased(tmp_path):
    """Both branches emit the same register — the type-word fallback is upper
    case, so a category that isn't would read differently for no reason."""
    assert _export(tmp_path, {**_BASE, "category": "Sea Otter"})["PokedexType"] == "SEA OTTER"


def test_pokedex_type_category_is_clipped_to_the_field_budget(tmp_path):
    """An over-long category would overrun exactly the budget that dropping
    the " POKEMON" suffix reclaimed."""
    fields = _export(tmp_path, {**_BASE, "category": "A REALLY LONG CATEGORY NOUN"})
    assert fields["PokedexType"] == "A REALLY LO"
    assert len(fields["PokedexType"]) == 11


def test_pokedex_type_eleven_chars_survives(tmp_path):
    """`> 11` is the failure condition, not `>= 11`."""
    assert len("TINY TURTLE") == 11
    assert _export(tmp_path, {**_BASE, "category": "TINY TURTLE"})["PokedexType"] == "TINY TURTLE"


def test_pokedex_type_clip_leaves_no_trailing_space(tmp_path):
    assert _export(tmp_path, {**_BASE, "category": "GIANT SEED PODS"})["PokedexType"] == "GIANT SEED"


def test_pokedex_type_whitespace_only_category_falls_back(tmp_path):
    assert _export(tmp_path, {**_BASE, "category": "   "})["PokedexType"] == "FIRE"


@pytest.mark.parametrize("category", ["", None, 42, ["Flame"]])
def test_pokedex_type_falls_back_to_type_word(tmp_path, category):
    fields = _export(tmp_path, {**_BASE, "category": category})
    assert fields["PokedexType"] == "FIRE"
    assert " POKEMON" not in fields["PokedexType"]


def test_pokedex_type_uses_primary_type_of_dual_type(tmp_path):
    fields = _export(tmp_path, {**_BASE, "types": ["Water", "Flying"]})
    assert fields["PokedexType"] == "WATER"


# ---------------------------------------------------------------------------
# Type encoding
# ---------------------------------------------------------------------------

def _types(fields):
    """(type1, type2) hex bytes at offsets 6/7 of the BaseStats blob."""
    blob = fields["BaseStats"]
    return blob[12:14], blob[14:16]


def test_every_generator_type_encodes():
    """The two pools share one resource file, so anything the model may now be
    given is guaranteed to have a byte."""
    for type_name in _TYPE_POOL:
        assert _resolve_type(type_name) == _type_index()[type_name]


def test_mono_type_repeats_its_index(tmp_path):
    t1, t2 = _types(_export(tmp_path, {**_BASE, "types": ["Water"]}))
    assert t1 == t2


def test_unknown_type_degrades_to_normal_instead_of_raising(tmp_path, capsys):
    """A stats.json written before the generator constrained types — the run
    that produced it is already fully rendered, so the export warns rather than
    throwing all of it away."""
    fields = _export(tmp_path, {**_BASE, "types": ["Grass", "Sound"]})
    t1, t2 = _types(fields)
    assert t1 == f"{_type_index()['Grass']:02X}"
    assert t2 == f"{_type_index()['Normal']:02X}"
    assert "unknown type 'Sound'" in capsys.readouterr().err


def test_fairy_still_encodes_as_normal(tmp_path):
    """Fairy is outside the generator pool but kept as an export alias."""
    t1, _ = _types(_export(tmp_path, {**_BASE, "types": ["Fairy"]}))
    assert t1 == f"{_type_index()['Normal']:02X}"


# ---------------------------------------------------------------------------
# Level-up moves
# ---------------------------------------------------------------------------

_TACKLE, _GROWL = 33, 45


def _moves(fields):
    """Decoded (level, move_id) pairs from the original-format level-up blob."""
    blob = fields["LevelUpAttacksOriginal"]
    assert blob.endswith("FFFF0000")
    body = blob[:-8]
    pairs = []
    for i in range(0, len(body), 4):
        val = int(body[i:i + 2], 16) | (int(body[i + 2:i + 4], 16) << 8)
        pairs.append((val >> 9, val & 0x1FF))
    return pairs


def test_every_species_learns_the_normal_backbone(tmp_path):
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Ghost"]}))
    assert (1, _TACKLE) in moves
    assert (4, _GROWL) in moves


def test_backbone_does_not_displace_the_primary_type_level_one_attack(tmp_path):
    """Tackle also sits at level 1 — Ember must coexist with it, not vanish."""
    moves = _moves(_export(tmp_path, dict(_BASE)))
    assert (1, 52) in moves  # Ember
    assert (1, _TACKLE) in moves


def test_backbone_is_not_duplicated_for_normal_types(tmp_path):
    """A Normal mon draws Tackle/Growl from its own pool too — once each."""
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Normal"]}))
    ids = [mid for _, mid in moves]
    assert len(ids) == len(set(ids))
    assert moves.count((1, _TACKLE)) == 1
    # Growl keeps its backbone slot; the pool's level-6 copy is the duplicate.
    assert (4, _GROWL) in moves and (6, _GROWL) not in moves


@pytest.mark.parametrize("type_name, attack", [
    ("Ground", 189),  # Mud-Slap, not Sand Attack
    ("Steel", 232),   # Metal Claw, not Harden
    ("Bug", 42),      # Pin Missile, not String Shot
])
def test_every_type_pool_opens_with_a_damaging_move(tmp_path, type_name, attack):
    """The three pools that used to open with a status move — a low-level
    encounter drew only that move and had no way to deal damage."""
    moves = _moves(_export(tmp_path, {**_BASE, "types": [type_name]}))
    assert (1, attack) in moves


def test_same_level_moves_from_both_types_coexist(tmp_path):
    """Water's Rain Dance and Fire's Flamethrower slice both land on level 26;
    Gen 3 tables allow that, so neither is shifted or dropped."""
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Water", "Fire"]}))
    assert (26, 240) in moves
    assert (26, 53) in moves


def test_ability_move_survives_a_double_level_collision(tmp_path):
    """Fire/Normal + Comfy Hide: Rest's level 24 and the +2 bump slot were both
    taken, which used to drop the move entirely."""
    fields = _export(tmp_path, {
        **_BASE, "types": ["Fire", "Normal"], "ability": "Comfy Hide",
    })
    assert 156 in [mid for _, mid in _moves(fields)]  # Rest


def test_moveset_is_sorted_by_level(tmp_path):
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Water", "Fire"]}))
    assert moves == sorted(moves)


# ---------------------------------------------------------------------------
# Trait buckets and filler
# ---------------------------------------------------------------------------

_STEEL_WING, _WING_ATTACK, _FEATHER_DANCE = 211, 17, 297
_FILLER_IDS = [129, 263, 161, 36]  # Swift, Facade, Tri Attack, Take Down


def test_no_trait_means_no_bucket_moves(tmp_path):
    """The complaint that started this: Steelit learned Steel Wing without
    wings. Without the trait, nothing from the wings bucket may appear."""
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Steel"]}))
    ids = {mid for _, mid in moves}
    assert not ids & {_STEEL_WING, _WING_ATTACK, _FEATHER_DANCE}


def test_trait_unlocks_its_bucket(tmp_path):
    """A mono-Steel's shortfall exceeds the wings bucket, so having the trait
    deterministically pulls in the whole bucket."""
    fields = _export(tmp_path, {**_BASE, "types": ["Steel"], "traits": ["wings"]})
    ids = {mid for _, mid in _moves(fields)}
    assert {_STEEL_WING, _WING_ATTACK, _FEATHER_DANCE} <= ids


@pytest.mark.parametrize("traits", ["wings", 42, ["wings", 7, "gills"]])
def test_malformed_traits_are_tolerated(tmp_path, traits):
    """A non-list, or unknown/non-string entries, mean fewer buckets — never
    a raise, matching every other optional stats.json field."""
    fields = _export(tmp_path, {**_BASE, "types": ["Steel"], "traits": traits})
    assert "LevelUpAttacksOriginal" in fields


def test_filler_tops_a_thin_moveset_up(tmp_path):
    """Mono-Steel with no traits has the thinnest base; all four filler
    moves must be drawn in."""
    moves = _moves(_export(tmp_path, {**_BASE, "types": ["Steel"]}))
    ids = {mid for _, mid in moves}
    assert set(_FILLER_IDS) <= ids


def test_mono_and_dual_types_reach_the_same_size(tmp_path):
    """The old builder gave a mono-type visibly fewer moves than a dual."""
    mono = _moves(_export(tmp_path, {**_BASE, "types": ["Water"]}))
    dual = _moves(_export(tmp_path, {**_BASE, "types": ["Water", "Fire"]}))
    assert len(mono) == len(dual) == 13


_FISTS_IDS = {4, 325, 7, 8, 9, 327, 309, 223, 264}


def test_traits_survive_a_full_base_moveset(tmp_path):
    """Water/Fire's pools alone fill the 13-move target; without the trait
    floor a dual-type with fists never threw a single punch."""
    fields = _export(tmp_path, {
        **_BASE, "types": ["Water", "Fire"], "traits": ["fists"],
    })
    moves = _moves(fields)
    ids = {mid for _, mid in moves}
    assert len(ids & _FISTS_IDS) == 2  # the floor, and only the floor
    assert len(moves) == 15


def test_moveset_is_stable_across_reexports(tmp_path):
    """Trait/filler picks are seeded from the name: re-exporting the same
    stage must never reshuffle its moves."""
    data = {**_BASE, "types": ["Steel"], "traits": ["wings", "claws"]}
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    first = _export(dir_a, data)
    second = _export(dir_b, data)
    assert first["LevelUpAttacksOriginal"] == second["LevelUpAttacksOriginal"]


def test_moveset_never_repeats_a_move(tmp_path):
    """Trait buckets overlap the pools (Iron Defense is in both the Steel
    pool and the shell bucket); dedup is by move id."""
    fields = _export(tmp_path, {
        **_BASE, "types": ["Steel"], "traits": ["shell", "fists", "tail"],
    })
    ids = [mid for _, mid in _moves(fields)]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# TM / HM compatibility
# ---------------------------------------------------------------------------

_TOXIC, _HYPER_BEAM, _FLAMETHROWER, _STEEL_WING_TM = 6, 15, 35, 47
_FOCUS_PUNCH, _BRICK_BREAK, _STRENGTH = 1, 31, 54
_CUT, _FLY, _SURF, _WATERFALL, _DIVE = 51, 52, 53, 57, 58

# _BASE's stats sum to 331 with attack 52 — under both stat gates, so tests
# opt in explicitly where a gate is the subject.
_STRONG_STATS = {"hp": 80, "attack": 80, "defense": 70,
                 "sp_atk": 70, "sp_def": 60, "speed": 60}


def _machines(fields):
    """TM/HM numbers (1-50, 51-58 = HM01-08) decoded from the bitfield."""
    raw = int.from_bytes(bytes.fromhex(fields["TMHMCompatibility"]), "little")
    return {n for n in range(1, 59) if raw & (1 << (n - 1))}


def test_tmhm_blob_stays_eight_bytes(tmp_path):
    assert len(_export(tmp_path, dict(_BASE))["TMHMCompatibility"]) == 16


def test_tmhm_bit_layout_matches_the_rom_record(tmp_path):
    """Pins endianness against the injector's verbatim 8-byte write: TM06
    (Toxic) is byte 0 bit 5, HM03 (Surf) is byte 6 bit 4."""
    blob = bytes.fromhex(
        _export(tmp_path, {**_BASE, "types": ["Water"]})["TMHMCompatibility"]
    )
    assert blob[0] >> 5 & 1  # TM06 Toxic, universal
    assert blob[6] >> 4 & 1  # HM03 Surf, Water


def test_every_mon_gets_the_universal_machines(tmp_path):
    machines = _machines(_export(tmp_path, {**_BASE, "types": ["Ghost"]}))
    assert _TOXIC in machines


def test_type_machines_follow_the_types(tmp_path):
    machines = _machines(_export(tmp_path, dict(_BASE)))  # mono-Fire
    assert _FLAMETHROWER in machines
    assert not machines & {_SURF, _WATERFALL, _DIVE}


def test_water_types_can_learn_the_field_hms(tmp_path):
    machines = _machines(_export(tmp_path, {**_BASE, "types": ["Water"]}))
    assert {_SURF, _WATERFALL, _DIVE} <= machines


def test_trait_machines_follow_the_traits(tmp_path):
    bare = _machines(_export(tmp_path, {**_BASE, "types": ["Steel"]}))
    winged = _machines(_export(
        tmp_path, {**_BASE, "types": ["Steel"], "traits": ["wings"]},
    ))
    assert not bare & {_STEEL_WING_TM, _FLY}
    assert {_STEEL_WING_TM, _FLY} <= winged


def test_fists_unlock_the_punch_machines(tmp_path):
    machines = _machines(_export(
        tmp_path, {**_BASE, "types": ["Steel"], "traits": ["fists"]},
    ))
    assert {_FOCUS_PUNCH, _BRICK_BREAK, _STRENGTH} <= machines


def test_hyper_beam_needs_the_bst_gate(tmp_path):
    weak = _machines(_export(tmp_path, dict(_BASE)))
    strong = _machines(_export(
        tmp_path, {**_BASE, "base_stats": _STRONG_STATS},
    ))
    assert _HYPER_BEAM not in weak
    assert _HYPER_BEAM in strong


def test_strength_comes_from_attack_without_fists(tmp_path):
    machines = _machines(_export(
        tmp_path, {**_BASE, "base_stats": _STRONG_STATS},
    ))
    assert _STRENGTH in machines


def test_claws_unlock_cut(tmp_path):
    machines = _machines(_export(
        tmp_path, {**_BASE, "types": ["Grass"], "traits": ["claws"]},
    ))
    assert _CUT in machines


# ---------------------------------------------------------------------------
# Species record — catch rate, exp, gender, growth, egg groups
# ---------------------------------------------------------------------------

def _blob(fields):
    return bytes.fromhex(fields["BaseStats"])


# _BASE's BST is 315 (attack 52): the 300-379 catch band, no legendary gate.
_LEGENDARY_STATS = {"hp": 100, "attack": 100, "defense": 95,
                    "sp_atk": 95, "sp_def": 95, "speed": 95}  # BST 580


def test_species_record_is_derived_not_flat(tmp_path):
    """Regression: catch 120 / exp 80 / gender 0x7F / Amorphous eggs were
    hardcoded for every species."""
    blob = _blob(_export(tmp_path, dict(_BASE)))
    assert blob[8] == 180                 # catch: 300-379 band
    assert blob[9] == 86                  # exp: round(315*0.4)-40
    assert blob[16] == 0x7F               # gendered
    assert blob[18] == 70                 # happiness
    assert blob[19] == 0                  # growth: Medium Fast (own BST < 400)
    assert (blob[20], blob[21]) == (5, 5)  # Field/Field, not Amorphous


def test_legendary_grade_record(tmp_path):
    """BST 580 is the legendary band's flat total: genderless, unbreedable,
    catch rate 3, aloof happiness."""
    blob = _blob(_export(tmp_path, {**_BASE, "base_stats": _LEGENDARY_STATS}))
    assert blob[8] == 3
    assert blob[16] == 0xFF
    assert blob[18] == 35
    assert (blob[20], blob[21]) == (15, 15)  # Undiscovered


@pytest.mark.parametrize("extra, expected", [
    ({"types": ["Water"], "traits": ["wings"]}, (2, 4)),    # Water 1 / Flying
    ({"types": ["Steel"], "traits": ["fists"]}, (10, 8)),   # Mineral / Human-Like
    ({"types": ["Ghost"]}, (11, 11)),                       # Amorphous
    ({"types": ["Fire"], "levitates": True}, (11, 11)),     # Amorphous
    ({"types": ["Dragon"]}, (14, 14)),                      # Dragon
    # Ground is not a mineral body: a Ground mammal breeds in Field like
    # Sandshrew and Phanpy, not with Geodude.
    ({"types": ["Ground"], "traits": ["claws"]}, (5, 5)),   # Field
    ({"types": ["Fairy"]}, (6, 6)),                         # export alias
])
def test_egg_groups_follow_type_and_anatomy(tmp_path, extra, expected):
    blob = _blob(_export(tmp_path, {**_BASE, **extra}))
    assert (blob[20], blob[21]) == expected


def test_stored_record_fields_win_over_derivation(tmp_path):
    fields = _export(tmp_path, {
        **_BASE, "catch_rate": 45, "gender_ratio": 254, "egg_groups": [1, 5],
    })
    blob = _blob(fields)
    assert blob[8] == 45
    assert blob[16] == 254
    assert (blob[20], blob[21]) == (1, 5)


def test_malformed_stored_record_falls_back(tmp_path):
    fields = _export(tmp_path, {
        **_BASE, "catch_rate": "many", "base_exp": -3, "egg_groups": [0, 99],
    })
    blob = _blob(fields)
    assert blob[8] == 180
    assert blob[9] == 86
    assert (blob[20], blob[21]) == (5, 5)


def test_out_of_range_growth_rate_falls_back(tmp_path):
    """Growth's byte range is 0-5, not 0-255: a stored 77 would be a broken
    exp curve, so it fails validation even though it fits the byte."""
    blob = _blob(_export(tmp_path, {**_BASE, "growth_rate": 77}))
    assert blob[19] == 0  # derived: own BST 315 < 400 → Medium Fast


# ---------------------------------------------------------------------------
# Stored derivations — stats.json as the machine contract
# ---------------------------------------------------------------------------

def test_stored_moveset_is_serialized_verbatim(tmp_path):
    fields = _export(tmp_path, {**_BASE, "moveset": [[5, 52], [1, 33]]})
    assert _moves(fields) == [(1, 33), (5, 52)]


@pytest.mark.parametrize("bad", [
    "junk", [], [[1]], [["a", "b"]], [[0, 33]], [[1, 999]], [[1, 33, 7]],
])
def test_malformed_stored_moveset_falls_back_to_derivation(tmp_path, bad):
    fields = _export(tmp_path, {**_BASE, "moveset": bad})
    assert len(_moves(fields)) == 13


def test_stored_tmhm_is_serialized_verbatim(tmp_path):
    fields = _export(tmp_path, {**_BASE, "tmhm": "00000000000000ff"})
    assert fields["TMHMCompatibility"] == "00000000000000FF"


@pytest.mark.parametrize("bad", ["xyz", "1234", 42, None])
def test_malformed_stored_tmhm_falls_back_to_derivation(tmp_path, bad):
    fields = _export(tmp_path, {**_BASE, "tmhm": bad})
    assert _TOXIC in _machines(fields)


# ---------------------------------------------------------------------------
# enrich_line — line-coherent derivation
# ---------------------------------------------------------------------------

def _stage(name, stage_no, types, stats_delta=0, traits=("fangs", "tail")):
    stats = {k: v + stats_delta for k, v in _BASE["base_stats"].items()}
    return {**_BASE, "name": name, "stage": stage_no, "types": list(types),
            "base_stats": stats, "traits": list(traits)}


def test_enrich_attaches_the_full_contract():
    stage = enrich_line([_stage("Solo", 1, ["Fire"])])[0]
    assert stage["moveset"] and stage["tmhm"]
    assert {"catch_rate", "base_exp", "gender_ratio", "egg_cycles",
            "base_happiness", "growth_rate", "egg_groups"} <= set(stage)


def test_enriched_stats_roundtrip_through_export(tmp_path):
    """The ini serializes exactly what stats.json carries — derivation
    happens once, at enrichment."""
    stage = enrich_line([_stage("Solo", 1, ["Fire"])])[0]
    fields = _export(tmp_path, stage)
    assert _moves(fields) == sorted(tuple(m) for m in stage["moveset"])
    assert fields["TMHMCompatibility"] == stage["tmhm"]
    assert _blob(fields)[8] == stage["catch_rate"]


def test_line_shares_one_growth_rate():
    """Evolving must never change a mon's exp group: one rate per line,
    from the final stage's BST (base+60 on each stat = 675, Slow)."""
    stages = enrich_line([
        _stage("Pup", 1, ["Normal"], 0),
        _stage("Adult", 2, ["Normal"], 30),
        _stage("Apex", 3, ["Normal"], 60),
    ])
    assert [s["growth_rate"] for s in stages] == [5, 5, 5]


def test_line_stages_share_one_pick_order():
    """Same line, same traits, different names: both stages draw their
    bucket picks from one shuffled order (seeded by stage 1's name), so the
    stage with less room takes a prefix of the other's picks instead of
    re-rolling — the Pupwol line's Bite-vanishes-at-stage-2 bug."""
    pup, apex = enrich_line([
        _stage("Pup", 1, ["Normal"]),
        _stage("Apex", 3, ["Normal", "Dark"]),
    ])
    bucket_ids = {mid for t in ("fangs", "tail") for _, mid in _TRAIT_MOVES[t]}
    pup_picks = {mid for _, mid in pup["moveset"]} & bucket_ids
    apex_picks = {mid for _, mid in apex["moveset"]} & bucket_ids
    assert apex_picks  # the trait floor guarantees some
    assert apex_picks <= pup_picks


def test_identical_siblings_get_identical_movesets():
    """The seed is the line, not the stage name — two stages that differ
    only in name learn the same moves."""
    a, b = enrich_line([
        _stage("Pup", 1, ["Normal"]), _stage("Wolfy", 2, ["Normal"]),
    ])
    assert a["moveset"] == b["moveset"]


def test_reenriching_rebuilds_stale_derivations():
    """Enrichment is authoritative: a stage carrying values from an earlier
    enrichment (or a hand edit) gets them re-derived, so re-enriching after
    a table change refreshes movesets and TM bits, not just the record.
    Stored-field trust belongs to export_ini alone."""
    stage = {**_stage("Solo", 1, ["Fire"]),
             "moveset": [[1, 1]], "tmhm": "00" * 8}
    enriched = enrich_line([stage])[0]
    assert enriched["moveset"] != [[1, 1]]
    assert len(enriched["moveset"]) == 13
    assert enriched["tmhm"] != "00" * 8


# ---------------------------------------------------------------------------
# Errors — unchanged propagation for genuinely missing input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["stats.json", "entry.md"])
def test_missing_input_file_raises(tmp_path, missing):
    _write_stage(tmp_path, dict(_BASE))
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        export_ini(tmp_path)


@pytest.mark.parametrize("key", ["name", "types", "base_stats"])
def test_missing_required_legacy_key_raises(tmp_path, key):
    data = {k: v for k, v in _BASE.items() if k != key}
    with pytest.raises(KeyError):
        export_ini(_write_stage(tmp_path, data))
