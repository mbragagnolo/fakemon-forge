import json
import pytest

from fakemon_forge.export_ini import export_ini, _resolve_type, _type_index
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
