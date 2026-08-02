import json

from fakemon_forge.export_ini import export_ini

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


def _write_stage(tmp_path, data, entry=_ENTRY):
    (tmp_path / "stats.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "entry.md").write_text(entry, encoding="utf-8")
    return tmp_path


def _read_ini(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def test_new_format_round_trip(tmp_path):
    data = {
        **_BASE,
        "height_dm": 7,
        "weight_hg": 120,
        "abilities_gen3": ["Blaze", "Sand Veil"],
        "category": "Flame",
    }
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["Hght"] == "7"
    assert fields["Wght"] == "120"
    assert fields["PokedexType"] == "Flame"

    base_stats = fields["BaseStats"]
    ability1 = base_stats[44:46]
    ability2 = base_stats[46:48]
    assert ability1 == f"{66:02X}"  # Blaze
    assert ability2 == f"{8:02X}"   # Sand Veil


def test_legacy_fallback(tmp_path):
    stage_dir = _write_stage(tmp_path, dict(_BASE))
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["Hght"] == "5"
    assert fields["Wght"] == "30"
    assert " POKEMON" not in fields["PokedexType"]

    base_stats = fields["BaseStats"]
    ability2 = base_stats[46:48]
    assert ability2 == "00"


def test_height_and_weight_independent_fallback(tmp_path):
    data = {**_BASE, "height_dm": 0}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["Hght"] == "0"
    assert fields["Wght"] == "30"


def test_abilities_gen3_single_entry(tmp_path):
    data = {**_BASE, "abilities_gen3": ["Blaze"]}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    base_stats = fields["BaseStats"]
    ability1 = base_stats[44:46]
    ability2 = base_stats[46:48]
    assert ability1 == f"{66:02X}"
    assert ability2 == "00"


def test_abilities_gen3_empty_falls_back_to_ability(tmp_path):
    data = {**_BASE, "abilities_gen3": []}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    base_stats = fields["BaseStats"]
    ability1 = base_stats[44:46]
    ability2 = base_stats[46:48]
    assert ability1 == f"{66:02X}"  # from data["ability"] == "Blaze"
    assert ability2 == "00"


def test_pokedex_type_from_category(tmp_path):
    data = {**_BASE, "category": "Flame"}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["PokedexType"] == "Flame"
    assert " POKEMON" not in fields["PokedexType"]


def test_pokedex_type_falls_back_to_type_word(tmp_path):
    data = {**_BASE, "category": ""}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["PokedexType"] == "FIRE"
    assert " POKEMON" not in fields["PokedexType"]


def test_pokedex_type_non_string_category_falls_back(tmp_path):
    data = {**_BASE, "category": 42}
    stage_dir = _write_stage(tmp_path, data)
    out = export_ini(stage_dir)
    fields = _read_ini(out)

    assert fields["PokedexType"] == "FIRE"
    assert " POKEMON" not in fields["PokedexType"]
