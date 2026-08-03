"""Conformance of `_BST_TARGETS` to the observed Gen 3 bands (issue #59).

The bands in ``tests/fixtures/gen3_bst_bands.json`` are aggregate statistics
derived offline from real game data. These tests are what make the constants
"hardened" rather than merely edited: every target must be its band's median,
so any value can be re-derived and verified without knowing which were once
hand-picked.

Nothing in ``fakemon_forge`` reads the fixture — the package gains no runtime
file I/O. Only these tests do.
"""
import json
from pathlib import Path

import pytest

from fakemon_forge.generator import _BST_TARGETS

_FIXTURE = Path(__file__).parent / "fixtures" / "gen3_bst_bands.json"


def _bands() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _targets(tier: str, stage_count: int) -> tuple[int, ...]:
    """The per-stage targets for one (tier, stage count), as a flat tuple.

    Written against the frozen contract in tasks/00: `_BST_TARGETS` is keyed
    tier -> stage count -> per-stage targets.
    """
    return tuple(_BST_TARGETS[tier][stage_count])


# ---------------------------------------------------------------------------
# The lookups the contract must support
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier, stage_count, expected", [
    ("standard", 1, (430,)),
    ("standard", 2, (305, 468)),
    ("standard", 3, (295, 405, 518)),
    ("pseudo", 3, (300, 420, 600)),
    ("legendary", 1, (580,)),
    ("mythical", 1, (600,)),
])
def test_supported_lookups(tier, stage_count, expected):
    assert _targets(tier, stage_count) == expected


def test_pseudo_has_no_two_stage_row():
    """A pseudo-legendary line is always 3 stages; the CLI rejects the combo,
    so the table must not quietly offer it either."""
    assert 2 not in _BST_TARGETS["pseudo"]


@pytest.mark.parametrize("tier", ["legendary", "mythical"])
def test_single_form_tiers_have_only_a_single_row(tier):
    assert set(_BST_TARGETS[tier]) == {1}


# ---------------------------------------------------------------------------
# Band conformance -- the actual hardening
# ---------------------------------------------------------------------------

# (tier, stage count) -> the fixture bucket each stage index maps to.
_BAND_OF = {
    ("standard", 1): ["standalone"],
    ("standard", 2): [("2", "0"), ("2", "1")],
    ("standard", 3): [("3", "0"), ("3", "1"), ("3", "2")],
    ("legendary", 1): ["legendary"],
    ("mythical", 1): ["mythical"],
}


def _band(bands: dict, key):
    return bands[key] if isinstance(key, str) else bands[key[0]][key[1]]


@pytest.mark.parametrize("lookup", list(_BAND_OF))
def test_every_target_equals_its_band_median(lookup):
    """The rule the table follows. Stricter than the in-band check below, and
    the one that catches a value hand-edited to something still inside a band."""
    bands = _bands()
    for target, band_key in zip(_targets(*lookup), _BAND_OF[lookup]):
        assert target == _band(bands, band_key)["median"]


@pytest.mark.parametrize("lookup", list(_BAND_OF))
def test_every_target_lies_inside_its_band(lookup):
    bands = _bands()
    for target, band_key in zip(_targets(*lookup), _BAND_OF[lookup]):
        band = _band(bands, band_key)
        assert band["p10"] <= target <= band["p90"]


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 2), ("standard", 3), ("pseudo", 3),
])
def test_targets_increase_monotonically_across_a_line(tier, stage_count):
    targets = _targets(tier, stage_count)
    assert list(targets) == sorted(targets)
    assert len(set(targets)) == len(targets)


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 1), ("standard", 2), ("standard", 3),
    ("pseudo", 3), ("legendary", 1), ("mythical", 1),
])
def test_target_count_matches_the_stage_count(tier, stage_count):
    assert len(_targets(tier, stage_count)) == stage_count


# ---------------------------------------------------------------------------
# The placeholder-filter check
# ---------------------------------------------------------------------------

def test_high_band_totals_exactly_21():
    """The real count of Gen 3 legendaries and mythicals.

    This is the placeholder-filter assertion: the species table contains unused
    slots that all share one high total, and a fixture regenerated without
    excluding them reads 46 here. Failing loudly beats silently shifting every
    band -- unfiltered, the standalone median reads 502 instead of 430.

    All three high buckets count. ``box_legendary`` has no tier pointing at it
    (a fifth tier was declined), but it is part of the band and omitting it
    would make this check pass at 15 while quietly under-representing the data.
    """
    bands = _bands()
    total = sum(bands[k]["n"] for k in ("legendary", "mythical", "box_legendary"))
    assert total == 21


def test_single_member_buckets_account_for_every_line():
    """standalone + the high band == every single-member line observed."""
    bands = _bands()
    high = sum(bands[k]["n"] for k in ("legendary", "mythical", "box_legendary"))
    assert bands["standalone"]["n"] + high == 75


def test_box_legendary_band_sits_above_every_tier_target():
    """Records why no tier targets it: the band is strictly above the highest
    value any tier prompts for, which is what the declined fifth tier covered."""
    bands = _bands()
    highest_tier_target = max(
        max(targets) for rows in _BST_TARGETS.values() for targets in rows.values()
    )
    assert bands["box_legendary"]["p10"] > highest_tier_target


# ---------------------------------------------------------------------------
# The two consistency properties the corrected values exist to establish
# ---------------------------------------------------------------------------

def test_standalone_target_exceeds_a_three_stage_juvenile():
    """Issue #48 settled that a single form is a standalone species, not a
    juvenile, for height/weight. This pins the same reading for BST -- it is
    the reason `single` moved off the stage-1 value."""
    assert _targets("standard", 1)[0] > _targets("standard", 3)[0]


def test_two_stage_final_exceeds_the_three_stage_middle():
    """A 2-stage stage 2 is a final form, not a middle one."""
    assert _targets("standard", 2)[-1] > _targets("standard", 3)[1]


def test_two_stage_final_is_below_the_three_stage_final():
    """...but still a shorter line's peak, so it sits under a 3-stage final."""
    assert _targets("standard", 2)[-1] < _targets("standard", 3)[-1]


# ---------------------------------------------------------------------------
# The fixture must never carry game data
# ---------------------------------------------------------------------------

def test_fixture_holds_only_numbers():
    """No species names, IDs, ROM offsets or slot ranges may reach this repo.

    Every leaf is an int and every key is a digit string or a known bucket
    name, so there is nowhere for such data to hide.
    """
    bands = _bands()
    allowed_stats = {"median", "p10", "p90", "n"}
    for key, value in bands.items():
        if key.startswith("_"):
            assert isinstance(value, str)
            continue
        assert key in {
            "2", "3", "standalone", "legendary", "mythical", "box_legendary",
        }
        buckets = value.values() if key in {"2", "3"} else [value]
        for bucket in buckets:
            assert set(bucket) == allowed_stats
            assert all(isinstance(v, int) for v in bucket.values())
