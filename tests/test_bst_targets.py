"""Conformance of `_BST_TARGETS` to the observed Gen 3 bands (issues #59, #85).

The bands in ``tests/fixtures/gen3_bst_bands.json`` are aggregate statistics
derived offline from real game data. These tests are what make the constants
"hardened" rather than merely edited: every entry must be its band's
``(p10, median, p90)``, so any value can be re-derived and verified without
knowing which were once hand-picked.

#59 put the median in the prompt. #85 established that the prompt number was
never load-bearing and made the p10..p90 span the range an enforced total is
picked inside, so the table now carries all three numbers and the tests check
all three.

Nothing in ``fakemon_forge`` reads the fixture — the package gains no runtime
file I/O. Only these tests do.
"""
import json
from pathlib import Path

import pytest

from fakemon_forge.generator import (
    _BST_TARGETS,
    _HASH_SPACE,
    _MEDIAN,
    _P10,
    _P90,
    _bst_target,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gen3_bst_bands.json"


def _bands() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _rows(tier: str, stage_count: int) -> tuple[tuple[int, int, int], ...]:
    """The per-stage bands for one (tier, stage count), as a flat tuple.

    `_BST_TARGETS` is keyed tier -> stage count -> per-stage
    ``(p10, median, p90)`` triples.
    """
    return tuple(tuple(band) for band in _BST_TARGETS[tier][stage_count])


def _targets(tier: str, stage_count: int) -> tuple[int, ...]:
    """Just the medians — the numbers the prompt names."""
    return tuple(band[_MEDIAN] for band in _rows(tier, stage_count))


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
    """The medians #59 shipped, unchanged. Widening the table to triples must
    not move the number the prompt asks the model for."""
    assert _targets(tier, stage_count) == expected


@pytest.mark.parametrize("tier, stage_count, expected", [
    ("standard", 1, ((336, 430, 500),)),
    ("standard", 2, ((240, 305, 360), (410, 468, 515))),
    ("standard", 3, ((205, 295, 314), (278, 405, 420), (450, 518, 600))),
    ("pseudo", 3, ((300, 300, 300), (410, 420, 420), (600, 600, 600))),
    ("legendary", 1, ((580, 580, 580),)),
    ("mythical", 1, ((600, 600, 600),)),
])
def test_supported_band_lookups(tier, stage_count, expected):
    assert _rows(tier, stage_count) == expected


def test_pseudo_has_no_two_stage_row():
    """A pseudo-legendary line is always 3 stages; the CLI rejects the combo,
    so the table must not quietly offer it either."""
    assert 2 not in _BST_TARGETS["pseudo"]


@pytest.mark.parametrize("tier", ["legendary", "mythical"])
def test_single_form_tiers_have_only_a_single_row(tier):
    assert set(_BST_TARGETS[tier]) == {1}


def test_every_band_is_a_p10_median_p90_triple():
    """The shape the whole module depends on. A row left as a bare int would
    make `_bst_target` unpack a number and `_user_prompt` index into one."""
    for rows in _BST_TARGETS.values():
        for row in rows.values():
            for band in row:
                assert len(band) == 3
                assert all(isinstance(v, int) for v in band)


# ---------------------------------------------------------------------------
# Band conformance -- the actual hardening
# ---------------------------------------------------------------------------

# (tier, stage count) -> the fixture bucket each stage index maps to.
# Every tier is covered: `pseudo` was hand-picked in #59 and re-derived from the
# source data in #85, so there is no longer a row that answers to nothing.
_BAND_OF = {
    ("standard", 1): ["standalone"],
    ("standard", 2): [("2", "0"), ("2", "1")],
    ("standard", 3): [("3", "0"), ("3", "1"), ("3", "2")],
    ("pseudo", 3): [("pseudo", "0"), ("pseudo", "1"), ("pseudo", "2")],
    ("legendary", 1): ["legendary"],
    ("mythical", 1): ["mythical"],
}


def test_every_tier_row_is_checked_against_the_fixture():
    """No unverifiable rows. A tier or stage count added to `_BST_TARGETS`
    without a matching bucket would otherwise be hand-picked forever, which is
    exactly the state `pseudo` was in before #85."""
    covered = {(tier, count) for tier, count in _BAND_OF}
    declared = {
        (tier, count) for tier, rows in _BST_TARGETS.items() for count in rows
    }
    assert declared == covered


def _band(bands: dict, key):
    return bands[key] if isinstance(key, str) else bands[key[0]][key[1]]


@pytest.mark.parametrize("lookup", list(_BAND_OF))
def test_every_band_equals_its_fixture_band(lookup):
    """The rule the table follows, on all three numbers. Catches a value
    hand-edited to something still inside its band — the failure mode the
    median-only version of this test was written for, now covering the p10 and
    p90 that `_bst_target` actually reads."""
    bands = _bands()
    for band, band_key in zip(_rows(*lookup), _BAND_OF[lookup]):
        observed = _band(bands, band_key)
        assert band[_P10] == observed["p10"]
        assert band[_MEDIAN] == observed["median"]
        assert band[_P90] == observed["p90"]


@pytest.mark.parametrize("tier", list(_BST_TARGETS))
def test_every_band_is_ordered(tier):
    """p10 <= median <= p90 for every row, including the hand-picked pseudo
    one. A band with its ends crossed would make `_bst_target` return the p10
    for every position and silently reinstate a flat target."""
    for row in _BST_TARGETS[tier].values():
        for band in row:
            assert band[_P10] <= band[_MEDIAN] <= band[_P90]


def test_pseudo_outer_stages_are_flat_and_the_middle_is_not():
    """The shape of the observed pseudo-legendary population, pinned exactly.

    All four lines open at 300 and finish at 600 — there is genuinely no spread
    to pick inside, so those stages resolve to one number for every name. The
    middle stage is the sole exception: one of the four sits 10 under the other
    three, which is where the 410 floor comes from. Asserting the asymmetry
    rather than just "flat" is what would catch the middle row being widened to
    match a band it doesn't have.
    """
    opening, middle, final = _rows("pseudo", 3)
    assert opening[_P10] == opening[_MEDIAN] == opening[_P90] == 300
    assert final[_P10] == final[_MEDIAN] == final[_P90] == 600
    assert middle == (410, 420, 420)


def test_pseudo_is_a_subset_of_the_three_stage_bucket():
    """`pseudo` is the one bucket that is not its own population — its lines are
    also counted in "3". Recorded because the n values otherwise look like they
    should sum with the others, and they don't."""
    bands = _bands()
    assert bands["pseudo"]["0"]["n"] == 4
    assert bands["pseudo"]["0"]["n"] < bands["3"]["0"]["n"]


@pytest.mark.parametrize("stage_index", ["0", "1", "2"])
def test_every_pseudo_stage_sits_inside_the_three_stage_band(stage_index):
    """The containment that makes the subset claim checkable: a pseudo stage
    cannot fall outside the wider bucket it is drawn from."""
    bands = _bands()
    pseudo = bands["pseudo"][stage_index]
    wider = bands["3"][stage_index]
    assert wider["p10"] <= pseudo["p10"] <= wider["p90"]
    assert wider["p10"] <= pseudo["p90"] <= wider["p90"]


@pytest.mark.parametrize("tier", ["legendary", "mythical"])
def test_flat_fixture_bands_stay_flat(tier):
    """legendary and mythical are flat in the observed data itself — every Gen 3
    member shares one total. The table must not widen them."""
    bands = _bands()
    assert bands[tier]["p10"] == bands[tier]["median"] == bands[tier]["p90"]
    for band in _rows(tier, 1):
        assert band[_P10] == band[_MEDIAN] == band[_P90]


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 2), ("standard", 3), ("pseudo", 3),
])
def test_targets_increase_monotonically_across_a_line(tier, stage_count):
    targets = _targets(tier, stage_count)
    assert list(targets) == sorted(targets)
    assert len(set(targets)) == len(targets)


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 2), ("standard", 3), ("pseudo", 3),
])
@pytest.mark.parametrize("edge", [_P10, _P90])
def test_band_edges_increase_monotonically_across_a_line(tier, stage_count, edge):
    """The precondition that makes band-picking safe.

    `_bst_target` places every stage at the same quantile of its own band. That
    only yields an ascending line if both ends of the band ascend — if a stage
    2's p10 sat below its stage 1's p10, a low-quantile line would go backwards.
    """
    edges = [band[edge] for band in _rows(tier, stage_count)]
    assert edges == sorted(edges)


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 1), ("standard", 2), ("standard", 3),
    ("pseudo", 3), ("legendary", 1), ("mythical", 1),
])
def test_target_count_matches_the_stage_count(tier, stage_count):
    assert len(_rows(tier, stage_count)) == stage_count


# ---------------------------------------------------------------------------
# The property band-picking exists to preserve
# ---------------------------------------------------------------------------

# Endpoints plus a sweep across the interior. The endpoints are where an
# off-by-one in `_bst_target`'s integer division would show up.
_POSITIONS = [0, 1, _HASH_SPACE - 2, _HASH_SPACE - 1] + [
    i * (_HASH_SPACE - 1) // 32 for i in range(33)
]


@pytest.mark.parametrize("tier, stage_count", [
    ("standard", 2), ("standard", 3), ("pseudo", 3),
])
@pytest.mark.parametrize("position", _POSITIONS)
def test_picked_totals_ascend_across_a_line(tier, stage_count, position):
    """Whatever quantile a line lands on, its stages get ascending totals.

    This is the guarantee #85 called out as absent before: nothing stopped the
    model handing back a stage 2 weaker than its stage 1. `pseudo` is flat, so
    it holds there by construction — parametrized anyway, because a future row
    given a real band would silently drop out of coverage otherwise.
    """
    picked = [_bst_target(band, position) for band in _rows(tier, stage_count)]
    assert picked == sorted(picked)
    assert len(set(picked)) == len(picked)


@pytest.mark.parametrize("lookup", list(_BAND_OF) + [("pseudo", 3)])
@pytest.mark.parametrize("position", _POSITIONS)
def test_picked_totals_stay_inside_their_band(lookup, position):
    for band in _rows(*lookup):
        assert band[_P10] <= _bst_target(band, position) <= band[_P90]


@pytest.mark.parametrize("lookup", list(_BAND_OF) + [("pseudo", 3)])
def test_band_endpoints_are_reachable(lookup):
    """Both ends of every band can actually come out, so the scatter spans the
    full observed range rather than an interior slice of it."""
    for band in _rows(*lookup):
        assert _bst_target(band, 0) == band[_P10]
        assert _bst_target(band, _HASH_SPACE - 1) == band[_P90]


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
    value any tier can now *produce* — not just prompt for. Band-picking raised
    that ceiling from the largest median (600) to the largest p90, so this is
    the check that catches a future band widened up into box-legendary range.
    """
    bands = _bands()
    highest_reachable = max(
        band[_P90]
        for rows in _BST_TARGETS.values()
        for row in rows.values()
        for band in row
    )
    assert bands["box_legendary"]["p10"] > highest_reachable


# ---------------------------------------------------------------------------
# The two consistency properties the corrected values exist to establish
# ---------------------------------------------------------------------------

def test_standalone_target_exceeds_a_three_stage_juvenile():
    """Issue #48 settled that a single form is a standalone species, not a
    juvenile, for height/weight. This pins the same reading for BST -- it is
    the reason `single` moved off the stage-1 value."""
    assert _targets("standard", 1)[0] > _targets("standard", 3)[0]


def test_standalone_band_outranks_a_three_stage_juvenile_at_every_position():
    """...and the same reading survives band-picking. Medians alone would let a
    low-quantile standalone (336) land under a high-quantile juvenile (314) if
    the bands ever overlapped; today they don't, and this is what says so."""
    standalone = _rows("standard", 1)[0]
    juvenile = _rows("standard", 3)[0]
    assert standalone[_P10] > juvenile[_P90]


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
    nested = {"2", "3", "pseudo"}
    for key, value in bands.items():
        if key.startswith("_"):
            assert isinstance(value, str)
            continue
        assert key in nested | {
            "standalone", "legendary", "mythical", "box_legendary",
        }
        buckets = value.values() if key in nested else [value]
        for bucket in buckets:
            assert set(bucket) == allowed_stats
            assert all(isinstance(v, int) for v in bucket.values())
