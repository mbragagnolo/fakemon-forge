# Prototype findings: what are the observed Gen 3 BST bands?

> Tracking: #59 — spike for `bst-hardening-spec.md`

## Question

Three clustered unknowns from the spec's *Open questions / assumptions*:

1. What are the observed BST bands per (line length, stage index)?
2. Is the **median** the right target value, or would the mean do just as well?
3. Does the single-member bucket **split cleanly** into ordinary no-evolution
   species vs legendary/mythical by a BST threshold?

## What was tried

A throwaway script (kept out of this repo, per the spec — it imports the private
injector and reads a local-only game image) that:

- derives every vanilla evolution line and its per-species base-stat totals,
- buckets **linear** lines by `(line length, stage index)`, skipping branched
  ones,
- reports `n / min / p10 / median / mean / p90 / max` per bucket,
- dumps the sorted single-member totals and their largest gaps, then tries
  several candidate split thresholds.

Ten derived values were cross-checked against the public Serebii RS Pokédex
stat listing. **All ten matched exactly.**

## Result

### The bands

| bucket | n | p10 | **median** | p90 | min | max |
|---|---|---|---|---|---|---|
| 2-stage · stage 1 | 84 | 240 | **305** | 360 | 180 | 500 |
| 2-stage · stage 2 | 84 | 410 | **468** | 515 | 330 | 555 |
| 3-stage · stage 1 | 37 | 205 | **295** | 314 | 190 | 330 |
| 3-stage · stage 2 | 37 | 278 | **405** | 420 | 205 | 455 |
| 3-stage · stage 3 | 37 | 450 | **518** | 600 | 385 | 670 |
| standalone | 54 | 336 | **430** | 500 | 250 | 540 |
| legendary band | 9 | — | **580** | — | 580 | 580 |
| mythical band | 6 | — | **600** | — | 600 | 600 |

### Q2 — medians confirmed, and it materially matters

For the 3-stage buckets the mean is dragged well below the median by a long
low tail (very weak mid-stage forms):

| bucket | median | mean | gap |
|---|---|---|---|
| 3-stage · stage 2 | 405 | 371.5 | **33.5** |
| 3-stage · stage 1 | 295 | 274.6 | **20.4** |
| 3-stage · stage 3 | 518 | 510.0 | 8.0 |

A mean-based target would have prompted a mid-stage roughly 34 points weaker
than typical. For the 2-stage buckets the two agree closely (305 vs 305.5,
468 vs 465.5), so the choice only bites on 3-stage lines — but there it bites.

### Q3 — it splits cleanly, but not the way the spec implied

Two corrections, one of them serious.

**(a) Placeholder slots must be filtered first — this is the big one.**
The species table contains **25 unused placeholder slots that all carry BST
600**. Unfiltered they masquerade as legendary-tier single-member lines and
corrupt two buckets badly:

| | unfiltered | filtered | |
|---|---|---|---|
| single-member lines | 100 | 75 | |
| standalone median | **502** | **430** | 72 points off |
| high-band members | **46** | **21** | more than double |

The filtered count of 21 is *exactly* the real number of Gen 3 legendaries and
mythicals, and the cluster breakdown (9 · 580, 6 · 600, 2 · 670, 4 · 680) matches
known values species-for-species. That is the check that proves the filter right.

**A "name fails to decode" filter is wrong** — one genuine legendary also fails
the strict name decode and would be dropped. The filter must be the documented
unused-slot ID range (it lives in the throwaway script, deliberately not here).

**(b) "Largest gap" picks the wrong threshold.** The biggest gap in the sorted
single-member totals is 600 → 670 (70 points) — which sits *inside* the
legendary cluster, not at its boundary. The real boundary is the
**second**-largest gap, 540 → 580 (40 points). Any threshold in `(540, 580]`
yields the same clean 54 / 21 partition; **560** is the natural pick.

### Bonus: the existing table is mostly already right

The spec framed this as re-deriving every value. It isn't — most are sound:

| target | current | observed | verdict |
|---|---|---|---|
| single (standard) | 300 | **430** | **wrong — the real fix** |
| 3-stage · stage 2 | 420 | **405** | at the very top edge (p90 = 420); bring down |
| 3-stage · stage 1 | 300 | 295 | fine, inside band |
| 3-stage · stage 3 | 520 | 518 | fine, essentially exact |
| pseudo 300 / 420 / 600 | — | — | **confirmed**; the 600 final sits at p90 of the 3-stage final band, i.e. genuinely "rivals legendaries" |
| legendary | 580 | 580 | **confirmed** — modal value, 9 species |
| mythical | 600 | 600 | **confirmed** — 6 species, all four real mythicals exactly 600 |

So "hardening" is mostly **validation**. Only two numbers actually move, plus
one new row.

### Also observed

8 branched lines exist (member counts 3, 3, 3, 4, 4, 4, 5, 6) against 222
linear ones — confirming branched evolution is a real, recurring shape and worth
the deferred companion issue, not a one-off curiosity.

## Recommendation

Changes to `bst-hardening-spec.md`:

1. **Behavior §1 (derivation)** — add a mandatory step: exclude the unused
   placeholder slots *before* bucketing, and assert the resulting high-band
   count is 21 as the correctness check. The spec currently omits this entirely,
   and without it the standalone and legendary bands are both wrong.
2. **Behavior §1 step 5** — replace "split point is an implementation detail to
   be reported" with the answer: **560** (any value in `(540, 580]`). Explicitly
   warn against a largest-gap heuristic, which picks 670 and is wrong.
3. **§3 `_BST_TARGETS`** — fill in the values: single 430; 2-stage 305 / 468;
   3-stage 295 / 405 / 518; pseudo, legendary and mythical unchanged. Reframe
   the section as "correct two values and add the 2-stage row", not a rewrite.
4. **Open questions** — resolve the medians assumption to **[confirmed]**, with
   the 33.5-point stage-2 gap as the evidence.
5. **Testing** — the band fixture should carry `n` per bucket (already
   specified) *and* the assertion that the high band contains exactly 21
   members, since that is what proves the placeholder filter ran.
