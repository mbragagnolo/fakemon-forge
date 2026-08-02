# Spec: Constrain generated names to the Gen 3 charset and 10-char limit with a corrective retry

## Summary

`generate_fakemon` in `fakemon_forge/generator.py` parses an LLM response into
a list of stage dicts and returns it as-is. Nothing checks that each stage's
`name` fits the Gen 3 text encoding: names longer than 10 characters or
containing characters outside the Gen 3 charset can silently reach
`writer.write_output` (directory/file names) and `export_ini._dex_number`
(MD5 hash of the name), corrupting output.

This slice adds a post-parse normalization step, `_normalize(stages, mode,
tier)`, and wires it into `generate_fakemon`'s existing 2-attempt retry loop
so that:

- A name violation on attempt 1 triggers one corrective retry (reusing
  attempt 2 of the existing budget, not a new attempt).
- A name violation still present after attempt 2 is mechanically repaired
  (truncate / strip) as a last resort.
- Valid names pass through unchanged, in a single call.

`_normalize` is introduced as a general scaffold — this slice only
implements the name rule, but the signature accepts `mode` and `tier` so
later slices (per the parent issue #48) can extend it for other fields
without changing its call site or signature.

## Inputs

- `_normalize(stages, mode, tier)`:
  - `stages`: `list[dict]`, the parsed JSON array from the LLM response
    (already `json.loads`'d), each dict containing at least a `name` key
    (string).
  - `mode`: `"single"` or `"line"` — unused by the name rule in this slice,
    threaded through for future field rules.
  - `tier`: `"standard"` / `"pseudo"` / `"legendary"` / `"mythical"` —
    unused by the name rule in this slice, threaded through for future field
    rules.
- `generate_fakemon`'s existing inputs (`description`, `mode`, `tier`,
  `client`, `api_key`) are unchanged.

## Outputs

- `_normalize` returns the same `stages` list (mutated in place or rebuilt —
  an implementation detail) with every stage's `name` guaranteed to satisfy
  the Gen 3 contract: ≤ 10 characters, every character in the allowed set.
- `generate_fakemon` returns this normalized `stages` list instead of the
  raw `json.loads(...)` result. Return type (`list[dict]`) and all other
  fields are unchanged.
- No new exceptions or exit paths are introduced by name normalization
  itself — the only `sys.exit(1)` calls remain the two pre-existing ones
  (malformed JSON after 2 attempts, Mistral API failure).

## Behavior

1. `generate_fakemon` builds the initial `messages` list as today (system +
   user prompt) and enters the existing 2-attempt loop (`for attempt in
   range(2)`).
2. On each attempt, it calls the LLM, strips fences, and `json.loads`s the
   response, exactly as today (JSON-decode failure handling is unchanged:
   retry once, then print raw response and `sys.exit(1)`).
3. After a successful parse, check every stage's `name` against the Gen 3
   contract:
   - **Length check**: `len(name) > 10`.
   - **Charset check**: any character not in the allowed set (see
     Constraints below).
4. **If this is attempt 1 (`attempt == 0`) and any stage violates either
   check**: do not return yet. Append a corrective user message to
   `messages` naming the offending stage names and the violation kind(s),
   e.g.:
   - `"These names exceed 10 characters: Flamburronix. Return the full array again with shorter names."`
   - and/or an illegal-character equivalent, e.g. `"These names contain characters that can't be used: Flam@burr. Return the full array again using only letters, numbers, spaces, and standard punctuation."`
   - If both kinds of violation are present (whether on the same stage or
     across different stages), both are communicated before the next
     attempt — there is still only one retry, covering both problems at
     once.
   Then continue the loop to attempt 2, re-calling the LLM with the
   extended `messages`.
5. **If this is attempt 2 (`attempt == 1`) and any stage still violates
   either check**: apply the mechanical repair, per stage, per violation:
   - Too long → `name = name[:10]`.
   - Illegal characters → strip every character not in the allowed set.
   - If both apply to the same name, strip illegal characters first, then
     truncate to 10 (see Assumptions for why this order).
   Do not retry again — this is the last-resort path within the existing
   2-attempt budget.
6. **If a stage's name is already valid** (on either attempt), it passes
   through untouched — no mutation, no mention in a corrective message.
7. Once a parsed response either has no violations, or has been repaired on
   attempt 2, `generate_fakemon` returns the resulting `stages` list. In
   other words, the normalization/violation check runs once per successful
   parse: on attempt 1 it decides whether to retry; on attempt 2 it always
   repairs rather than retrying again.

### Corrective message content

- The message names only the offending stage names (not all stages), so the
  model has a precise target.
- Length violations and charset violations are reported separately (each
  lists its own offenders), since they call for different corrections
  ("shorter" vs "different characters"), but both are sent in the same
  round-trip when both occur — whether as one combined message or two
  appended messages is an implementation choice (see Assumptions).
- The message is appended as a new `{"role": "user", "content": ...}`
  entry onto the existing `messages` list (matching how the conversation
  already accumulates turns), not a replacement of the original user
  prompt.

## Edge cases

- **Name is exactly 10 characters**: valid, not a length violation (`> 10`
  is the failure condition, not `>= 10`).
- **Name is empty after stripping illegal characters**: not specially
  handled — passed through as-is. Downstream behavior for empty names is
  out of scope for this slice; nothing today guards against it either.
- **Name has both a length violation and a charset violation on attempt
  1**: one combined corrective retry is sent, not two sequential retries.
- **Name has both violations and still fails on attempt 2**: both repairs
  apply, in the order strip-then-truncate (see Behavior step 5 and
  Assumptions).
- **Multiple stages violate in the same response (`mode="line"`)**: all
  offending names across all stages are collected for the corrective
  message (or repaired individually on attempt 2) — the check runs over
  every stage in the list, not just stage 1.
- **Attempt 1 has a name violation but attempt 2's JSON is malformed**: the
  existing JSON-decode-failure path takes precedence — malformed JSON on
  attempt 2 still hits the existing "malformed JSON after 2 attempts"
  `sys.exit(1)`, since normalization only runs after a successful parse.
- **A name is valid on attempt 1 but another stage's name is bad
  (`mode="line"`)**: the valid name is left untouched; only violating
  stages are named in the corrective message and only violating names are
  repaired on attempt 2.

## Errors

- No new error paths. Name violations never raise or `sys.exit` — they are
  corrected via retry or mechanically repaired; truncation/stripping is the
  floor applied in code, not surfaced as a user-facing error.
- Pre-existing error paths (malformed JSON after 2 attempts, Mistral API
  exception) are untouched, including their messages and `sys.exit(1)`
  behavior.

## Constraints & dependencies

- **Gen 3 allowed charset** (exact set, per the issue):
  `A–Z`, `a–z`, `0–9`, space, `é`, `♂`, `♀`, and the punctuation marks
  `.` `,` `'` `-` `…` `!` `?` `/` `(` `)` `"` `:` `;`.
  Newline is explicitly never allowed — it is simply not in the allowed
  set, so no separate code path is needed for it.
- **Max length**: 10 characters, checked with plain `len()` (character
  count), consistent with Python string semantics used throughout this
  codebase.
- The retry budget stays at 2 attempts total — this slice adds no new
  attempts; it repurposes attempt 2 of the existing loop for corrective
  retries when needed, and attempt 2 remains the final attempt whether the
  retry was triggered by malformed JSON or by a name violation.
- `_normalize` must not require network access or an LLM call itself — it
  is a pure post-processing function operating on already-parsed data, with
  no side effects beyond mutating/returning the stages list.
- No new third-party dependencies. The charset check can be implemented
  with a plain Python string/set membership test.
- Must not touch ML code paths (`sprites.py`, `test_sprites_ml.py`) or add
  any `@pytest.mark.ml` tests, per `CLAUDE.md`.

## Assumptions

- **Strip-then-truncate order for last-resort repair**: when a name fails
  both checks on attempt 2, illegal characters are stripped first and the
  result is then truncated to 10 characters. The issue doesn't specify
  order for the combined case; stripping first ensures the final ≤10
  characters are exactly the ones that will appear in the encoded output,
  rather than truncating first and potentially cutting valid characters
  that come after an illegal one within the first 10.
- **`_normalize`'s role in the loop**: `_normalize` (or the shared
  length/charset predicates it uses internally) is consulted on both
  attempts, but behaves differently by attempt: on attempt 1 a violation
  triggers a retry instead of a repair; on attempt 2 a violation is always
  repaired. This spec doesn't mandate one specific internal code shape
  (e.g., a shared `_violations(stages)` helper reused by both the
  attempt-1 check and attempt-2's repair pass vs. two call sites), only
  that attempt 1 must not silently repair instead of retrying.
- **Corrective message wording is illustrative, not contractual**: the
  issue gives example phrasing ("These names exceed 10 characters: ...").
  Tests should check that offending names appear in the appended
  message(s) and that a second API call happens, not match exact string
  wording — this keeps phrasing free to read naturally.
- **One combined message vs. two appended messages** when both violation
  kinds occur in the same round-trip: left as an implementation choice,
  since the issue says "and/or" and the observable contract is "one retry
  covers both." Tests should assert on message content (offender names
  present) and call count, not on the exact number of appended messages.
- **`_normalize`'s mutation style**: assumed to mutate the `name` key on
  each stage dict in place or return equivalent new dicts — either is
  acceptable since `generate_fakemon` only uses `_normalize`'s return
  value; not constrained further.
- **This slice's scope is name-only**: per the issue, `_normalize` is a
  scaffold that later slices of parent #48 extend for other fields (e.g.,
  height/weight defaults). This spec covers only the name rule; `mode`/
  `tier` are accepted but unused parameters for now.
- **Uniqueness / thematic consistency across a line's stage names**: not
  addressed — the hard contract is purely about individual name length and
  charset, not about names differing between stages or matching a
  thematic throughline (that remains a prompt-quality concern already
  covered by `_SYSTEM_PROMPT`, not something enforced in code).
- **Testing scope** matches the issue's own list exactly: too-long
  triggers retry (2 calls) then accepts a valid second response; illegal
  chars trigger the same path; both-at-once; last-resort repair when
  attempt 2 still violates; a valid name passes through in one call. No
  additional scenarios (e.g. empty-name handling) are added beyond what's
  listed, per "smallest coherent slice."
