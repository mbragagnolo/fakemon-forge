import json
import random
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fakemon_forge.generator import (
    generate_fakemon,
    _normalize,
    _corrective_message,
    _size_defaults,
    _user_prompt,
    _ALLOWED_NAME_CHARS,
    _SYSTEM_PROMPT,
    _ABILITY_POOL,
    _TYPE_POOL,
    _normalize_types,
    _HASH_SPACE,
    _MEDIAN,
    _STAT_KEYS,
    _STAT_MAX,
    _STAT_MIN,
    _apportion,
    _bst_target,
    _name_position,
    _normalize_base_stats,
    _stage_band,
    _stat_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STAGE_1 = {
    "name": "Flamburr",
    "stage": 1,
    "types": ["Fire"],
    "ability": "Blaze",
    "base_stats": {
        "hp": 45, "attack": 52, "defense": 43,
        "sp_atk": 60, "sp_def": 50, "speed": 50,
    },
    "pokedex_entry": "A small fiery creature with a burning tail.",
    "sprite_prompt": "A small fire lizard, GBA pixel art style, white background",
}

_STAGE_2 = {**_STAGE_1, "name": "Flamburro", "stage": 2,
            "base_stats": {**_STAGE_1["base_stats"], "hp": 65, "attack": 72, "speed": 70}}

_STAGE_3 = {**_STAGE_1, "name": "Flamburron", "stage": 3,
            "base_stats": {**_STAGE_1["base_stats"], "hp": 85, "attack": 92, "speed": 90}}

_LINE = [_STAGE_1, _STAGE_2, _STAGE_3]


def _make_client(*responses):
    """Return a mock client that yields each JSON string in sequence."""
    side_effects = []
    for content in responses:
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        side_effects.append(resp)

    client = MagicMock()
    if len(side_effects) == 1:
        client.chat.complete.return_value = side_effects[0]
    else:
        client.chat.complete.side_effect = side_effects
    return client


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_single_mode_returns_list_of_one():
    client = _make_client(json.dumps([_STAGE_1]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert len(result) == 1


def test_line_mode_returns_list_of_three():
    client = _make_client(json.dumps(_LINE))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert len(result) == 3


def test_stage_has_all_required_fields():
    client = _make_client(json.dumps([_STAGE_1]))
    result = generate_fakemon("fire lizard", "single", client=client)
    stage = result[0]
    for field in ("name", "stage", "types", "ability", "base_stats",
                  "pokedex_entry", "sprite_prompt"):
        assert field in stage, f"missing field: {field}"


def test_base_stats_has_six_keys():
    client = _make_client(json.dumps([_STAGE_1]))
    result = generate_fakemon("fire lizard", "single", client=client)
    stats = result[0]["base_stats"]
    assert set(stats.keys()) == {"hp", "attack", "defense", "sp_atk", "sp_def", "speed"}


def test_returns_parsed_data_not_string():
    client = _make_client(json.dumps([_STAGE_1]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert isinstance(result[0]["types"], list)
    assert isinstance(result[0]["base_stats"], dict)


# ---------------------------------------------------------------------------
# Model and prompt
# ---------------------------------------------------------------------------

def test_calls_correct_model():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_args.kwargs["model"] == "mistral-large-latest"


def test_description_included_in_prompt():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("spiky ice wolf with blue fur", "single", client=client)
    messages = client.chat.complete.call_args.kwargs["messages"]
    full_text = " ".join(m["content"] for m in messages)
    assert "spiky ice wolf with blue fur" in full_text


def test_single_mode_prompt_mentions_one_stage():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", client=client)
    messages = client.chat.complete.call_args.kwargs["messages"]
    full_text = " ".join(m["content"] for m in messages)
    assert "1" in full_text or "one" in full_text.lower() or "single" in full_text.lower()


def test_line_mode_prompt_mentions_three_stages():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    messages = client.chat.complete.call_args.kwargs["messages"]
    full_text = " ".join(m["content"] for m in messages)
    assert "3" in full_text or "three" in full_text.lower()


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

def test_strips_markdown_code_fence():
    fenced = "```json\n" + json.dumps([_STAGE_1]) + "\n```"
    client = _make_client(fenced)
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["name"] == "Flamburr"


def test_strips_plain_code_fence():
    fenced = "```\n" + json.dumps([_STAGE_1]) + "\n```"
    client = _make_client(fenced)
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["name"] == "Flamburr"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def test_retries_once_on_malformed_json():
    bad = "not valid json {{{"
    good = json.dumps([_STAGE_1])
    client = _make_client(bad, good)
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburr"


def test_exits_after_two_malformed_responses(capsys):
    client = _make_client("garbage", "still garbage")
    with pytest.raises(SystemExit) as exc:
        generate_fakemon("fire lizard", "single", client=client)
    assert exc.value.code == 1


def test_prints_raw_response_on_double_failure(capsys):
    client = _make_client("garbage output", "garbage output 2")
    with pytest.raises(SystemExit):
        generate_fakemon("fire lizard", "single", client=client)
    err = capsys.readouterr().err
    assert "garbage output 2" in err


# ---------------------------------------------------------------------------
# Name normalization (Gen 3 charset + 10-char limit)
# ---------------------------------------------------------------------------

def test_valid_name_passes_through_in_one_call():
    client = _make_client(json.dumps([_STAGE_1]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 1
    assert result[0]["name"] == "Flamburr"


def test_too_long_name_triggers_corrective_retry():
    too_long = {**_STAGE_1, "name": "Flamburronix"}
    good = {**_STAGE_1, "name": "Flamburron"}
    client = _make_client(json.dumps([too_long]), json.dumps([good]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburron"
    second_call_messages = client.chat.complete.call_args.kwargs["messages"]
    corrective = second_call_messages[-1]
    assert corrective["role"] == "user"
    assert "Flamburronix" in corrective["content"]


def test_illegal_char_name_triggers_corrective_retry():
    illegal = {**_STAGE_1, "name": "Flam@burr"}
    good = {**_STAGE_1, "name": "Flamburr"}
    client = _make_client(json.dumps([illegal]), json.dumps([good]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburr"
    second_call_messages = client.chat.complete.call_args.kwargs["messages"]
    corrective = second_call_messages[-1]
    assert corrective["role"] == "user"
    assert "Flam@burr" in corrective["content"]


def test_both_violations_at_once_trigger_single_retry():
    both_bad = {**_STAGE_1, "name": "Flamburronix@"}
    good = {**_STAGE_1, "name": "Flamburron"}
    client = _make_client(json.dumps([both_bad]), json.dumps([good]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburron"
    second_call_messages = client.chat.complete.call_args.kwargs["messages"]
    corrective = second_call_messages[-1]
    assert "Flamburronix@" in corrective["content"]


def test_last_resort_repair_truncates_too_long_name():
    too_long = {**_STAGE_1, "name": "Flamburronix"}
    still_too_long = {**_STAGE_1, "name": "Flamburronix"}
    client = _make_client(json.dumps([too_long]), json.dumps([still_too_long]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburronix"[:10]
    assert len(result[0]["name"]) == 10


def test_last_resort_repair_strips_illegal_chars():
    illegal = {**_STAGE_1, "name": "Flam@burr"}
    still_illegal = {**_STAGE_1, "name": "Flam@burr"}
    client = _make_client(json.dumps([illegal]), json.dumps([still_illegal]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburr"


def test_last_resort_repair_strips_then_truncates():
    both_bad = {**_STAGE_1, "name": "Flamburronix@wow"}
    still_bad = {**_STAGE_1, "name": "Flamburronix@wow"}
    client = _make_client(json.dumps([both_bad]), json.dumps([still_bad]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    stripped = "Flamburronixwow"
    assert result[0]["name"] == stripped[:10]


def test_line_mode_only_names_offending_stages_in_corrective_message():
    # Deliberately unrelated to the valid stage names so "not in" assertions
    # below can't pass by substring accident.
    bad_stage2 = {**_STAGE_2, "name": "Infernodrake"}
    line = [_STAGE_1, bad_stage2, _STAGE_3]
    good_line = _LINE
    client = _make_client(json.dumps(line), json.dumps(good_line))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert client.chat.complete.call_count == 2
    assert [s["name"] for s in result] == ["Flamburr", "Flamburro", "Flamburron"]
    second_call_messages = client.chat.complete.call_args.kwargs["messages"]
    corrective = second_call_messages[-1]
    assert "Infernodrake" in corrective["content"]
    # The two already-valid stage names are not called out as offenders.
    assert "Flamburr" not in corrective["content"]
    assert "Flamburron" not in corrective["content"]


def test_line_mode_repairs_only_the_offending_stage():
    bad_stage2 = {**_STAGE_2, "name": "Infernodrake"}
    line = [_STAGE_1, bad_stage2, _STAGE_3]
    client = _make_client(json.dumps(line), json.dumps(line))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert client.chat.complete.call_count == 2
    # Valid siblings untouched; only the offender is truncated.
    assert [s["name"] for s in result] == ["Flamburr", "Infernodra", "Flamburron"]


# --- boundary and charset coverage ------------------------------------------

def test_name_of_exactly_ten_chars_is_valid():
    """`> 10` is the failure condition, not `>= 10`."""
    exactly_ten = {**_STAGE_1, "name": "Flamburron"}
    client = _make_client(json.dumps([exactly_ten]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 1
    assert result[0]["name"] == "Flamburron"


@pytest.mark.parametrize("name", [
    "Nidoran♂",   # ♂
    "Nidoran♀",   # ♀
    "Flambé",     # é
    "Ho-Oh",
    "Farfetch'd",
    "Mr. Mime",
    "Type: Null",
    "A1.,'-…!?",  # … plus the rest of the punctuation set
    "/()\":;",
])
def test_gen3_charset_names_pass_through_untouched(name):
    stage = {**_STAGE_1, "name": name}
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 1
    assert result[0]["name"] == name


def test_newline_in_name_is_an_illegal_char():
    """Newline is never in the allowed set, so it takes the illegal-char path."""
    bad = {**_STAGE_1, "name": "Flam\nburr"}
    client = _make_client(json.dumps([bad]), json.dumps([bad]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburr"


# --- interaction with the shared 2-attempt budget ----------------------------

def test_name_violation_then_malformed_json_still_exits(capsys):
    """The pre-existing malformed-JSON exit wins on attempt 2."""
    bad_name = {**_STAGE_1, "name": "Flamburronix"}
    client = _make_client(json.dumps([bad_name]), "garbage output 2")
    with pytest.raises(SystemExit) as exc:
        generate_fakemon("fire lizard", "single", client=client)
    assert exc.value.code == 1
    assert client.chat.complete.call_count == 2
    err = capsys.readouterr().err
    assert "malformed JSON after 2 attempts" in err
    assert "garbage output 2" in err


def test_malformed_json_then_name_violation_repairs_without_third_call():
    """A JSON retry spends attempt 1, so attempt 2 repairs instead of retrying."""
    bad_name = {**_STAGE_1, "name": "Flamburronix"}
    client = _make_client("garbage", json.dumps([bad_name]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 2
    assert result[0]["name"] == "Flamburron"


# --- _normalize as a standalone scaffold -------------------------------------

def test_normalize_enforces_contract_and_returns_stages():
    stages = [
        {**_STAGE_1, "name": "Flamburr"},
        {**_STAGE_2, "name": "Flamburronix@wow"},
    ]
    result = _normalize(stages, "line", "standard")
    assert [s["name"] for s in result] == ["Flamburr", "Flamburron"]
    assert all(len(s["name"]) <= 10 for s in result)


def test_normalize_makes_no_api_call():
    """`_normalize` is pure post-processing — it must never touch a client."""
    client = MagicMock()
    with patch("fakemon_forge.generator.Mistral", return_value=client):
        _normalize([{**_STAGE_1, "name": "Flamburronix"}], "single", "standard")
    client.chat.complete.assert_not_called()


# --- the prompt states the contract before the retry has to enforce it -------

def test_system_prompt_states_the_name_length_limit():
    """The retry is the fallback; the prompt is the first line of defence."""
    assert "max 10 characters" in _SYSTEM_PROMPT


@pytest.mark.parametrize("char", ["♂", "♀"])
def test_system_prompt_states_the_name_charset(char):
    """The gendered signs are the charset's least guessable members, and the
    name spec is the only place they appear — so they pin the whole table."""
    assert char in _SYSTEM_PROMPT


def test_corrective_message_names_the_allowed_extras():
    """The illegal-char corrective must not under-describe the charset — é/♂/♀
    are legal, and a corrective that omits them makes the model over-restrict."""
    message = _corrective_message([], ["Flam@burr"])
    for char in "é♂♀":
        assert char in message


# --- the retry carries the offending array with it ---------------------------

def test_corrective_retry_includes_the_offending_response():
    raw = json.dumps([{**_STAGE_1, "name": "Flamburronix"}])
    client = _make_client(raw, json.dumps([{**_STAGE_1, "name": "Flamburron"}]))
    generate_fakemon("fire lizard", "single", client=client)
    messages = client.chat.complete.call_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == raw


def test_corrective_retry_keeps_valid_sibling_names_in_context():
    """In line mode the two already-valid names must still be visible to the
    model, or "return the full array again" silently rebuilds the whole line."""
    bad_line = json.dumps([_STAGE_1, {**_STAGE_2, "name": "Infernodrake"}, _STAGE_3])
    client = _make_client(bad_line, json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    assistant = client.chat.complete.call_args.kwargs["messages"][2]
    assert "Flamburr" in assistant["content"]
    assert "Flamburron" in assistant["content"]


# --- malformed model output degrades instead of raising ----------------------

@pytest.mark.parametrize("bad_name", [42, None, 7.5, ["Flamburr"]])
def test_non_string_name_is_repaired_not_raised(bad_name):
    """`_normalize` never raises — a non-string name degrades to its str()
    form, repaired against the same charset as any other name."""
    result = _normalize([{**_STAGE_1, "name": bad_name}], "single", "standard")
    assert isinstance(result[0]["name"], str)
    assert len(result[0]["name"]) <= 10
    assert all(ch in _ALLOWED_NAME_CHARS for ch in result[0]["name"])


def test_non_string_name_does_not_crash_generate_fakemon():
    client = _make_client(json.dumps([{**_STAGE_1, "name": 12345678901234}]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["name"] == "1234567890"


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

def test_exits_on_api_exception(capsys):
    client = MagicMock()
    client.chat.complete.side_effect = Exception("connection error")
    with pytest.raises(SystemExit) as exc:
        generate_fakemon("fire lizard", "single", client=client)
    assert exc.value.code == 1


def test_api_error_message_mentions_env_var(capsys):
    client = MagicMock()
    client.chat.complete.side_effect = Exception("401")
    with pytest.raises(SystemExit):
        generate_fakemon("fire lizard", "single", client=client)
    assert "MISTRAL_API_KEY" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def test_build_client_from_api_key():
    with patch("fakemon_forge.generator.Mistral") as MockMistral:
        fake_client = _make_client(json.dumps([_STAGE_1]))
        MockMistral.return_value = fake_client
        generate_fakemon("fire lizard", "single", api_key="sk-test")
        MockMistral.assert_called_once_with(api_key="sk-test")


# ---------------------------------------------------------------------------
# Tier BST targets in prompt
# ---------------------------------------------------------------------------

def _get_prompt_text(client):
    messages = client.chat.complete.call_args.kwargs["messages"]
    return " ".join(m["content"] for m in messages)


def test_standard_tier_bst_in_prompt():
    """A standalone form gets the standalone budget, not a juvenile's (#59).

    Asserts the BST hint itself, not a bare substring: the size anchors in the
    system prompt contain other three-digit numbers, so `"430" in text` could
    pass without the hint being right at all.
    """
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", tier="standard", client=client)
    assert "BST target: ~430." in _get_prompt_text(client)


def test_standard_line_bst_includes_stage3_target():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", tier="standard", client=client)
    assert (
        "BST targets: stage 1 ~295, stage 2 ~405, stage 3 ~518."
        in _get_prompt_text(client)
    )


def test_pseudo_tier_bst_includes_600():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", tier="pseudo", client=client)
    text = _get_prompt_text(client)
    assert "600" in text


def test_legendary_tier_bst_in_prompt():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", tier="legendary", client=client)
    text = _get_prompt_text(client)
    assert "580" in text


def test_mythical_tier_bst_in_prompt():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", tier="mythical", client=client)
    text = _get_prompt_text(client)
    assert "600" in text


# ---------------------------------------------------------------------------
# Evolutionary line progression guidance
# ---------------------------------------------------------------------------

def test_line_prompt_mentions_juvenile_stage():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    text = _get_prompt_text(client)
    assert any(w in text.lower() for w in ("juvenile", "child", "small"))


def test_line_prompt_mentions_adult_stage():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    text = _get_prompt_text(client)
    assert any(w in text.lower() for w in ("adult", "final form", "fully"))


def test_line_prompt_mentions_visual_distinction():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    text = _get_prompt_text(client)
    assert any(w in text.lower() for w in ("silhouette", "distinct", "different"))


def test_pseudo_tier_prompt_mentions_pseudo_legendary():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", tier="pseudo", client=client)
    text = _get_prompt_text(client)
    assert "pseudo" in text.lower()


def test_legendary_tier_prompt_mentions_legendary():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", tier="legendary", client=client)
    text = _get_prompt_text(client)
    assert "legendary" in text.lower()


def test_single_mode_does_not_mention_evo_progression():
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", client=client)
    text = _get_prompt_text(client)
    assert "juvenile" not in text.lower() and "adolescent" not in text.lower()


# ---------------------------------------------------------------------------
# levitates field in system prompt
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_levitates():
    assert "levitates" in _SYSTEM_PROMPT


def test_system_prompt_levitates_defines_never_touches_ground():
    assert "ground" in _SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# height_dm / weight_hg in system prompt
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_height_dm():
    assert "height_dm" in _SYSTEM_PROMPT


def test_system_prompt_mentions_weight_hg():
    assert "weight_hg" in _SYSTEM_PROMPT


def test_system_prompt_has_scale_anchors():
    assert "rodent" in _SYSTEM_PROMPT.lower()
    assert "dragon" in _SYSTEM_PROMPT.lower()


@pytest.mark.parametrize(
    "anchor", ["3 dm", "35 hg", "10 dm", "300 hg", "20 dm", "2100 hg"]
)
def test_system_prompt_scale_anchors_are_calibrated(anchor):
    assert anchor in _SYSTEM_PROMPT


def test_system_prompt_requires_growth_across_a_line():
    assert "grow across an evolutionary line" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# sprite_prompt carries the type conditioning
# ---------------------------------------------------------------------------
# The SDXL sprite backend dropped the SD1.5 LoRA's mechanical "firetype" tag
# table (see build_prompt in sprites.py), so this prompt spec is now the only
# thing putting a type signal in front of the image model. If it stops asking,
# a Fire/Flying creature renders with nothing anywhere in the pipeline saying
# so — and nothing downstream can notice, because sprites.py accepts `types`
# without reading it.

def test_system_prompt_requires_sprite_prompt_to_show_the_types():
    assert "show the creature's types" in _SYSTEM_PROMPT


def test_system_prompt_gives_type_to_visual_examples():
    """Abstract "describe the types" under-delivers; the worked examples are
    what turn a type into flames-and-embers wording the model can render."""
    assert "embers for Fire" in _SYSTEM_PROMPT
    assert "feathers for Flying" in _SYSTEM_PROMPT


def test_system_prompt_says_the_sprite_model_never_sees_the_types_field():
    """The reason the requirement is not optional — stated so a future edit
    trimming the spec for length can see what it would be giving up."""
    assert "never sees the types field" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# sprite_prompt requires a colour scheme
# ---------------------------------------------------------------------------
# The model echoes whatever colours the user's description mentions, so a
# monochrome description becomes a monochrome sprite: a live 3-stage dragon
# line ("orange-red ... amber belly ...") rendered warm-on-warm across all
# three stages. And generate_shiny's hue rotation preserves the palette's
# internal structure, so a monochrome normal necessarily yields a monochrome
# shiny — the scheme requirement here is the only counterweight.

def test_system_prompt_requires_a_colour_scheme():
    assert "2 or 3 DISTINCT colours" in _SYSTEM_PROMPT


def test_system_prompt_says_shades_are_not_a_scheme():
    """Without this, "orange scales, amber belly" satisfies a naive reading of
    "2 colours" — the failure observed live."""
    assert "one colour twice" in _SYSTEM_PROMPT


def test_system_prompt_gives_colour_scheme_example():
    """Abstract rules under-deliver with these models; the worked example is
    what turns the rule into renderable wording."""
    assert "orange body" in _SYSTEM_PROMPT
    assert "cream belly" in _SYSTEM_PROMPT


def test_system_prompt_tells_model_to_invent_secondary_for_monochrome_input():
    """The trigger case: a user description naming only one colour family must
    gain a contrasting secondary, not have its shades echoed back."""
    assert "invent a fitting secondary" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# sprite_prompt is tags, and short enough to survive CLIP
# ---------------------------------------------------------------------------
# The sprite LoRA is trained on comma-separated tags, so prose is off
# distribution. Worse, prose runs long: build_prompt appends "white background"
# (and extra_tags like the chibi set) AFTER this string, and CLIP truncates at
# 77 tokens — so an over-long sprite_prompt does not merely waste tokens, it
# silently deletes the styling instructions. Observed live: every prompt in a
# 3-stage run lost "white background", and the resulting vignetted backdrops
# then defeated split_front_back_canvas's uniform-border check on all 3 stages.

def test_system_prompt_demands_tags_not_sentences():
    assert "TAGS" in _SYSTEM_PROMPT
    assert "Tags, never sentences" in _SYSTEM_PROMPT
    assert "No verbs, no clauses, no full stops" in _SYSTEM_PROMPT


def test_system_prompt_caps_sprite_prompt_length():
    """A cap the model can act on, not a vague "keep it short"."""
    assert "at most 18 tags and 35 words" in _SYSTEM_PROMPT


def test_system_prompt_bans_framing_words_in_sprite_prompt():
    """"large"/"imposing" are framing instructions to the image model, not
    facts about the creature. A live stage-2 prompt carrying both rendered a
    full-bleed close-up instead of a sprite, while stage 1 of the same line
    ("tiny", "cute expression") came out clean."""
    for banned in ("large", "towering", "imposing", "close-up"):
        assert f'"{banned}"' in _SYSTEM_PROMPT


def test_system_prompt_bans_small_size_words_in_sprite_prompt():
    """The other direction of the same failure: on the first ROM-injection
    round, 94 of 99 stage-1 prompts said "tiny"/"small" and their sprites
    filled a median 48% of the canvas against 78% without — unreadable once a
    GBA cell scales that by 1/12. build_prompt strips these mechanically; the
    ban here is what lets the model spend the tag on something that helps."""
    for banned in ("tiny", "small", "little", "miniature"):
        assert f'"{banned}"' in _SYSTEM_PROMPT


def test_system_prompt_redirects_juvenile_smallness_into_proportions():
    """The ban needs somewhere for juvenile-ness to go, or the model just picks
    a synonym — a child form shows as simpler, rounder proportions."""
    assert "SIMPLER, ROUNDER" in _SYSTEM_PROMPT
    assert "plump undeveloped body" in _SYSTEM_PROMPT


def test_system_prompt_bans_enclosing_and_scenery_words_in_sprite_prompt():
    """An enclosing shape becomes the picture's frame, and a setting becomes
    painted scenery. Both defeat the flat-backdrop precondition that
    split_front_back_canvas and _flatten_background_to_key share (issue #83):
    a live "porthole" tag rendered the whole creature inside a circular frame,
    and a live battleship prompt painted a full cloudy sky."""
    for banned in ("porthole", "frame", "vignette", "clouds", "background"):
        assert f'"{banned}"' in _SYSTEM_PROMPT


def test_system_prompt_redirects_enclosing_features_onto_the_body():
    """The ban needs somewhere for a legitimately ship-like feature to go, or
    the model just drops the detail — attach it to the body instead."""
    assert "portholes set into its flank" in _SYSTEM_PROMPT


def test_system_prompt_redirects_stage_growth_into_features():
    """The ban needs somewhere for the growth to go, or the model just picks a
    synonym — evolution shows as more/bigger features, and the numeric size
    difference already lives in height_dm / weight_hg."""
    assert "MORE and BIGGER FEATURES" in _SYSTEM_PROMPT
    assert "height_dm" in _SYSTEM_PROMPT


def test_system_prompt_explains_what_going_long_costs():
    """States the consequence, so the cap reads as load-bearing rather than
    stylistic — a future edit relaxing it can see what it would reintroduce."""
    assert "77 tokens" in _SYSTEM_PROMPT
    assert "deletes the background and style" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# height_dm / weight_hg defaulting in _normalize
# ---------------------------------------------------------------------------

def test_normalize_defaults_line_stage1_height_and_weight():
    stage = {**_STAGE_1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 5
    assert result[0]["weight_hg"] == 30


def test_normalize_defaults_line_stage2_height_and_weight():
    stage = {**_STAGE_2}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 10
    assert result[0]["weight_hg"] == 150


def test_normalize_defaults_line_stage3_height_and_weight():
    stage = {**_STAGE_3}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 17
    assert result[0]["weight_hg"] == 600


def test_normalize_defaults_single_standard_uses_stage2_values():
    stage = {**_STAGE_1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "single", "standard")
    assert result[0]["height_dm"] == 10
    assert result[0]["weight_hg"] == 150


@pytest.mark.parametrize("tier", ["pseudo", "legendary", "mythical"])
def test_normalize_defaults_single_big_tiers_use_stage3_values(tier):
    stage = {**_STAGE_1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "single", tier)
    assert result[0]["height_dm"] == 17
    assert result[0]["weight_hg"] == 600


@pytest.mark.parametrize("tier", ["standard", "pseudo", "legendary", "mythical"])
def test_normalize_line_defaults_ignore_tier(tier):
    """In line mode the table is keyed off stage["stage"] alone, never tier."""
    stage = {**_STAGE_1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", tier)
    assert result[0]["height_dm"] == 5
    assert result[0]["weight_hg"] == 30


def test_normalize_single_defaults_ignore_stage_number():
    """A single standard form is a standalone species, not a stage-1 juvenile."""
    stage = {**_STAGE_1, "stage": 1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "single", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (10, 150)


def test_normalize_defaults_only_missing_field():
    """If only one of the two keys is present, only the missing one is defaulted."""
    stage = {**_STAGE_1, "height_dm": 42}
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 42
    assert result[0]["weight_hg"] == 30


# ---------------------------------------------------------------------------
# height_dm / weight_hg clamping in _normalize
# ---------------------------------------------------------------------------

def test_normalize_clamps_zero_height_to_one():
    stage = {**_STAGE_1, "height_dm": 0, "weight_hg": 30}
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 1


def test_normalize_clamps_zero_weight_to_one():
    stage = {**_STAGE_1, "height_dm": 5, "weight_hg": 0}
    result = _normalize([stage], "line", "standard")
    assert result[0]["weight_hg"] == 1


def test_normalize_clamps_negative_height_to_one():
    stage = {**_STAGE_1, "height_dm": -10, "weight_hg": 30}
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 1


def test_normalize_clamps_huge_height_to_999():
    stage = {**_STAGE_1, "height_dm": 50000, "weight_hg": 30}
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 999


def test_normalize_clamps_huge_weight_to_9999():
    stage = {**_STAGE_1, "height_dm": 5, "weight_hg": 999999}
    result = _normalize([stage], "line", "standard")
    assert result[0]["weight_hg"] == 9999


def test_normalize_clamps_negative_weight_to_one():
    stage = {**_STAGE_1, "height_dm": 5, "weight_hg": -250}
    result = _normalize([stage], "line", "standard")
    assert result[0]["weight_hg"] == 1


def test_normalize_present_in_range_values_pass_through():
    stage = {**_STAGE_1, "height_dm": 8, "weight_hg": 77}
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == 8
    assert result[0]["weight_hg"] == 77


def test_normalize_lower_bound_values_pass_through():
    stage = {**_STAGE_1, "height_dm": 1, "weight_hg": 1}
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (1, 1)


def test_normalize_upper_bound_values_pass_through():
    """The bounds are inclusive — 999/9999 must not be clamped down."""
    stage = {**_STAGE_1, "height_dm": 999, "weight_hg": 9999}
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (999, 9999)


def test_normalize_clamps_one_past_each_upper_bound():
    stage = {**_STAGE_1, "height_dm": 1000, "weight_hg": 10000}
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (999, 9999)


def test_normalize_is_idempotent_for_height_and_weight():
    stage = {**_STAGE_1, "height_dm": 50000, "weight_hg": 0}
    once = _normalize([stage], "line", "standard")[0]
    twice = _normalize([once], "line", "standard")[0]
    assert (twice["height_dm"], twice["weight_hg"]) == (999, 1)


def test_normalize_height_weight_result_from_generate_fakemon():
    stage = {**_STAGE_1}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("fire lizard", "single", tier="standard", client=client)
    assert result[0]["height_dm"] == 10
    assert result[0]["weight_hg"] == 150


def test_generate_fakemon_line_defaults_every_stage():
    client = _make_client(json.dumps(_LINE))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert [(s["height_dm"], s["weight_hg"]) for s in result] == [
        (5, 30), (10, 150), (17, 600),
    ]


def test_generate_fakemon_clamps_model_supplied_values():
    stages = [
        {**_STAGE_1, "height_dm": 0, "weight_hg": 0},
        {**_STAGE_2, "height_dm": 12, "weight_hg": 400},
        {**_STAGE_3, "height_dm": 50000, "weight_hg": 999999},
    ]
    client = _make_client(json.dumps(stages))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert [(s["height_dm"], s["weight_hg"]) for s in result] == [
        (1, 1), (12, 400), (999, 9999),
    ]


def test_generate_fakemon_always_emits_in_range_height_and_weight():
    """The output contract, independent of what the model returned."""
    stages = [
        {**_STAGE_1, "height_dm": -1},
        {**_STAGE_2, "weight_hg": 65535},
        {**_STAGE_3},
    ]
    client = _make_client(json.dumps(stages))
    for stage in generate_fakemon("fire lizard", "line", client=client):
        assert isinstance(stage["height_dm"], int) and 1 <= stage["height_dm"] <= 999
        assert isinstance(stage["weight_hg"], int) and 1 <= stage["weight_hg"] <= 9999


# ---------------------------------------------------------------------------
# abilities_gen3 pool and system prompt
# ---------------------------------------------------------------------------

def _ability_table() -> dict:
    """The resource table read straight from disk.

    Expectations are derived from the file, never a hardcoded count or copied
    list — the table has drifted before and the pool must track it.
    """
    path = Path(__file__).parent.parent / "resources" / "gen3_abilities.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_ability_list_items() -> list[str]:
    """The comma-separated items of the closed set embedded in the prompt."""
    joined = ", ".join(_ABILITY_POOL)
    line = next(l for l in _SYSTEM_PROMPT.splitlines() if joined in l)
    return [item.strip() for item in line.split(",")]


def test_ability_pool_is_the_table_minus_indexes_0_and_76():
    table = _ability_table()
    assert _ABILITY_POOL == [n for i, n in table.items() if i not in ("0", "76")]


def test_ability_pool_drops_exactly_two_entries():
    """Whatever the table's size, indexes 0 and 76 are the only exclusions."""
    assert len(_ABILITY_POOL) == len(_ability_table()) - 2


def test_ability_pool_excludes_none():
    assert "None" not in _ABILITY_POOL


def test_ability_pool_excludes_cacophony():
    assert "Cacophony" not in _ABILITY_POOL


def test_ability_pool_includes_air_lock():
    """Index 77 (Air Lock) is usable — only indexes 0 and 76 are excluded."""
    assert "Air Lock" in _ABILITY_POOL


def test_system_prompt_mentions_abilities_gen3_field():
    assert "abilities_gen3" in _SYSTEM_PROMPT


def test_system_prompt_lists_real_abilities():
    assert "Blaze" in _SYSTEM_PROMPT
    assert "Compound Eyes" in _SYSTEM_PROMPT


def test_system_prompt_excludes_cacophony():
    assert "Cacophony" not in _SYSTEM_PROMPT


def test_system_prompt_embeds_the_whole_pool():
    """The prompt lists the pool verbatim as the closed set to choose from."""
    assert ", ".join(_ABILITY_POOL) in _SYSTEM_PROMPT


def test_system_prompt_ability_list_excludes_index_0_and_76_names():
    """Whatever those two indexes are named in the table, they stay out."""
    table = _ability_table()
    items = _prompt_ability_list_items()
    assert table["0"] not in items
    assert table["76"] not in items


def test_system_prompt_states_one_or_two_distinct():
    assert "1 or 2 distinct" in _SYSTEM_PROMPT


def test_system_prompt_prefers_two_abilities():
    assert "Prefer two abilities" in _SYSTEM_PROMPT


def test_system_prompt_states_line_sharing_convention():
    assert "all stages should share the same abilities_gen3" in _SYSTEM_PROMPT


def test_system_prompt_ties_free_text_ability_to_abilities_gen3():
    assert "free-text ability above should express the same concept" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# abilities_gen3 validation in _normalize
# ---------------------------------------------------------------------------

def test_normalize_keeps_valid_ability():
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Blaze"]


def test_normalize_drops_invalid_ability_keeps_valid():
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze", "Solar Power"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Blaze"]


def test_normalize_drops_all_invented_abilities():
    stage = {**_STAGE_1, "abilities_gen3": ["Molten Core", "Ashwalk"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_normalize_drops_none_literal():
    stage = {**_STAGE_1, "abilities_gen3": ["None"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_normalize_drops_cacophony_literal():
    stage = {**_STAGE_1, "abilities_gen3": ["Cacophony"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_normalize_missing_key_defaults_to_empty_list():
    stage = {**_STAGE_1}
    stage.pop("abilities_gen3", None)
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_normalize_empty_list_stays_empty():
    stage = {**_STAGE_1, "abilities_gen3": []}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_normalize_canonicalizes_casing_and_spacing():
    stage = {**_STAGE_1, "abilities_gen3": ["compoundeyes"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Compound Eyes"]


def test_normalize_canonicalizes_extra_whitespace():
    stage = {**_STAGE_1, "abilities_gen3": ["COMPOUND  EYES"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Compound Eyes"]


def test_normalize_canonicalizes_tabs_and_newlines():
    """The contract is "".join(name.split()).lower() — a plain
    .replace(" ", "") would leave the tab/newline in and fail to match."""
    stage = {**_STAGE_1, "abilities_gen3": ["Compound\tEyes", " blaze\n"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Compound Eyes", "Blaze"]


def test_normalize_collapses_duplicates_different_casing():
    stage = {**_STAGE_1, "abilities_gen3": ["Compound Eyes", "compoundeyes"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Compound Eyes"]


def test_normalize_dedup_then_cap_keeps_two_distinct():
    """Cap-then-dedup would wrongly yield only one entry here."""
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze", "blaze", "Flash Fire"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Blaze", "Flash Fire"]


def test_normalize_caps_three_plus_valid_distinct_to_two():
    stage = {**_STAGE_1, "abilities_gen3": ["Blaze", "Flash Fire", "Compound Eyes"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Blaze", "Flash Fire"]


def test_normalize_leaves_free_text_ability_untouched():
    """`ability` stays creative flavour even when it isn't a real Gen 3 name."""
    stage = {**_STAGE_1, "ability": "Molten Core", "abilities_gen3": ["blaze"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["ability"] == "Molten Core"


def test_normalize_validates_each_stage_independently():
    """No cross-stage repair: a stage's invalid entries don't borrow from its
    siblings, and a valid pair on the final stage survives."""
    stages = [
        {**_STAGE_1, "abilities_gen3": ["Blaze"]},
        {**_STAGE_2, "abilities_gen3": ["Ashwalk"]},
        {**_STAGE_3, "abilities_gen3": ["Blaze", "Flash Fire"]},
    ]
    result = _normalize(stages, "line", "standard")
    assert [s["abilities_gen3"] for s in result] == [
        ["Blaze"], [], ["Blaze", "Flash Fire"],
    ]


def test_normalize_abilities_gen3_makes_no_api_call():
    client = MagicMock()
    with patch("fakemon_forge.generator.Mistral", return_value=client):
        _normalize([{**_STAGE_1, "abilities_gen3": ["blaze"]}], "single", "standard")
    client.chat.complete.assert_not_called()


def test_generate_fakemon_normalizes_abilities_gen3():
    stage = {**_STAGE_1, "abilities_gen3": ["blaze", "Solar Power"]}
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["abilities_gen3"] == ["Blaze"]


def test_generate_fakemon_emits_abilities_gen3_on_every_stage():
    """Even when the model omits the field entirely, every stage carries it."""
    assert all("abilities_gen3" not in s for s in _LINE)
    client = _make_client(json.dumps(_LINE))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert [s["abilities_gen3"] for s in result] == [[], [], []]


# ---------------------------------------------------------------------------
# types field: pool constraint + repair
# ---------------------------------------------------------------------------

def test_type_pool_is_the_17_gen3_types():
    assert len(_TYPE_POOL) == 17
    assert "Fairy" not in _TYPE_POOL  # Gen 6; has no Gen 3 byte
    assert set(_TYPE_POOL) == {
        "Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug",
        "Ghost", "Steel", "Fire", "Water", "Grass", "Electric", "Psychic",
        "Ice", "Dragon", "Dark",
    }


def test_system_prompt_lists_every_type():
    for type_name in _TYPE_POOL:
        assert type_name in _SYSTEM_PROMPT


def test_system_prompt_forbids_inventing_a_type():
    assert "Do not invent one" in _SYSTEM_PROMPT
    assert "Sound" in _SYSTEM_PROMPT  # named as the example non-type


def test_normalize_types_drops_an_invented_type():
    """Regression: a generated line came back Grass/Sound and killed export_ini
    with KeyError: 'Sound' after every sprite and cry had been rendered."""
    assert _normalize_types(["Grass", "Sound"]) == ["Grass"]


def test_normalize_types_canonicalizes_spelling():
    assert _normalize_types(["grass", "FLYING"]) == ["Grass", "Flying"]
    assert _normalize_types(["  water  "]) == ["Water"]


def test_normalize_types_collapses_duplicates():
    assert _normalize_types(["Fire", "fire"]) == ["Fire"]


def test_normalize_types_caps_at_two():
    assert _normalize_types(["Fire", "Water", "Grass"]) == ["Fire", "Water"]


def test_normalize_types_drops_fairy():
    """Fairy is outside the pool, so it is dropped rather than encoded; a
    Water/Fairy concept lands as mono-Water."""
    assert _normalize_types(["Water", "Fairy"]) == ["Water"]


@pytest.mark.parametrize("raw", [[], ["Sound"], ["Fairy"], None, "Fire", [42], [None]])
def test_normalize_types_never_returns_empty(raw):
    """Every stage must carry a primary type for export_ini to encode."""
    assert _normalize_types(raw) == ["Normal"]


def test_normalize_repairs_types_on_every_stage():
    stages = [
        {**_STAGE_1, "types": ["Grass", "Sound"]},
        {**_STAGE_2, "types": ["Sound"]},
        {**_STAGE_3, "types": ["grass", "Poison"]},
    ]
    result = _normalize(stages, "line", "standard")
    assert [s["types"] for s in result] == [
        ["Grass"], ["Normal"], ["Grass", "Poison"],
    ]


def test_normalize_category_falls_back_to_a_repaired_type():
    """The category fallback uses the primary type word, so it must never
    inherit an invented type."""
    stage = {**_STAGE_1, "types": ["Sound"], "category": ""}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "NORMAL"


def test_generate_fakemon_normalizes_types():
    stage = {**_STAGE_1, "types": ["grass", "Sound"]}
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("singing onion", "single", client=client)
    assert result[0]["types"] == ["Grass"]


# ---------------------------------------------------------------------------
# category field: prompt
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_category_field():
    assert "category" in _SYSTEM_PROMPT


def test_system_prompt_category_gives_examples():
    assert '"SEED"' in _SYSTEM_PROMPT
    assert '"MOUSE"' in _SYSTEM_PROMPT
    assert '"TINY TURTLE"' in _SYSTEM_PROMPT


def test_system_prompt_category_forbids_type_word():
    assert "not its type" in _SYSTEM_PROMPT.lower()
    assert '"FIRE"' in _SYSTEM_PROMPT or '"WATER"' in _SYSTEM_PROMPT


def test_system_prompt_category_forbids_trailing_pokemon():
    assert "POKEMON" in _SYSTEM_PROMPT
    assert "No trailing" in _SYSTEM_PROMPT


def test_system_prompt_category_states_max_length():
    assert "11 characters" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# category field: normalization in _normalize
# ---------------------------------------------------------------------------

def test_normalize_uppercases_lowercase_category():
    stage = {**_STAGE_1, "category": "seed"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEED"


def test_normalize_uppercases_mixed_case_category():
    stage = {**_STAGE_1, "category": "Tiny Turtle"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "TINY TURTLE"


def test_normalize_category_exactly_eleven_chars_passes_through():
    """`> 11` is the failure condition, not `>= 11`."""
    stage = {**_STAGE_1, "category": "TINY TURTLE"}
    assert len("TINY TURTLE") == 11
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "TINY TURTLE"


def test_normalize_category_over_eleven_chars_truncated_no_retry():
    stage = {**_STAGE_1, "category": "TINY TURTLES"}
    assert len("TINY TURTLES") == 12
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "TINY TURTLE"
    assert len(result[0]["category"]) == 11


def test_over_long_category_does_not_trigger_corrective_retry():
    """An over-long name costs a retry; an over-long category never does."""
    stage = {**_STAGE_1, "category": "WAY TOO LONG NOUN HERE"}
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert client.chat.complete.call_count == 1
    assert result[0]["category"] == "WAY TOO LON"


def test_normalize_category_strips_illegal_chars():
    stage = {**_STAGE_1, "category": "SE@ED"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEED"


def test_normalize_category_strips_illegal_chars_then_truncates():
    stage = {**_STAGE_1, "category": "TINY@ TURTLE!!!!"}
    result = _normalize([stage], "single", "standard")
    stripped = "TINY TURTLE!!!!"
    assert result[0]["category"] == stripped.upper()[:11]


def test_normalize_category_strips_trailing_pokemon_case_insensitive():
    stage = {**_STAGE_1, "category": "Seed Pokemon"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEED"


def test_normalize_category_keeps_pokemon_as_mid_word_substring():
    """Only an exact trailing " POKEMON" token is stripped."""
    stage = {**_STAGE_1, "category": "Seedmon"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEEDMON"


def test_normalize_category_strip_pokemon_before_truncate():
    long_with_suffix = "A VERY LONG SEED POKEMON"
    stage = {**_STAGE_1, "category": long_with_suffix}
    result = _normalize([stage], "single", "standard")
    expected = long_with_suffix.upper()[: -len(" POKEMON")][:11]
    assert result[0]["category"] == expected


def test_normalize_category_missing_falls_back_to_type_word():
    stage = {**_STAGE_1}
    stage.pop("category", None)
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "FIRE"


def test_normalize_category_empty_string_falls_back_to_type_word():
    stage = {**_STAGE_1, "category": ""}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "FIRE"


@pytest.mark.parametrize("bad_value", [None, 42, ["Seed"], {"noun": "Seed"}])
def test_normalize_category_non_string_falls_back_to_type_word(bad_value):
    stage = {**_STAGE_1, "category": bad_value}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "FIRE"


def test_normalize_category_all_illegal_chars_falls_back_to_type_word():
    stage = {**_STAGE_1, "category": "\n\t"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "FIRE"


@pytest.mark.parametrize("blank", ["   ", "\n \t", " POKEMON"])
def test_normalize_category_blank_after_cleaning_falls_back_to_type_word(blank):
    """Whitespace is not a usable noun — a blank category never survives."""
    stage = {**_STAGE_1, "category": blank}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "FIRE"


def test_normalize_category_truncation_does_not_leave_trailing_space():
    stage = {**_STAGE_1, "category": "GIANT SEED PODS"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "GIANT SEED"


def test_normalize_category_strips_trailing_pokemon_despite_stray_spaces():
    """A stray space must not defeat the suffix check and leave "POKEMO"."""
    stage = {**_STAGE_1, "category": "Seed Pokemon "}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEED"


def test_normalize_category_uses_second_type_never_used():
    """Fallback always uses the primary (first) type, not any secondary type."""
    stage = {**_STAGE_1, "types": ["Water", "Flying"]}
    stage.pop("category", None)
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "WATER"


def test_normalize_validates_category_independently_per_stage():
    stages = [
        {**_STAGE_1, "category": "seed"},
        {**_STAGE_2, "category": ""},
        {**_STAGE_3, "category": "TALL SEED PODS"},
    ]
    result = _normalize(stages, "line", "standard")
    assert result[0]["category"] == "SEED"
    assert result[1]["category"] == "FIRE"
    assert result[2]["category"] == "TALL SEED P"


def test_generate_fakemon_emits_category_when_model_omits_it():
    assert all("category" not in s for s in _LINE)
    client = _make_client(json.dumps(_LINE))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert [s["category"] for s in result] == ["FIRE", "FIRE", "FIRE"]


def test_generate_fakemon_normalizes_category():
    stage = {**_STAGE_1, "category": "seed"}
    client = _make_client(json.dumps([stage]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["category"] == "SEED"


# --- malformed abilities_gen3 degrades instead of raising --------------------

@pytest.mark.parametrize("entry", [42, None, 7.5, ["Blaze"], {"name": "Blaze"}])
def test_non_string_ability_entry_is_dropped_not_raised(entry):
    """A non-string entry hit _normalize_ability_name's .split() and raised
    AttributeError out of _normalize, which runs outside the try/except."""
    stage = {**_STAGE_1, "abilities_gen3": [entry]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_non_string_entry_does_not_hide_valid_siblings():
    stage = {**_STAGE_1, "abilities_gen3": [None, "Blaze", 42, "Flash Fire"]}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == ["Blaze", "Flash Fire"]


@pytest.mark.parametrize("raw", ["Blaze", 42, None, {"1": "Blaze"}])
def test_non_list_abilities_gen3_is_treated_as_absent(raw):
    """Matches export_ini's reading of the same field — a bare string must not
    be iterated character by character."""
    stage = {**_STAGE_1, "abilities_gen3": raw}
    result = _normalize([stage], "single", "standard")
    assert result[0]["abilities_gen3"] == []


def test_generate_fakemon_survives_malformed_abilities_gen3():
    stages = [{**_STAGE_1, "abilities_gen3": [{"name": "Blaze"}, 7]}]
    client = _make_client(json.dumps(stages))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert result[0]["abilities_gen3"] == []


# ---------------------------------------------------------------------------
# Off-spec stage numbers and dimension types degrade instead of raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage_no", [4, 0, -1, None, "one", [1], {"stage": 1}])
def test_off_spec_stage_number_falls_back_to_the_tier_table(stage_no):
    """A hallucinated stage number must not KeyError out of _normalize, which
    runs outside generate_fakemon's try/except."""
    stage = {**_STAGE_1, "stage": stage_no}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (10, 150)


@pytest.mark.parametrize("stage_no", ["3", 3.0])
def test_coercible_stage_number_recovers_the_line_table(stage_no):
    """A stage number the model wrote as a string still resolves to its own
    row — falling back to the tier table there would lose real information.
    Stage 3 is deliberately chosen: its (17, 600) differs from the standard
    tier fallback, so the assertion can't pass by coincidence."""
    stage = {**_STAGE_1, "stage": stage_no}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (17, 600)


def test_missing_stage_key_falls_back_to_the_tier_table():
    stage = {k: v for k, v in _STAGE_1.items() if k != "stage"}
    result = _normalize([stage], "line", "pseudo")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (17, 600)


def test_off_spec_stage_number_still_uses_the_line_table_when_valid():
    """The fallback must not swallow the stage table for well-formed input."""
    stage = {**_STAGE_1, "stage": 3}
    stage.pop("height_dm", None)
    stage.pop("weight_hg", None)
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (17, 600)


@pytest.mark.parametrize("value, expected", [
    ("12", 12),        # JSON string instead of a number
    (7.8, 7),          # float truncates rather than persisting a non-integer
    (True, 5),         # bool is not a measurement
    (None, 5),
    ("tall", 5),
    ([7], 5),
    ({"dm": 7}, 5),
])
def test_non_integer_height_degrades_to_default_or_coerces(value, expected):
    stage = {**_STAGE_1, "height_dm": value, "weight_hg": 30}
    result = _normalize([stage], "line", "standard")
    assert result[0]["height_dm"] == expected
    assert isinstance(result[0]["height_dm"], int)


@pytest.mark.parametrize("value, expected", [
    ("400", 400),
    (149.9, 149),
    ("heavy", 30),
    (None, 30),
])
def test_non_integer_weight_degrades_to_default_or_coerces(value, expected):
    stage = {**_STAGE_1, "height_dm": 5, "weight_hg": value}
    result = _normalize([stage], "line", "standard")
    assert result[0]["weight_hg"] == expected
    assert isinstance(result[0]["weight_hg"], int)


def test_coerced_string_dimension_is_still_clamped():
    """Coercion happens before the bounds check, not instead of it."""
    stage = {**_STAGE_1, "height_dm": "50000", "weight_hg": "0"}
    result = _normalize([stage], "line", "standard")
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (999, 1)


def test_generate_fakemon_survives_fully_off_spec_sizes():
    stages = [{**_STAGE_1, "stage": 9, "height_dm": "big", "weight_hg": None}]
    client = _make_client(json.dumps(stages))
    result = generate_fakemon("fire lizard", "line", client=client)
    assert (result[0]["height_dm"], result[0]["weight_hg"]) == (10, 150)


# ---------------------------------------------------------------------------
# category: the accented suffix and off-spec types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "Seed Pokémon",
    "SEED POKÉMON",
    "Seed POKéMON",
    "Seed Pokémon ",
])
def test_normalize_category_strips_the_accented_pokemon_suffix(raw):
    """The prompt writes "Pokémon" with the accent throughout, so that is the
    likelier echo — and é survives charset stripping, so the un-accented check
    alone left "SEED POKÉMO" behind."""
    result = _normalize([{**_STAGE_1, "category": raw}], "single", "standard")
    assert result[0]["category"] == "SEED"


@pytest.mark.parametrize("blank", [" POKÉMON", " Pokémon"])
def test_normalize_category_bare_accented_suffix_falls_back(blank):
    result = _normalize([{**_STAGE_1, "category": blank}], "single", "standard")
    assert result[0]["category"] == "FIRE"


def test_normalize_category_keeps_accented_word_that_is_not_the_suffix():
    """Only a trailing " POKÉMON" token goes — é is otherwise a legal char."""
    result = _normalize([{**_STAGE_1, "category": "Flambé"}], "single", "standard")
    assert result[0]["category"] == "FLAMBé"


def test_normalize_category_never_emits_uppercase_e_acute():
    """The Gen 3 set has é but no É, so upcasing must not manufacture one —
    the category would otherwise carry a character the charset just rejected."""
    result = _normalize([{**_STAGE_1, "category": "flambé cake"}], "single", "standard")
    assert "É" not in result[0]["category"]
    assert all(ch in _ALLOWED_NAME_CHARS for ch in result[0]["category"])


@pytest.mark.parametrize("raw", ["Flambé", "FLAMBÉ", "flambé", "SEED", "Tiny Turtle"])
def test_normalize_category_output_is_always_inside_the_charset(raw):
    result = _normalize([{**_STAGE_1, "category": raw}], "single", "standard")
    assert all(ch in _ALLOWED_NAME_CHARS for ch in result[0]["category"])


@pytest.mark.parametrize("types", [[], None, "Fire", [42], {"0": "Fire"}])
def test_off_spec_types_degrade_to_normal_not_raised(types):
    """types=[] raised IndexError out of the category fallback."""
    stage = {**_STAGE_1, "types": types}
    stage.pop("category", None)
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "NORMAL"


def test_missing_types_key_degrades_to_normal():
    stage = {k: v for k, v in _STAGE_1.items() if k != "types"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "NORMAL"


def test_valid_category_survives_off_spec_types():
    """The fallback is only reached when the category itself is unusable."""
    stage = {**_STAGE_1, "types": [], "category": "seed"}
    result = _normalize([stage], "single", "standard")
    assert result[0]["category"] == "SEED"


# ---------------------------------------------------------------------------
# Stage count: 2-stage lines (#59)
# ---------------------------------------------------------------------------

# The prompts as they read before #59, verbatim. The structure-identical
# guarantee is asserted against these with only the BST numbers substituted --
# literal byte-identity is impossible, since correcting those numbers is the
# point of the issue.
_PRE_59_LINE_PROMPT = (
    "Generate three evolutionary stages (stages 1, 2, and 3) for a Fakemon "
    "based on this description:\n"
    "\n"
    "DESC\n"
    "\n"
    "BST targets: stage 1 ~300, stage 2 ~420, stage 3 ~520.\n"
    "Evolutionary progression — each stage must look and feel visually "
    "distinct:\n"
    "  Stage 1: juvenile/child form — small and simple, cute or curious "
    "expression, limited limbs or features, undeveloped power.\n"
    "  Stage 2: adolescent/teenage form — noticeably larger, silhouette more "
    "defined, signature features emerging, power becoming apparent.\n"
    "  Stage 3: adult/final form — fully developed, imposing presence, complex "
    "design with a different silhouette from stage 1, design complexity at its "
    "peak."
)

_PRE_59_SINGLE_PROMPT = (
    "Generate one stage (stage 1 only) for a Fakemon based on this "
    "description:\n"
    "\n"
    "DESC\n"
    "\n"
    "BST target: ~300."
)


def _bst_line(prompt: str) -> str:
    return next(l for l in prompt.splitlines() if l.startswith("BST"))


# --- the structure-identical guarantee ---------------------------------------

def test_default_line_prompt_is_structure_identical_to_pre_59():
    """`--mode line` with no `stages` must differ from the pre-#59 prompt in
    the corrected BST numbers and nothing else. Full-string equality, so a
    reworded hint, a reordered section or a lost line all fail."""
    expected = _PRE_59_LINE_PROMPT.replace(
        "stage 1 ~300, stage 2 ~420, stage 3 ~520",
        "stage 1 ~295, stage 2 ~405, stage 3 ~518",
    )
    assert _user_prompt("DESC", "line", "standard") == expected


def test_default_single_prompt_is_structure_identical_to_pre_59():
    expected = _PRE_59_SINGLE_PROMPT.replace("~300.", "~430.")
    assert _user_prompt("DESC", "single", "standard") == expected


def test_explicit_three_stages_matches_the_default():
    """Passing the default explicitly must change nothing."""
    assert (
        _user_prompt("DESC", "line", "standard", 3)
        == _user_prompt("DESC", "line", "standard")
    )


# --- BST target counts and values --------------------------------------------

@pytest.mark.parametrize("mode, stage_count, expected", [
    ("single", 1, "BST target: ~430."),
    ("line", 2, "BST targets: stage 1 ~305, stage 2 ~468."),
    ("line", 3, "BST targets: stage 1 ~295, stage 2 ~405, stage 3 ~518."),
])
def test_bst_hint_matches_the_stage_count(mode, stage_count, expected):
    prompt = _user_prompt("DESC", mode, "standard", stage_count)
    assert _bst_line(prompt) == expected


def test_two_stage_prompt_names_two_stages():
    prompt = _user_prompt("DESC", "line", "standard", 2)
    assert "three evolutionary stages" not in prompt
    assert "stages 1 and 2" in prompt


# --- progression wording ------------------------------------------------------

def test_two_stage_progression_omits_the_adolescent_middle():
    """A 2-stage line goes juvenile -> adult; describing a middle form would
    ask for a stage that is not being generated."""
    prompt = _user_prompt("DESC", "line", "standard", 2)
    assert "Stage 1:" in prompt
    assert "Stage 2:" in prompt
    assert "Stage 3:" not in prompt
    assert "adolescent" not in prompt


def test_three_stage_progression_keeps_the_adolescent_middle():
    prompt = _user_prompt("DESC", "line", "standard", 3)
    assert "adolescent" in prompt
    assert "Stage 3:" in prompt


def test_single_mode_has_no_progression_text():
    prompt = _user_prompt("DESC", "single", "standard", 1)
    assert "Evolutionary progression" not in prompt


# --- stage count is ignored in single mode ------------------------------------

@pytest.mark.parametrize("stage_count", [1, 2, 3])
def test_stage_count_is_ignored_in_single_mode(stage_count):
    assert (
        _user_prompt("DESC", "single", "standard", stage_count)
        == _user_prompt("DESC", "single", "standard")
    )


# --- pseudo + single is preserved until task 11 rejects it --------------------

def test_pseudo_single_still_prompts_its_pre_59_value():
    """pseudo has only a 3-stage row now; `_bst_row`'s fallback keeps this
    combination working rather than raising. README calls it line-only and
    task 11 rejects it at the CLI -- until then it must not crash."""
    assert _bst_line(_user_prompt("DESC", "single", "pseudo")) == "BST target: ~300."


def test_pseudo_line_is_unchanged():
    assert (
        _bst_line(_user_prompt("DESC", "line", "pseudo"))
        == "BST targets: stage 1 ~300, stage 2 ~420, stage 3 ~600."
    )


# --- size defaults ------------------------------------------------------------

def test_two_stage_final_takes_the_stage_three_size_row():
    """A 2-stage stage 2 is a final form. Reusing the 3-stage middle row
    (10 dm / 150 hg) would under-size it."""
    assert _size_defaults({"stage": 2}, "line", "standard", 2) == (17, 600)


def test_two_stage_base_takes_the_stage_one_size_row():
    assert _size_defaults({"stage": 1}, "line", "standard", 2) == (5, 30)


@pytest.mark.parametrize("stage_no, expected", [
    (1, (5, 30)), (2, (10, 150)), (3, (17, 600)),
])
def test_three_stage_size_defaults_are_unchanged(stage_no, expected):
    assert _size_defaults({"stage": stage_no}, "line", "standard", 3) == expected
    assert _size_defaults({"stage": stage_no}, "line", "standard") == expected


def test_off_spec_stage_number_still_falls_back_with_a_stage_count():
    """The #55 fallback must survive the new parameter."""
    assert _size_defaults({"stage": 9}, "line", "standard", 2) == (10, 150)
    assert _size_defaults({"stage": "x"}, "line", "standard", 2) == (10, 150)


def test_generate_fakemon_applies_two_stage_size_defaults():
    stages = [
        {k: v for k, v in _STAGE_1.items()},
        {**_STAGE_2, "stage": 2},
    ]
    client = _make_client(json.dumps(stages))
    result = generate_fakemon("fire lizard", "line", client=client, stages=2)
    assert [(s["height_dm"], s["weight_hg"]) for s in result] == [(5, 30), (17, 600)]


# --- the public keyword -------------------------------------------------------

def test_generate_fakemon_accepts_stages_keyword():
    client = _make_client(json.dumps([_STAGE_1, _STAGE_2]))
    generate_fakemon("fire lizard", "line", client=client, stages=2)
    text = _get_prompt_text(client)
    assert "BST targets: stage 1 ~305, stage 2 ~468." in text


def test_generate_fakemon_defaults_to_three_stages():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", client=client)
    text = _get_prompt_text(client)
    assert "BST targets: stage 1 ~295, stage 2 ~405, stage 3 ~518." in text


def test_stages_is_keyword_only():
    """Positional passing must not silently land on `tier`."""
    client = _make_client(json.dumps(_LINE))
    with pytest.raises(TypeError):
        generate_fakemon("fire lizard", "line", "standard", 2, client=client)


# --------------------------------------------------------------------------
# Pokedex entry: charset + display budget (Gen 3 text contract)
# --------------------------------------------------------------------------

def test_repair_entry_folds_typographic_punctuation_to_ascii():
    from fakemon_forge.generator import _repair_entry
    out = _repair_entry("Glitchick\u2019s wings \u2014 a \u201cmess\u201d \u2013 of code\u2026")
    assert "\u2019" not in out and "\u201c" not in out and "\u2014" not in out
    assert out.startswith("Glitchick's wings - a \"mess\" - of code")


def test_repair_entry_drops_characters_outside_the_contract():
    from fakemon_forge.generator import _repair_entry, _ALLOWED_NAME_CHARS
    out = _repair_entry("Bug\u4e2dbit \u2603 glows")
    assert all(ch in _ALLOWED_NAME_CHARS for ch in out)
    assert "Bugbit" in out


def test_repair_entry_trims_to_the_display_budget_at_a_word_boundary():
    from fakemon_forge.generator import _repair_entry, _entry_fits_budget
    long_entry = (
        "Corrupto's body is a swirling mass of digital static and code "
        "fragments. It phases in and out of reality, leaving behind corrupted "
        "data trails. Entire networks shut down in its presence."
    )
    out = _repair_entry(long_entry)
    assert _entry_fits_budget(out)
    # Trimmed at a word boundary, not mid-word.
    assert long_entry.startswith(out.rstrip("."))


def test_repair_entry_leaves_a_conforming_entry_untouched():
    from fakemon_forge.generator import _repair_entry
    entry = "A small bug that clings to wires. It hums when startled."
    assert _repair_entry(entry) == entry


def test_entry_fits_budget_rejects_five_lines():
    from fakemon_forge.generator import _entry_fits_budget
    assert _entry_fits_budget("short one")
    assert not _entry_fits_budget(" ".join(["wordy"] * 60))


def test_normalize_repairs_the_pokedex_entry():
    from fakemon_forge.generator import _normalize
    stages = [{
        "name": "Bugbit", "types": ["Bug"],
        "pokedex_entry": "It\u2019s a bug.",
        "base_stats": {}, "abilities_gen3": [],
    }]
    out = _normalize(stages, "line", "standard", 1)
    assert out[0]["pokedex_entry"] == "It's a bug."


# ---------------------------------------------------------------------------
# Deterministic BST grounding (issue #85)
#
# The model owns the stat *shape*; these establish that the code owns the
# magnitude. #59 put "BST target: ~430." in the prompt and nothing enforced it:
# eleven standard single forms all prompted with ~430 averaged 337, with ten of
# the eleven undershooting.
# ---------------------------------------------------------------------------

# The under-scaled shape from a real run: sums to 330 against a 430 target.
_UNDERSCALED = {
    "hp": 60, "attack": 45, "defense": 70,
    "sp_atk": 50, "sp_def": 65, "speed": 40,
}


def _bst(stats: dict) -> int:
    return sum(stats[key] for key in _STAT_KEYS)


# --- the total is imposed, not requested ------------------------------------

@pytest.mark.parametrize("target", [205, 295, 336, 430, 500, 518, 580, 600])
def test_rescaled_stats_sum_to_exactly_the_target(target):
    """Exactly, not approximately. Largest-remainder apportionment is the whole
    reason this lands on 430 rather than 429 or 431."""
    assert _bst(_normalize_base_stats(_UNDERSCALED, target)) == target


def test_an_undershooting_line_is_raised_to_its_target():
    """The measured failure, directly: 330 in, 430 out."""
    assert sum(_UNDERSCALED.values()) == 330
    assert _bst(_normalize_base_stats(_UNDERSCALED, 430)) == 430


def test_an_overshooting_line_is_lowered_to_its_target():
    """The bias runs one way in practice, but the mechanism is not one-way."""
    inflated = {key: 150 for key in _STAT_KEYS}
    assert _bst(_normalize_base_stats(inflated, 430)) == 430


def test_a_line_already_on_target_is_left_alone():
    """Rescaling is idempotent on a conforming stat line \u2014 it must not jitter
    stats that already sum correctly."""
    on_target = {"hp": 80, "attack": 70, "defense": 60,
                 "sp_atk": 90, "sp_def": 70, "speed": 60}
    assert sum(on_target.values()) == 430
    assert _normalize_base_stats(on_target, 430) == on_target


# --- the shape is preserved --------------------------------------------------

def test_rescaling_preserves_the_ranking_of_the_stats():
    """A glass cannon stays a glass cannon. This is the model's actual
    contribution and the reason the fix is a rescale rather than a lookup."""
    out = _normalize_base_stats(_UNDERSCALED, 500)
    ranked = sorted(_STAT_KEYS, key=lambda k: _UNDERSCALED[k], reverse=True)
    assert sorted(_STAT_KEYS, key=lambda k: out[k], reverse=True) == ranked


def test_rescaling_preserves_proportions_within_rounding():
    """Each stat lands within a point of its exact proportional share."""
    target = 430
    out = _normalize_base_stats(_UNDERSCALED, target)
    total = sum(_UNDERSCALED.values())
    for key in _STAT_KEYS:
        assert abs(out[key] - _UNDERSCALED[key] * target / total) < 1


def test_a_flat_stat_line_stays_flat():
    flat = {key: 50 for key in _STAT_KEYS}
    out = _normalize_base_stats(flat, 432)
    assert set(out.values()) == {72}


def test_float_stats_are_accepted_as_weights():
    """The prompt asks for integers, but a model that returns 60.5 has still
    expressed a usable shape \u2014 that is not a reason to discard it."""
    out = _normalize_base_stats({key: 60.5 for key in _STAT_KEYS}, 300)
    assert _bst(out) == 300


# --- the byte range ----------------------------------------------------------

def test_a_degenerate_spike_clamps_at_255_and_still_sums_to_target():
    """Five 1s and a 250 against a 600 target would scale the spike past the
    Gen 3 byte. The clamp redistributes rather than truncating the total."""
    spiky = {"hp": 1, "attack": 1, "defense": 1,
             "sp_atk": 1, "sp_def": 1, "speed": 250}
    out = _normalize_base_stats(spiky, 600)
    assert max(out.values()) == _STAT_MAX
    assert _bst(out) == 600


def test_a_zero_stat_is_raised_to_the_minimum():
    """A base stat of 0 makes the Gen 3 damage formula degenerate."""
    out = _normalize_base_stats({**_UNDERSCALED, "speed": 0}, 430)
    assert out["speed"] == _STAT_MIN
    assert _bst(out) == 430


@pytest.mark.parametrize("target", [6, 205, 430, 600, 1530])
def test_every_stat_lands_inside_the_byte_range(target):
    spiky = {"hp": 1, "attack": 0, "defense": 3,
             "sp_atk": 0, "sp_def": 900, "speed": 2}
    out = _normalize_base_stats(spiky, target)
    assert all(_STAT_MIN <= v <= _STAT_MAX for v in out.values())
    assert _bst(out) == target


@pytest.mark.parametrize("target", [-50, 0, 5, 99999])
def test_an_out_of_range_target_is_bounded_to_what_six_bytes_can_hold(target):
    """No `_BST_TARGETS` value can reach here, but the function stays total \u2014
    the redistribution loop needs a target it can actually satisfy."""
    out = _normalize_base_stats(_UNDERSCALED, target)
    assert all(_STAT_MIN <= v <= _STAT_MAX for v in out.values())
    assert _STAT_MIN * 6 <= _bst(out) <= _STAT_MAX * 6


# --- the degenerate guard ----------------------------------------------------

@pytest.mark.parametrize("raw", [
    None,
    {},
    "60/45/70/50/65/40",
    [60, 45, 70, 50, 65, 40],
    {"hp": 60, "attack": 45},                              # missing keys
    {**_UNDERSCALED, "speed": "fast"},                     # non-numeric
    {**_UNDERSCALED, "speed": None},
    {**_UNDERSCALED, "speed": True},                       # bool, not an int
    {**_UNDERSCALED, "speed": -40},                        # negative weight
    {key: 0 for key in _STAT_KEYS},                        # sum <= 0
])
def test_unusable_stats_fall_back_to_an_even_split(raw):
    """Malformed input carries no shape worth preserving. The target is still
    met, so a downstream export never sees a missing or half-filled stat line."""
    out = _normalize_base_stats(raw, 432)
    assert out == {key: 72 for key in _STAT_KEYS}


def test_the_even_split_still_sums_exactly_when_the_target_is_indivisible():
    out = _normalize_base_stats(None, 430)
    assert _bst(out) == 430
    assert sorted(out.values()) == [71, 71, 72, 72, 72, 72]


def test_the_result_always_has_exactly_the_six_gen3_stats():
    out = _normalize_base_stats({**_UNDERSCALED, "luck": 99}, 430)
    assert set(out) == set(_STAT_KEYS)


@pytest.mark.parametrize("raw, expected", [
    ({key: 10 for key in _STAT_KEYS}, [10] * 6),
    ({**{key: 10 for key in _STAT_KEYS}, "hp": 10.9}, [10] * 6),
    (None, None),
    ({"hp": 1}, None),
    ({**{key: 10 for key in _STAT_KEYS}, "hp": True}, None),
    ({**{key: 10 for key in _STAT_KEYS}, "hp": -1}, None),
])
def test_stat_weights_reads_or_rejects(raw, expected):
    assert _stat_weights(raw) == expected


# --- apportionment -----------------------------------------------------------

def test_apportion_hands_the_leftover_to_the_largest_remainders():
    """Three equal weights over 100: two get 33, one gets 34. Which one is
    settled by remainder then by index, never by dict order."""
    assert _apportion([1, 1, 1], 100) == [34, 33, 33]


def test_apportion_is_exact_for_any_split():
    for target in range(6, 200):
        assert sum(_apportion([3, 1, 4, 1, 5, 9], target)) == target


def test_apportion_of_nothing_is_nothing():
    """The terminating case of the clamp loop, once every stat is pinned."""
    assert _apportion([], 430) == []


# --- deterministic target selection ------------------------------------------

def test_the_same_name_always_picks_the_same_total():
    """The reproducibility property run.json (#81) was added to record."""
    band = (336, 430, 500)
    first = _bst_target(band, _name_position("Skyship"))
    for _ in range(5):
        assert _bst_target(band, _name_position("Skyship")) == first


def test_different_names_scatter_across_the_band():
    """Not a flat median: the point of band-picking is that standalone species
    genuinely spread 336-500, and every standard single form being 430 would be
    less faithful to Gen 3 than the scatter it replaces."""
    band = (336, 430, 500)
    names = ["Cheezit", "Libuff", "Alphadile", "Gourdle", "Dozlet", "Brixpuz",
             "Grinweb", "Calljet", "Skyship", "Ballast", "Flamburr", "Bugbit"]
    picked = {_bst_target(band, _name_position(name)) for name in names}
    assert len(picked) > 6


def test_a_flat_band_picks_its_single_value():
    for position in (0, 1, _HASH_SPACE // 2, _HASH_SPACE - 1):
        assert _bst_target((580, 580, 580), position) == 580


def test_name_position_stays_inside_the_hash_space():
    for name in ("", "A", "Skyship", "Zzzzzzzzzz", "Fl\u00e9ur\u2640"):
        assert 0 <= _name_position(name) < _HASH_SPACE


# --- band selection per stage ------------------------------------------------

def test_single_mode_takes_the_standalone_band_whatever_count_is_passed():
    """A single form is a standalone species, not a juvenile (issue #48)."""
    band = _stage_band({"stage": 1}, "single", "standard", 3, 0)
    assert band[_MEDIAN] == 430


@pytest.mark.parametrize("stage_number, expected_median", [(1, 295), (2, 405), (3, 518)])
def test_line_mode_reads_the_stages_own_number(stage_number, expected_median):
    band = _stage_band({"stage": stage_number}, "line", "standard", 3, 0)
    assert band[_MEDIAN] == expected_median


@pytest.mark.parametrize("stage_value", [None, "two", 0, 9, {}])
def test_an_off_spec_stage_number_falls_back_to_list_position(stage_value):
    """`_size_defaults` degrades the same way rather than raising KeyError on a
    stage number the model got wrong."""
    band = _stage_band({"stage": stage_value}, "line", "standard", 3, 1)
    assert band[_MEDIAN] == 405


def test_more_stages_than_the_row_has_falls_back_to_the_last_band():
    """The model may return more stages than were asked for, which `_normalize`
    accepts. Indexing off the end of the row must not raise."""
    band = _stage_band({"stage": 7}, "line", "standard", 2, 5)
    assert band[_MEDIAN] == 468


# --- end to end through _normalize -------------------------------------------

def _stage(name, number, stats=None):
    return {
        "name": name, "stage": number, "types": ["Fire"],
        "base_stats": dict(stats or _UNDERSCALED), "abilities_gen3": [],
    }


def test_normalize_grounds_a_single_form_inside_the_standalone_band():
    out = _normalize([_stage("Cheezit", 1)], "single", "standard", 1)
    assert 336 <= _bst(out[0]["base_stats"]) <= 500


def test_normalize_makes_a_line_ascend_even_when_the_model_did_not():
    """The guarantee #85 called out as absent: nothing stopped the model
    returning a stage 2 weaker than its stage 1. Here it does exactly that."""
    backwards = [
        _stage("Flamburr", 1, {key: 90 for key in _STAT_KEYS}),    # 540
        _stage("Flamburro", 2, {key: 30 for key in _STAT_KEYS}),   # 180
        _stage("Flamburron", 3, {key: 50 for key in _STAT_KEYS}),  # 300
    ]
    totals = [_bst(s["base_stats"]) for s in _normalize(backwards, "line", "standard", 3)]
    assert totals == sorted(totals)
    assert len(set(totals)) == 3


def test_a_line_is_seeded_from_its_first_stage_name():
    """One quantile for the whole line, taken from the line's name \u2014 the same
    reading main.py uses when it seeds cries off `forms[0]["name"]`. Renaming a
    later stage must not move any stage's total."""
    original = _normalize(
        [_stage("Flamburr", 1), _stage("Flamburro", 2), _stage("Flamburron", 3)],
        "line", "standard", 3,
    )
    renamed = _normalize(
        [_stage("Flamburr", 1), _stage("Zzz", 2), _stage("Qqq", 3)],
        "line", "standard", 3,
    )
    assert [_bst(s["base_stats"]) for s in original] == \
           [_bst(s["base_stats"]) for s in renamed]


def test_normalize_adds_base_stats_when_the_model_omitted_them():
    """export_ini requires the field; an even split beats a KeyError there."""
    stage = {"name": "Bugbit", "stage": 1, "types": ["Bug"], "abilities_gen3": []}
    out = _normalize([stage], "single", "standard", 1)
    assert set(out[0]["base_stats"]) == set(_STAT_KEYS)
    assert 336 <= _bst(out[0]["base_stats"]) <= 500


def test_normalize_uses_the_repaired_name_as_the_seed():
    """An over-long or illegal name is repaired before it seeds the position,
    so the BST matches the name actually written out."""
    long_name = _normalize([_stage("Flamburrific!!", 1)], "single", "standard", 1)
    repaired = _normalize([_stage("Flamburrif", 1)], "single", "standard", 1)
    assert long_name[0]["name"] == "Flamburrif"
    assert _bst(long_name[0]["base_stats"]) == _bst(repaired[0]["base_stats"])


def test_normalize_of_no_stages_does_not_raise():
    assert _normalize([], "line", "standard", 3) == []


def test_generate_fakemon_returns_stats_on_the_band():
    """Through the real entry point, with the model returning its measured
    under-scaled shape."""
    client = _make_client(json.dumps([{**_STAGE_1, "base_stats": dict(_UNDERSCALED)}]))
    result = generate_fakemon("fire lizard", "single", client=client)
    assert 336 <= _bst(result[0]["base_stats"]) <= 500


@pytest.mark.parametrize("tier, expected", [("legendary", 580), ("mythical", 600)])
def test_generate_fakemon_pins_the_flat_tiers_exactly(tier, expected):
    """legendary and mythical have no observed spread, so the enforced total is
    the number the prompt names \u2014 for every name."""
    client = _make_client(json.dumps([{**_STAGE_1, "base_stats": dict(_UNDERSCALED)}]))
    result = generate_fakemon("a storm", "single", tier=tier, client=client)
    assert _bst(result[0]["base_stats"]) == expected


# --- the two invariants, swept ------------------------------------------------

def _fuzz_stat_lines(count: int):
    """Deterministically seeded stat lines, weighted towards the shapes that
    actually stress the clamp: zeros, lone spikes, and near-flat lines."""
    rng = random.Random(20260804)
    for trial in range(count):
        shape = trial % 4
        if shape == 0:
            values = [rng.randint(0, 200) for _ in _STAT_KEYS]
        elif shape == 1:
            values = [rng.choice([0, 0, 1, 255, 999]) for _ in _STAT_KEYS]
        elif shape == 2:
            values = [rng.randint(1, 5) for _ in _STAT_KEYS]
            values[rng.randrange(len(_STAT_KEYS))] = rng.randint(500, 5000)
        else:
            values = [rng.randint(0, 3) for _ in _STAT_KEYS]
        yield dict(zip(_STAT_KEYS, values)), rng.randint(6, _STAT_MAX * 6)


def test_the_total_and_the_byte_range_hold_across_a_fuzz_sweep():
    """The two invariants everything downstream depends on, over stat lines no
    hand-written case would think to try.

    This is not decoration: a sweep like it is what caught the clamp handing
    back 1022 against a 1530 target, because a zero-weight stat pinned at the
    minimum had consumed the headroom needed to reach it.
    """
    for raw, target in _fuzz_stat_lines(3000):
        out = _normalize_base_stats(raw, target)
        assert _bst(out) == target, (raw, target, out)
        assert all(_STAT_MIN <= v <= _STAT_MAX for v in out.values()), (raw, target, out)
