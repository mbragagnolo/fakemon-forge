import json
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
