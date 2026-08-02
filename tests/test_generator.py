import json
import pytest
from unittest.mock import MagicMock, patch

from fakemon_forge.generator import generate_fakemon, _normalize, _SYSTEM_PROMPT

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
    client = _make_client(json.dumps([_STAGE_1]))
    generate_fakemon("fire lizard", "single", tier="standard", client=client)
    text = _get_prompt_text(client)
    assert "300" in text


def test_standard_line_bst_includes_stage3_target():
    client = _make_client(json.dumps(_LINE))
    generate_fakemon("fire lizard", "line", tier="standard", client=client)
    text = _get_prompt_text(client)
    assert "520" in text


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
