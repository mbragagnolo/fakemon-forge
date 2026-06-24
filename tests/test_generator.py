import json
import pytest
from unittest.mock import MagicMock, patch

from fakemon_forge.generator import generate_fakemon

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
