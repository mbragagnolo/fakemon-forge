import base64
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


from fakemon_forge.vision import describe_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(response_text="a small fire lizard with three tails"):
    """Return a mock Mistral client whose chat.complete returns response_text."""
    choice = MagicMock()
    choice.message.content = response_text

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_text_from_model(tmp_path):
    img = tmp_path / "creature.png"
    img.write_bytes(b"\x89PNG\r\n")

    client = _make_client("a chubby blue turtle with a cannon on its back")
    result = describe_image(str(img), client=client)

    assert result == "a chubby blue turtle with a cannon on its back"


def test_calls_correct_model(tmp_path):
    img = tmp_path / "creature.png"
    img.write_bytes(b"\x89PNG\r\n")

    client = _make_client()
    describe_image(str(img), client=client)

    call_kwargs = client.chat.complete.call_args.kwargs
    assert call_kwargs["model"] == "pixtral-large-latest"


def test_sends_image_as_base64_data_url_png(tmp_path):
    img = tmp_path / "creature.png"
    raw = b"\x89PNG\r\n\x1a\n"
    img.write_bytes(raw)

    client = _make_client()
    describe_image(str(img), client=client)

    messages = client.chat.complete.call_args.kwargs["messages"]
    image_block = next(
        b for b in messages[0]["content"] if b.get("type") == "image_url"
    )
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_sends_image_as_base64_data_url_jpeg(tmp_path):
    img = tmp_path / "photo.jpg"
    raw = b"\xff\xd8\xff"
    img.write_bytes(raw)

    client = _make_client()
    describe_image(str(img), client=client)

    messages = client.chat.complete.call_args.kwargs["messages"]
    image_block = next(
        b for b in messages[0]["content"] if b.get("type") == "image_url"
    )
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_jpeg_extension_also_uses_jpeg_mime(tmp_path):
    img = tmp_path / "photo.jpeg"
    img.write_bytes(b"\xff\xd8\xff")

    client = _make_client()
    describe_image(str(img), client=client)

    messages = client.chat.complete.call_args.kwargs["messages"]
    image_block = next(
        b for b in messages[0]["content"] if b.get("type") == "image_url"
    )
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_message_includes_text_prompt(tmp_path):
    img = tmp_path / "creature.png"
    img.write_bytes(b"\x89PNG")

    client = _make_client()
    describe_image(str(img), client=client)

    messages = client.chat.complete.call_args.kwargs["messages"]
    text_block = next(
        b for b in messages[0]["content"] if b.get("type") == "text"
    )
    assert text_block["text"]  # non-empty prompt


def test_exits_on_auth_error(tmp_path):
    img = tmp_path / "creature.png"
    img.write_bytes(b"\x89PNG")

    client = MagicMock()
    client.chat.complete.side_effect = Exception("Unauthorized")

    with pytest.raises(SystemExit) as exc:
        describe_image(str(img), client=client)

    assert exc.value.code == 1


def test_auth_error_message_mentions_env_var(tmp_path, capsys):
    img = tmp_path / "creature.png"
    img.write_bytes(b"\x89PNG")

    client = MagicMock()
    client.chat.complete.side_effect = Exception("401 Unauthorized")

    with pytest.raises(SystemExit):
        describe_image(str(img), client=client)

    err = capsys.readouterr().err
    assert "MISTRAL_API_KEY" in err


def test_build_client_from_api_key():
    """describe_image can build its own client when given api_key kwarg."""
    with patch("fakemon_forge.vision.Mistral") as MockMistral:
        fake_client = _make_client()
        MockMistral.return_value = fake_client

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG")
            path = f.name

        try:
            result = describe_image(path, api_key="test-key-123")
            MockMistral.assert_called_once_with(api_key="test-key-123")
            assert result == "a small fire lizard with three tails"
        finally:
            os.unlink(path)
