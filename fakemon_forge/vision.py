import base64
import sys
from pathlib import Path

from mistralai.client import Mistral

_VISION_MODEL = "pixtral-large-latest"

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

_PROMPT = (
    "Describe this creature's appearance, colors, body shape, and any notable "
    "features in plain English. Be specific about visual details."
)


def describe_image(image_path: str, *, api_key: str = None, client=None) -> str:
    """Call the Mistral vision model and return a plain-English description."""
    if client is None:
        client = Mistral(api_key=api_key)

    path = Path(image_path)
    mime = _MIME.get(path.suffix.lower(), "image/png")
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    data_url = f"data:{mime};base64,{b64}"

    try:
        response = client.chat.complete(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:
        print(
            f"Error: Mistral vision call failed ({exc}). "
            "Check that MISTRAL_API_KEY is set and valid.",
            file=sys.stderr,
        )
        sys.exit(1)

    return response.choices[0].message.content
