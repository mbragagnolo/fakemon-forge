import json
import sys

from mistralai.client import Mistral

_MODEL = "mistral-large-latest"

_SYSTEM_PROMPT = """\
You are a Pokémon game designer. Generate Fakemon data as a JSON array.
Each element represents one evolutionary stage and must have exactly these fields:
  name          – portmanteau-style name (string)
  stage         – stage number as an integer (1, 2, or 3)
  types         – list of 1 or 2 type strings, e.g. ["Fire"] or ["Water", "Flying"]
  ability       – one ability name (string)
  base_stats    – object with integer values for: hp, attack, defense, sp_atk, sp_def, speed
  pokedex_entry – 2–3 sentence flavour text (string)
  sprite_prompt – detailed visual description for pixel-art generation (string)

BST targets: stage 1 ≈ 300, stage 2 ≈ 420, stage 3 ≈ 520.
All three stage names and sprite prompts must share a clear thematic throughline.
Return ONLY the JSON array. No markdown fences, no explanation, no extra keys.\
"""


def _user_prompt(description: str, mode: str) -> str:
    count = "one stage (stage 1 only)" if mode == "single" else "three evolutionary stages (stages 1, 2, and 3)"
    return f"Generate {count} for a Fakemon based on this description:\n\n{description}"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_fakemon(
    description: str,
    mode: str,
    *,
    client=None,
    api_key: str = None,
) -> list[dict]:
    if client is None:
        client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(description, mode)},
    ]

    raw = None
    for attempt in range(2):
        try:
            response = client.chat.complete(model=_MODEL, messages=messages)
            raw = response.choices[0].message.content
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            if attempt == 1:
                print(
                    f"Error: LLM returned malformed JSON after 2 attempts.\n"
                    f"Raw response:\n{raw}",
                    file=sys.stderr,
                )
                sys.exit(1)
        except Exception as exc:
            print(
                f"Error: Mistral API call failed ({exc}). "
                "Check that MISTRAL_API_KEY is set and valid.",
                file=sys.stderr,
            )
            sys.exit(1)
