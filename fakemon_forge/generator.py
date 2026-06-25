import json
import sys

from mistralai.client import Mistral

_MODEL = "mistral-large-latest"

_BST_TARGETS = {
    "standard": {"stage1": 300, "stage2": 420, "stage3": 520},
    "pseudo":   {"stage1": 300, "stage2": 420, "stage3": 600},
    "legendary":{"stage1": 580},
    "mythical": {"stage1": 600},
}

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

All stage names and sprite prompts must share a clear thematic throughline.
Return ONLY the JSON array. No markdown fences, no explanation, no extra keys.\
"""

_EVO_PROGRESSION = """\

Evolutionary progression — each stage must look and feel visually distinct:
  Stage 1: juvenile/child form — small and simple, cute or curious expression, \
limited limbs or features, undeveloped power.
  Stage 2: adolescent/teenage form — noticeably larger, silhouette more defined, \
signature features emerging, power becoming apparent.
  Stage 3: adult/final form — fully developed, imposing presence, complex design \
with a different silhouette from stage 1, design complexity at its peak.\
"""

_TIER_NOTES = {
    "pseudo":    "\nThis is a pseudo-legendary line: the final form should rival legendary "
                 "Pokémon in visual impact and raw power.",
    "legendary": "\nThis is a legendary Pokémon: unique, awe-inspiring, and lore-significant. "
                 "It should feel like a force of nature.",
    "mythical":  "\nThis is a mythical Pokémon: mysterious, rarely seen, tied to ancient legend.",
}


def _user_prompt(description: str, mode: str, tier: str) -> str:
    targets = _BST_TARGETS[tier]

    if mode == "single":
        count = "one stage (stage 1 only)"
        bst_hint = f"BST target: ~{targets['stage1']}."
        evo_text = ""
    else:
        count = "three evolutionary stages (stages 1, 2, and 3)"
        bst_hint = (
            f"BST targets: stage 1 ~{targets['stage1']}, "
            f"stage 2 ~{targets['stage2']}, "
            f"stage 3 ~{targets['stage3']}."
        )
        evo_text = _EVO_PROGRESSION

    tier_note = _TIER_NOTES.get(tier, "")

    return (
        f"Generate {count} for a Fakemon based on this description:\n\n"
        f"{description}\n\n"
        f"{bst_hint}"
        f"{evo_text}"
        f"{tier_note}"
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_fakemon(
    description: str,
    mode: str,
    tier: str = "standard",
    *,
    client=None,
    api_key: str = None,
) -> list[dict]:
    if client is None:
        client = Mistral(api_key=api_key)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(description, mode, tier)},
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
