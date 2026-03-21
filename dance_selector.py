"""
Dance Selector
==============
Uses a small LLM call (GPT-4o-mini — very cheap, ~$0.001 per pick)
to choose dance moves from the library that match the vibe of the prompt.

Example:
  prompt: "heavy metal thrash"
  → ["dizzy_spin", "neck_recoil", "chicken_peck", "sharp_side_tilt"]

  prompt: "slow jazz late night"
  → ["pendulum_swing", "side_to_side_sway", "chin_lead"]

Falls back to a curated random selection if the API call fails.
"""

import os
import json
import random
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# All 19 available moves from reachy_mini_dances_library
ALL_MOVES = [
    "simple_nod",
    "head_tilt_roll",
    "side_to_side_sway",
    "dizzy_spin",
    "stumble_and_recover",
    "interwoven_spirals",
    "sharp_side_tilt",
    "side_peekaboo",
    "yeah_nod",
    "uh_huh_tilt",
    "neck_recoil",
    "chin_lead",
    "groovy_sway_and_roll",
    "chicken_peck",
    "side_glance_flick",
    "polyrhythm_combo",
    "grid_snap",
    "pendulum_swing",
    "jackson_square",
]

# Handcrafted fallback sets by energy level — used if LLM call fails
FALLBACK_MOVES = {
    "high_energy": [
        "dizzy_spin", "chicken_peck", "neck_recoil",
        "sharp_side_tilt", "polyrhythm_combo", "stumble_and_recover",
    ],
    "mid_energy": [
        "groovy_sway_and_roll", "yeah_nod", "side_peekaboo",
        "head_tilt_roll", "jackson_square", "grid_snap",
    ],
    "low_energy": [
        "pendulum_swing", "side_to_side_sway", "chin_lead",
        "simple_nod", "uh_huh_tilt", "side_glance_flick",
    ],
}

SYSTEM_PROMPT = """You are a choreographer for a small desktop robot called Reachy Mini.
Given a music prompt, pick 4-6 dance moves from the available list that best match the vibe.
Respond ONLY with a valid JSON array of move names, nothing else.
Example: ["groovy_sway_and_roll", "yeah_nod", "side_peekaboo", "chin_lead"]"""


def pick_dances_for_prompt(prompt: str) -> list[str]:
    """
    Returns an ordered list of dance moves that match the prompt's vibe.
    The dance loop will cycle through these while music plays.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        try:
            return _llm_pick(prompt, api_key)
        except Exception as e:
            logger.warning(f"LLM dance selection failed, using fallback: {e}")

    return _fallback_pick(prompt)


def _llm_pick(prompt: str, api_key: str) -> list[str]:
    """Ask GPT-4o-mini to pick moves. Costs ~$0.001 per call."""
    import requests

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Music prompt: \"{prompt}\"\n\n"
                        f"Available moves: {json.dumps(ALL_MOVES)}"
                    ),
                },
            ],
        },
        timeout=10,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"].strip()

    # Strip any accidental markdown fences
    content = content.replace("```json", "").replace("```", "").strip()
    moves = json.loads(content)

    # Validate — only keep names that actually exist
    valid = [m for m in moves if m in ALL_MOVES]
    if not valid:
        raise ValueError(f"LLM returned no valid moves: {moves}")

    logger.info(f"LLM selected dances: {valid}")
    return valid


def _fallback_pick(prompt: str) -> list[str]:
    """
    Simple keyword heuristic when the LLM isn't available.
    Good enough for offline use / Phase 2.
    """
    prompt_lower = prompt.lower()

    high_energy_keywords = [
        "metal", "rock", "punk", "thrash", "intense", "fast", "hard",
        "heavy", "aggressive", "edm", "dubstep", "drum and bass", "rave",
    ]
    low_energy_keywords = [
        "slow", "jazz", "ambient", "chill", "soft", "gentle", "relaxing",
        "lofi", "lo-fi", "ballad", "acoustic", "peaceful", "calm",
    ]

    if any(k in prompt_lower for k in high_energy_keywords):
        pool = FALLBACK_MOVES["high_energy"]
    elif any(k in prompt_lower for k in low_energy_keywords):
        pool = FALLBACK_MOVES["low_energy"]
    else:
        pool = FALLBACK_MOVES["mid_energy"]

    # Return a shuffled subset of 4-5 moves
    selected = random.sample(pool, min(4, len(pool)))
    logger.info(f"Fallback selected dances: {selected}")
    return selected
