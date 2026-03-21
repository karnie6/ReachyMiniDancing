"""
Voice Listener
==============
Records from Reachy's mic and transcribes with Whisper (local, no cloud).
Returns the raw transcribed text — no command parsing needed since
the full prompt goes straight to the music generator.

Install: pip install openai-whisper
"""

import logging
import numpy as np
from typing import Optional

from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

_whisper_model = None  # Lazy-loaded on first use


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper tiny.en model (~40MB, first run only)...")
        _whisper_model = whisper.load_model("tiny.en")
        logger.info("Whisper ready.")
    return _whisper_model


def listen_for_command(reachy_mini: ReachyMini, timeout_seconds: int = 10) -> Optional[str]:
    """
    Record audio from Reachy's mic, return transcribed text.
    Returns None if silence or transcription fails.
    """
    try:
        audio: np.ndarray = reachy_mini.microphones.record(
            duration=timeout_seconds,
            stop_on_silence=True,
            silence_threshold=0.01,
        )

        if audio is None or len(audio) < 2000:
            return None

        model = _get_model()
        result = model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()

        return text.lower() if text else None

    except Exception as e:
        logger.error(f"Voice listen error: {e}")
        return None
