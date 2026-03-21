"""
Music Generator
===============
Clean abstraction so we can swap Phase 1 (udioapi.pro) for
Phase 2 (local MusicGen) with zero changes to the rest of the app.

All implementations return raw MP3 bytes.
"""

import os
import time
import logging
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MusicGeneratorBase(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> bytes:
        """
        Given a text prompt, return MP3 audio bytes.
        Blocks until audio is ready (may take 20-90 seconds).
        Raises RuntimeError on failure.
        """
        ...


# ── Phase 1: udioapi.pro ──────────────────────────────────────────────────────

class UdioApiGenerator(MusicGeneratorBase):
    """
    Uses udioapi.pro — a third-party Suno/Udio wrapper.

    Setup:
      1. Sign up at https://udioapi.pro
      2. Get your API key from the dashboard (free tier available)
      3. Add to .env:  UDIO_API_KEY=your_key_here

    Costs: free tier gives ~10 generations/day, paid plans from ~$5/mo.
    This is Phase 1 only — swap for MusicGenGenerator before publishing.
    """

    BASE_URL = "https://udioapi.pro/api/v2"
    POLL_INTERVAL = 5   # seconds between status checks
    MAX_WAIT = 180      # give up after 3 minutes

    def __init__(self):
        self.api_key = os.getenv("UDIO_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "UDIO_API_KEY not set. Add it to your .env file.\n"
                "Get a key at https://udioapi.pro"
            )

    def generate(self, prompt: str) -> bytes:
        logger.info(f"Requesting generation: '{prompt}'")

        # ── Submit generation job ─────────────────────────────────────────
        response = requests.post(
            f"{self.BASE_URL}/generate",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "gpt_description_prompt": prompt,
                "make_instrumental": True,   # no AI vocals for dancing — cleaner
                "model": "chirp-v3-5",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {data}")

        logger.info(f"Generation task submitted: {task_id}")

        # ── Poll until done ───────────────────────────────────────────────
        elapsed = 0
        while elapsed < self.MAX_WAIT:
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

            status_resp = requests.get(
                f"{self.BASE_URL}/get",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"task_id": task_id},
                timeout=15,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            state = status_data.get("status", "").lower()
            logger.debug(f"Task {task_id} status: {state} ({elapsed}s elapsed)")

            if state in ("complete", "completed", "success"):
                audio_url = (
                    status_data.get("audio_url")
                    or status_data.get("url")
                    or (status_data.get("data", [{}])[0] or {}).get("audio_url")
                )
                if not audio_url:
                    raise RuntimeError(f"No audio_url in completed response: {status_data}")

                logger.info(f"Downloading audio from {audio_url}")
                audio_resp = requests.get(audio_url, timeout=60)
                audio_resp.raise_for_status()
                return audio_resp.content

            if state in ("failed", "error"):
                raise RuntimeError(f"Generation failed: {status_data}")

            # Still pending — keep polling
            logger.debug(f"Still generating... ({elapsed}s)")

        raise RuntimeError(f"Generation timed out after {self.MAX_WAIT}s")


# ── Phase 2: Local MusicGen (Meta, fully open source) ────────────────────────

class MusicGenGenerator(MusicGeneratorBase):
    """
    Local music generation using Meta's MusicGen model.
    Fully open source (Apache 2.0), no API keys, no cloud dependency.
    Publishable to HF app store without any restrictions.

    Install:
        pip install audiocraft   # Meta's library
        # or: pip install transformers accelerate

    Hardware:
        - Mac (MPS): ~15-30s for a 10s clip  ← recommended
        - Linux (CUDA GPU): ~5-15s
        - CPU only: ~2-4 min (painful but works)
        - RPi 5 (Reachy Wireless): too slow, run on laptop instead

    Quality:
        - Instrumental: excellent, especially for clear genre prompts
        - Vocals: not supported (MusicGen is instrumental only)
        - For a dance app, this is actually perfect

    TODO: uncomment and test once you're ready for Phase 2
    """

    def __init__(self, model_size: str = "small"):
        # model_size options: "small" (300M), "medium" (1.5B), "large" (3.3B)
        # "small" is the sweet spot for speed vs quality on a laptop
        raise NotImplementedError(
            "MusicGenGenerator is Phase 2 — not yet implemented.\n"
            "When ready:\n"
            "  1. pip install audiocraft\n"
            "  2. Uncomment the implementation below\n"
            "  3. In main.py, swap UdioApiGenerator() for MusicGenGenerator()"
        )

    # ── Uncomment for Phase 2 ────────────────────────────────────────────
    # def __init__(self, model_size: str = "small"):
    #     from audiocraft.models import MusicGen
    #     import torch
    #     logger.info(f"Loading MusicGen-{model_size}...")
    #     self.model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}")
    #     self.model.set_generation_params(duration=30)  # 30 second songs
    #     logger.info("MusicGen ready.")
    #
    # def generate(self, prompt: str) -> bytes:
    #     import torch, io
    #     from audiocraft.data.audio import audio_write
    #     import soundfile as sf
    #
    #     logger.info(f"Generating with MusicGen: '{prompt}'")
    #     wav = self.model.generate([prompt])  # returns tensor [1, channels, samples]
    #
    #     # Convert tensor → MP3 bytes in memory
    #     wav_np = wav[0].cpu().numpy().T  # shape: [samples, channels]
    #     sample_rate = self.model.sample_rate
    #
    #     buf = io.BytesIO()
    #     sf.write(buf, wav_np, sample_rate, format="mp3")
    #     buf.seek(0)
    #     return buf.read()

    def generate(self, prompt: str) -> bytes:
        raise NotImplementedError("See comment above")
