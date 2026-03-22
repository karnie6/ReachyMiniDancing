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


def create_generator(duration: int = 30) -> "MusicGeneratorBase":
    """
    Auto-selects the best available generator based on env vars:
      - SUNO_API_KEY set  → SunoGenerator       (best quality)
      - HF_API_TOKEN set  → HuggingFaceInferenceGenerator (no local model)
      - fallback          → MusicGenGenerator   (local, requires large download)
    """
    if os.getenv("SUNO_API_KEY"):
        logger.info("SUNO_API_KEY found — using Suno via sunoapi.org.")
        return SunoGenerator(duration=duration)
    if os.getenv("HF_API_TOKEN"):
        logger.info("HF_API_TOKEN found — using HuggingFace Inference API.")
        return HuggingFaceInferenceGenerator(duration=duration)
    logger.info("No API keys found — falling back to local MusicGen.")
    return MusicGenGenerator(duration=duration)


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


# ── Phase 1b: Suno via sunoapi.org (best quality, third-party wrapper) ────────

class SunoGenerator(MusicGeneratorBase):
    """
    High-quality music generation via Suno (sunoapi.org third-party wrapper).
    Suno doesn't have an official public API yet — sunoapi.org is the
    most stable third-party provider, same pattern as udioapi.pro.

    Setup:
      1. Sign up at sunoapi.org
      2. Add to .env:  SUNO_API_KEY=your_key_here

    Generates instrumental tracks (no AI vocals) — ideal for dancing.
    Suno typically returns two clip variants; we use the first one.
    """

    BASE_URL = "https://api.sunoapi.org/api/v1"
    POLL_INTERVAL = 5
    MAX_WAIT = 300  # Suno can take up to ~2 min

    def __init__(self, duration: int = 30):
        self.api_key = os.getenv("SUNO_API_KEY")
        if not self.api_key:
            raise EnvironmentError("SUNO_API_KEY not set. Add it to your .env file.")
        self.duration = duration

    def generate(self, prompt: str) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"Requesting Suno generation: '{prompt}'")
        response = requests.post(
            f"{self.BASE_URL}/generate",
            headers=headers,
            json={
                "prompt": prompt,
                "customMode": False,
                "instrumental": True,   # no vocals — cleaner for dancing
                "model": "V4_5ALL",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"No taskId in response: {data}")
        logger.info(f"Suno task submitted: {task_id}")

        # ── Poll until done ───────────────────────────────────────────────
        elapsed = 0
        while elapsed < self.MAX_WAIT:
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

            status_resp = requests.get(
                f"{self.BASE_URL}/generate/record-info",
                headers=headers,
                params={"taskId": task_id},
                timeout=15,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            state = status_data.get("data", {}).get("status", "").upper()
            logger.debug(f"Suno task {task_id}: {state} ({elapsed}s)")

            if state == "SUCCESS":
                clips = status_data.get("data", {}).get("response", {}).get("data", [])
                if not clips:
                    raise RuntimeError(f"No clips in completed response: {status_data}")
                audio_url = clips[0].get("audio_url")
                if not audio_url:
                    raise RuntimeError(f"No audio_url in clip: {clips[0]}")
                logger.info(f"Downloading Suno audio from {audio_url}")
                audio_resp = requests.get(audio_url, timeout=60)
                audio_resp.raise_for_status()
                return audio_resp.content  # MP3 bytes

            if state == "FAILED":
                raise RuntimeError(f"Suno generation failed: {status_data}")

        raise RuntimeError(f"Suno generation timed out after {self.MAX_WAIT}s")


# ── Phase 2a: HuggingFace Inference API (remote, no local model) ─────────────

class HuggingFaceInferenceGenerator(MusicGeneratorBase):
    """
    Remote MusicGen via HuggingFace Inference API.
    No local model download — inference runs on HF's servers.

    Setup:
      1. Get a token at huggingface.co/settings/tokens (free account works)
      2. Add to .env:  HF_API_TOKEN=hf_your_token_here

    Tiers:
      - Free: rate-limited, model may be cold (adds ~20s on first call)
      - PRO ($9/mo) or Serverless: faster, higher limits

    The API returns WAV bytes directly — no conversion needed.
    On cold start (503 + estimated_time), retries automatically.
    """

    API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    MAX_RETRIES = 10
    RETRY_BACKOFF = 5  # seconds between retries while model warms up

    def __init__(self, duration: int = 30):
        self.token = os.getenv("HF_API_TOKEN")
        if not self.token:
            raise EnvironmentError(
                "HF_API_TOKEN not set. Add it to your .env file.\n"
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.duration = duration
        # MusicGen generates ~50 tokens/sec of audio at 32kHz
        self.max_new_tokens = duration * 50

    def generate(self, prompt: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens},
        }

        logger.info(f"Requesting MusicGen via HF Inference API: '{prompt}'")

        for attempt in range(self.MAX_RETRIES):
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                logger.info("HF Inference API generation complete.")
                return response.content  # raw WAV bytes

            if response.status_code == 503:
                # Model is loading (cold start) — HF returns estimated wait time
                try:
                    wait = response.json().get("estimated_time", self.RETRY_BACKOFF)
                except Exception:
                    wait = self.RETRY_BACKOFF
                logger.info(f"Model warming up, retrying in {wait:.0f}s... (attempt {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(wait)
                continue

            response.raise_for_status()

        raise RuntimeError(f"HF Inference API failed after {self.MAX_RETRIES} retries")


# ── Phase 2b: Local MusicGen (Meta, fully open source) ───────────────────────

class MusicGenGenerator(MusicGeneratorBase):
    """
    Local music generation using Meta's MusicGen model.
    Fully open source (Apache 2.0), no API keys, no cloud dependency.

    Install:  pip install audiocraft

    Hardware:
        - Linux (CUDA GPU): ~5-15s
        - Mac (MPS): ~15-30s for a 30s clip
        - CPU only: ~2-4 min (works, just slow)

    model_size options: "small" (300M), "medium" (1.5B), "large" (3.3B)
    "small" is the sweet spot for speed vs quality on a laptop.
    """

    def __init__(self, model_size: str = "small", duration: int = 30):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        logger.info(f"Loading MusicGen-{model_size} (first run downloads ~300MB)...")
        model_id = f"facebook/musicgen-{model_size}"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        self.sample_rate = self.model.config.audio_encoder.sampling_rate  # 32000
        # tokens ≈ duration * 50 (MusicGen generates ~50 tokens/sec)
        self.max_new_tokens = duration * 50
        logger.info("MusicGen ready.")

    def generate(self, prompt: str) -> bytes:
        import io
        import soundfile as sf

        logger.info(f"Generating with MusicGen: '{prompt}'")
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
        audio_values = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # audio_values shape: [batch, channels, samples] → take first, squeeze to [samples]
        wav_np = audio_values[0, 0].cpu().numpy()

        buf = io.BytesIO()
        sf.write(buf, wav_np, self.sample_rate, format="WAV")
        buf.seek(0)
        logger.info("MusicGen generation complete.")
        return buf.read()
