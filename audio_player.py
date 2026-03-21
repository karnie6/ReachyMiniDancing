"""
Audio Player
============
Decodes MP3 bytes and streams PCM chunks to Reachy's speaker
using push_audio_sample().

Why chunked streaming vs loading all at once:
  - push_audio_sample() is non-blocking — it returns immediately
  - Feeding chunks lets the dance loop and audio stay in sync
  - Avoids loading a full 2-minute song into memory before starting

Reachy Mini's speaker expects:
  - Sample rate: 44100 Hz (resampled here if needed)
  - Format: float32, mono
  - Delivered via: mini.media.push_audio_sample(chunk, sample_rate)
"""

import logging
import time
import numpy as np

from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 44100
CHUNK_DURATION_SEC = 0.1   # push 100ms chunks — smooth without hammering the API


def stream_mp3_to_reachy(reachy_mini: ReachyMini, mp3_bytes: bytes) -> None:
    """
    Decode mp3_bytes and stream to Reachy's speaker in small chunks.
    Blocks until the full audio has been pushed (song is done).
    """
    # Decode MP3 → numpy float32 PCM
    audio, sample_rate = _decode_mp3(mp3_bytes)

    # Resample to 44100 if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Push in chunks
    chunk_size = int(sample_rate * CHUNK_DURATION_SEC)
    total_samples = len(audio)
    pushed = 0

    logger.info(f"Streaming {total_samples / sample_rate:.1f}s of audio to Reachy...")

    while pushed < total_samples:
        chunk = audio[pushed : pushed + chunk_size]
        reachy_mini.media.push_audio_sample(chunk, sample_rate)
        pushed += len(chunk)
        # Sleep per chunk so this call blocks for the actual song duration
        # (push_audio_sample is non-blocking — without this we'd dump the
        # entire song into the buffer instantly and return immediately)
        time.sleep(CHUNK_DURATION_SEC)

    # Wait for the last chunk to finish playing
    time.sleep(CHUNK_DURATION_SEC * 2)
    logger.info("Audio stream complete.")


def _decode_mp3(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode MP3 bytes to float32 numpy array.
    Uses pydub (ffmpeg backend) — widely available, handles all MP3 variants.
    Falls back to soundfile if pydub isn't available.
    """
    try:
        from pydub import AudioSegment
        import io

        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)

        # Normalize to [-1.0, 1.0]
        samples /= float(2 ** (8 * seg.sample_width - 1))

        # If stereo, reshape to [samples, 2]
        if seg.channels == 2:
            samples = samples.reshape(-1, 2)

        return samples, seg.frame_rate

    except ImportError:
        # Fallback: soundfile (supports MP3 with libsndfile)
        import soundfile as sf
        import io
        audio, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
        return audio, sr


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Simple linear resample using scipy."""
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(orig_rate, target_rate)
    up = target_rate // g
    down = orig_rate // g

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float32)
    else:
        # Stereo: resample each channel
        ch0 = resample_poly(audio[:, 0], up, down)
        ch1 = resample_poly(audio[:, 1], up, down)
        return np.stack([ch0, ch1], axis=1).astype(np.float32)
