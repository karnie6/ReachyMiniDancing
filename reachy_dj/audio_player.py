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
  - Sample rate: queried from mini.media.get_output_audio_samplerate()
  - Format: float32, mono
  - Delivered via: mini.media.push_audio_sample(chunk)  # one arg only
"""

import logging
import threading
import time
import numpy as np

from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

CHUNK_DURATION_SEC = 0.1   # push 100ms chunks — smooth without hammering the API
VOLUME_SCALE = 0.5         # scale to 50% volume


def stream_mp3_to_reachy(
    reachy_mini: ReachyMini,
    mp3_bytes: bytes,
    stop_event: threading.Event | None = None,
) -> None:
    """
    Decode mp3_bytes and stream to Reachy's speaker in small chunks.
    Blocks until the full audio has been pushed or stop_event is set.
    """
    audio, sample_rate = _decode_mp3(mp3_bytes)

    reachy_mini.media.start_playing()
    target_rate = reachy_mini.media.get_output_audio_samplerate()

    if sample_rate != target_rate:
        audio = _resample(audio, sample_rate, target_rate)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Volume control — scale down to 30%
    audio = audio * VOLUME_SCALE

    chunk_size = int(target_rate * CHUNK_DURATION_SEC)
    pushed = 0

    logger.info(f"Streaming {len(audio) / target_rate:.1f}s of audio to Reachy...")

    while pushed < len(audio):
        if stop_event is not None and stop_event.is_set():
            logger.info("Audio stream interrupted by stop_event.")
            break
        chunk = audio[pushed : pushed + chunk_size]
        reachy_mini.media.push_audio_sample(chunk)  # ONE arg only, no sample_rate
        time.sleep(CHUNK_DURATION_SEC)
        pushed += len(chunk)

    reachy_mini.media.stop_playing()
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

        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes))
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
