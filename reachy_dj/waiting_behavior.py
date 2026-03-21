"""
Waiting Behavior
================
What Reachy does during the 20-90s music generation window.

Sequence:
  1. Play a short "beat building" WAV loop from assets/ (2-4 bar drum loop)
  2. Simultaneously do a slow "thinking" head bob + antenna wiggle
  3. Stop cleanly when generation_done event fires

The pre-bundled WAV means this works fully offline — no extra API calls.
The motion makes it look intentional, like a DJ cueing up a track.
"""

import threading
import time
import logging
import numpy as np
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

WAITING_BEAT_PATH = Path(__file__).parent.parent / "loading.wav"


class WaitingBehavior:
    def __init__(self, reachy_mini: ReachyMini):
        self.robot = reachy_mini

    def start(self, done_event: threading.Event) -> None:
        """
        Runs the full waiting routine, blocking until done_event is set.
        Spins up audio + motion in background threads, then waits.
        """
        stop = threading.Event()

        threads = [
            threading.Thread(target=self._play_beat_loop, args=(stop,), daemon=True),
            threading.Thread(target=self._thinking_motion, args=(stop,), daemon=True),
        ]
        for t in threads:
            t.start()

        # Block until music generation finishes
        done_event.wait()

        # Stop the waiting behavior
        stop.set()
        for t in threads:
            t.join(timeout=2)

        # Return head to neutral
        self.robot.goto_target(head=create_head_pose(), duration=0.4)
        logger.info("Waiting behavior stopped.")

    def _play_beat_loop(self, stop_event: threading.Event) -> None:
        """
        Loops loading.wav through Reachy's speaker until stopped.
        Pushes audio in 100ms chunks with matching sleeps so we don't
        flood the buffer and can stop cleanly mid-file.
        If the WAV file doesn't exist, falls back to silence (no crash).
        """
        if not WAITING_BEAT_PATH.exists():
            logger.warning(f"Waiting beat not found at {WAITING_BEAT_PATH}.")
            stop_event.wait()
            return

        try:
            import soundfile as sf
            audio, sample_rate = sf.read(str(WAITING_BEAT_PATH), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)  # to mono

            chunk_size = int(sample_rate * 0.1)  # 100ms chunks
            logger.debug("Playing waiting beat loop...")

            while not stop_event.is_set():
                # Stream the full file in chunks, stopping early if signalled
                pos = 0
                while pos < len(audio) and not stop_event.is_set():
                    chunk = audio[pos : pos + chunk_size]
                    self.robot.media.push_audio_sample(chunk, sample_rate)
                    pos += chunk_size
                    time.sleep(0.1)  # match chunk duration for real-time pacing

        except Exception as e:
            logger.warning(f"Beat loop error: {e}")

    def _thinking_motion(self, stop_event: threading.Event) -> None:
        """
        Slow rhythmic head bob + occasional antenna wiggle.
        Looks like the robot is listening to an imaginary beat.
        """
        phase = 0.0
        while not stop_event.is_set():
            # Gentle side-to-side sway, 0.5 Hz
            yaw = 12 * np.sin(2 * np.pi * 0.5 * phase)
            # Slight nod on the beat
            pitch_z = 5 * np.sin(2 * np.pi * 1.0 * phase)

            try:
                pose = create_head_pose(yaw=yaw, z=pitch_z, degrees=True, mm=True)
                self.robot.set_target(head=pose)
            except Exception:
                pass  # Don't crash the waiting loop on motion errors

            # Antenna wiggle every ~4 seconds
            if int(phase * 10) % 40 == 0:
                try:
                    self.robot.set_target(antennas=[0.4, -0.4])
                    time.sleep(0.15)
                    self.robot.set_target(antennas=[-0.4, 0.4])
                    time.sleep(0.15)
                    self.robot.set_target(antennas=[0.0, 0.0])
                except Exception:
                    pass

            time.sleep(0.05)
            phase += 0.05
