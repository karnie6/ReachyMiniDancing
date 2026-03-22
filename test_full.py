"""
test_full.py — End-to-end audio + dance test (no music generation needed)
=========================================================================
Uses a local MP3 file so you can test the full pipeline without an API key.

Usage:
  # 1. Make sure the Reachy daemon is running:
  #    python -m reachy_mini.daemon.app.main

  # 2. Activate venv and run:
  #    source .venv/bin/activate
  #    python test_full.py /path/to/your/song.mp3

  # Optional: pass a genre hint as second arg for dance selection
  #    python test_full.py song.mp3 "upbeat funky groovy"
"""

import sys
import threading
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_full")

# ── Config ────────────────────────────────────────────────────────────────────

MP3_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_song.mp3")
PROMPT   = sys.argv[2] if len(sys.argv) > 2 else "upbeat funky groovy"
NUM_MOVES = 3  # how many dance moves to pick


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not MP3_PATH.exists():
        print(f"ERROR: MP3 file not found: {MP3_PATH}")
        print("Download a song from Suno/Udio, then pass its path as first arg.")
        sys.exit(1)

    logger.info(f"Loading MP3: {MP3_PATH} ({MP3_PATH.stat().st_size // 1024} KB)")
    mp3_bytes = MP3_PATH.read_bytes()

    # ── Dance selection (keyword fallback — no API key needed) ────────────────
    from reachy_dj.dance_selector import pick_dances_for_prompt
    dance_moves = pick_dances_for_prompt(PROMPT)[:NUM_MOVES]
    logger.info(f"Selected {len(dance_moves)} moves for prompt '{PROMPT}': {dance_moves}")

    # ── Connect to Reachy ─────────────────────────────────────────────────────
    from reachy_mini import ReachyMini
    logger.info("Connecting to Reachy Mini...")
    mini = ReachyMini()
    logger.info("Connected.")

    # ── Launch dance thread ───────────────────────────────────────────────────
    dance_stop = threading.Event()

    def dance_loop():
        from reachy_mini_dances_library import DanceMove
        dt = 0.05  # 20 Hz
        idx = 0
        while not dance_stop.is_set():
            name = dance_moves[idx % len(dance_moves)]
            try:
                move = DanceMove(name)
                t = 0.0
                while t < move.duration and not dance_stop.is_set():
                    head_pose, antennas, _ = move.evaluate(t)
                    mini.set_target(head=head_pose, antennas=antennas)
                    time.sleep(dt)
                    t += dt
            except Exception as e:
                logger.warning(f"Move '{name}' error: {e}")
                time.sleep(0.3)
            idx += 1

    dance_thread = threading.Thread(target=dance_loop, daemon=True)
    dance_thread.start()
    logger.info("Dance loop started.")

    # ── Stream audio (blocks until song finishes) ─────────────────────────────
    from reachy_dj.audio_player import stream_mp3_to_reachy
    logger.info("Streaming audio — robot should now dance and play music...")
    stream_mp3_to_reachy(mini, mp3_bytes)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.info("Audio done — stopping dance.")
    dance_stop.set()
    dance_thread.join(timeout=3)

    # Quick bow
    try:
        from reachy_mini.utils import create_head_pose
        mini.goto_target(head=create_head_pose(z=-15, mm=True), duration=0.6)
        time.sleep(0.7)
        mini.goto_target(head=create_head_pose(), duration=0.5)
    except Exception as e:
        logger.warning(f"Bow failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
