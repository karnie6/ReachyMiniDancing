"""
test_generate.py — Full pipeline test: generate → play → dance
===============================================================
Takes a text prompt, generates a song with MusicGen, plays it,
and dances. No pre-existing audio file needed.

Usage:
  # 1. Make sure the Reachy daemon is running:
  #    python -m reachy_mini.daemon.app.main

  # 2. Activate venv and run:
  #    source .venv/bin/activate
  #    python test_generate.py "upbeat funky groovy"
"""

import sys
import threading
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_generate")

# ── Config ────────────────────────────────────────────────────────────────────

PROMPT     = sys.argv[1] if len(sys.argv) > 1 else "upbeat funky groovy"
NUM_MOVES  = 3    # how many dance moves to cycle through
MAX_SECONDS = 120  # hard cutoff for both audio and dance


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load model + connect to Reachy first ──────────────────────────────────
    from reachy_dj.music_generator import create_generator
    from reachy_dj.dance_selector import pick_dances_for_prompt
    from reachy_mini import ReachyMini
    from reachy_dj.waiting_behavior import WaitingBehavior

    generator = create_generator()

    dance_moves = pick_dances_for_prompt(PROMPT)[:NUM_MOVES]
    logger.info(f"Selected moves: {dance_moves}")

    logger.info("Connecting to Reachy Mini...")
    mini = ReachyMini()
    logger.info("Connected.")

    # ── Generate music while playing loading beat + waiting motion ────────────
    gen_done = threading.Event()
    audio_result = {}

    def generate():
        try:
            logger.info(f"Generating song for prompt: '{PROMPT}'")
            audio_result["data"] = generator.generate(PROMPT)
            logger.info(f"Generation done ({len(audio_result['data']) // 1024} KB)")
        except Exception as e:
            audio_result["error"] = str(e)
        finally:
            gen_done.set()

    gen_thread = threading.Thread(target=generate, daemon=True)
    gen_thread.start()

    WaitingBehavior(mini).start(gen_done)  # blocks until gen_done is set

    if "error" in audio_result:
        raise RuntimeError(f"Generation failed: {audio_result['error']}")
    audio_bytes = audio_result["data"]

    # ── Shared stop event — set by timeout OR when audio finishes ────────────
    stop_event = threading.Event()
    timer = threading.Timer(MAX_SECONDS, lambda: (
        logger.info("120s timeout reached — cutting off."),
        stop_event.set(),
    ))
    timer.daemon = True
    timer.start()

    # ── Launch dance thread ───────────────────────────────────────────────────
    def dance_loop():
        from reachy_mini_dances_library import DanceMove
        dt = 0.05  # 20 Hz
        idx = 0
        while not stop_event.is_set():
            name = dance_moves[idx % len(dance_moves)]
            try:
                move = DanceMove(name)
                t = 0.0
                while t < move.duration and not stop_event.is_set():
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
    logger.info(f"Dance loop started (max {MAX_SECONDS}s).")

    # ── Stream audio (blocks until done or stop_event set) ───────────────────
    from reachy_dj.audio_player import stream_mp3_to_reachy
    logger.info("Streaming audio — robot should now dance and play music...")
    stream_mp3_to_reachy(mini, audio_bytes, stop_event=stop_event)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    timer.cancel()
    stop_event.set()
    logger.info("Audio done — stopping dance.")
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
