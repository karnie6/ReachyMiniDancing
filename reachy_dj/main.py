"""
Reachy DJ - AI Music Generator + Dancer
========================================
Tell Reachy what kind of song you want. It generates it with AI,
plays it through its speaker, and dances — with moves chosen to
match the vibe of your prompt.

Flow:
  1. Voice command (mic) OR typed prompt (web UI)
  2. Reachy says "let me cook" + plays a waiting beat + does idle animation
  3. Music generates in background (udioapi.pro → MP3 bytes)
  4. LLM picks dance moves based on the prompt
  5. Music streams to speaker + dance runs concurrently
  6. Song ends → Reachy takes a bow
"""

import queue
import threading
import logging
import time

from reachy_mini import ReachyMini, ReachyMiniApp

from .music_generator import MusicGeneratorBase, MusicGenGenerator
from .dance_selector import pick_dances_for_prompt
from .audio_player import stream_mp3_to_reachy
from .voice_listener import listen_for_command
from .waiting_behavior import WaitingBehavior
from .web_ui import launch_web_ui

logger = logging.getLogger(__name__)


class ReachyDJ(ReachyMiniApp):
    """
    Reachy DJ — describe a song, Reachy generates it and dances to it.

    Inputs:
      - Voice: say your prompt out loud (Whisper STT)
      - Web UI: type it at http://localhost:7860 (Gradio)

    Environment variables (in .env):
      UDIO_API_KEY=your_key_here
      OPENAI_API_KEY=your_key_here   (used only for dance selection — cheap)
    """

    # Gradio web UI spun up by the Reachy app framework
    custom_app_url: str | None = "http://localhost:7860"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        generator = MusicGenGenerator()
        waiter = WaitingBehavior(reachy_mini)

        # Shared queue for typed prompts from the web UI
        prompt_queue: queue.Queue[str] = queue.Queue()
        web_ui_thread = threading.Thread(
            target=launch_web_ui,
            args=(prompt_queue, stop_event),
            daemon=True,
        )
        web_ui_thread.start()

        reachy_mini.speaker.say("DJ Reachy online. Tell me what to make!")
        logger.info("Reachy DJ started.")

        while not stop_event.is_set():
            # ── 1. Get prompt (voice or web UI) ──────────────────────────────
            prompt = _get_prompt(reachy_mini, stop_event, prompt_queue)
            if prompt is None:
                continue  # timeout or stop signal

            logger.info(f"Prompt received: '{prompt}'")
            reachy_mini.speaker.say("On it. Give me a moment to cook.")

            # ── 2. Kick off music generation + waiting behavior in parallel ──
            mp3_result: dict = {"data": None, "error": None}
            gen_done = threading.Event()

            def generate():
                try:
                    mp3_result["data"] = generator.generate(prompt)
                except Exception as e:
                    mp3_result["error"] = str(e)
                finally:
                    gen_done.set()

            gen_thread = threading.Thread(target=generate, daemon=True)
            gen_thread.start()

            # Robot does its "thinking" routine while music generates
            waiter.start(gen_done)  # loops until gen_done is set

            # ── 3. Handle result ──────────────────────────────────────────────
            if mp3_result["error"] or mp3_result["data"] is None:
                logger.error(f"Generation failed: {mp3_result['error']}")
                reachy_mini.speaker.say("Sorry, I couldn't generate that. Try again?")
                continue

            mp3_bytes = mp3_result["data"]
            reachy_mini.speaker.say("Here we go!")
            time.sleep(0.8)  # brief pause before drop

            # ── 4. LLM picks dances for this prompt ──────────────────────────
            dance_moves = pick_dances_for_prompt(prompt)
            logger.info(f"Selected dances: {dance_moves}")

            # ── 5. Stream audio + dance simultaneously ────────────────────────
            dance_stop = threading.Event()
            dance_thread = threading.Thread(
                target=_dance_loop,
                args=(reachy_mini, dance_moves, dance_stop),
                daemon=True,
            )
            dance_thread.start()

            # stream_mp3_to_reachy blocks until the song finishes
            stream_mp3_to_reachy(reachy_mini, mp3_bytes)

            # ── 6. Song over ──────────────────────────────────────────────────
            dance_stop.set()
            dance_thread.join(timeout=3)
            _take_a_bow(reachy_mini)

            # Loop back — ready for next prompt
            reachy_mini.speaker.say("What else should I make?")

        logger.info("Reachy DJ stopped.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_prompt(
    reachy_mini: ReachyMini,
    stop_event: threading.Event,
    prompt_queue: queue.Queue,
) -> str | None:
    """
    Check the web UI queue first (instant), then listen for voice (up to 10s),
    then check the queue once more in case something arrived while we listened.
    Returns None if nothing received — caller will loop.
    """
    # Check queue immediately — typed prompt may already be waiting
    try:
        return prompt_queue.get_nowait()
    except queue.Empty:
        pass

    logger.debug("Listening for voice prompt...")
    text = listen_for_command(reachy_mini, timeout_seconds=10)
    if stop_event.is_set():
        return None

    if text:
        return text

    # One more queue check — user may have typed while we were listening
    try:
        return prompt_queue.get_nowait()
    except queue.Empty:
        return None


def _dance_loop(reachy_mini: ReachyMini, dance_moves: list[str], stop_event: threading.Event):
    """
    Cycles through the LLM-selected dance moves until stop_event is set.
    Uses DanceMove.evaluate() at 20Hz so stop_event is checked every tick.
    """
    from reachy_mini_dances_library import DanceMove

    dt = 0.05  # 20 Hz
    idx = 0
    while not stop_event.is_set():
        name = dance_moves[idx % len(dance_moves)]
        logger.debug(f"Dancing: {name}")
        try:
            move = DanceMove(name)
            t = 0.0
            while t < move.duration and not stop_event.is_set():
                head_pose, antennas, _ = move.evaluate(t)
                reachy_mini.set_target(head=head_pose, antennas=antennas)
                time.sleep(dt)
                t += dt
        except Exception as e:
            logger.warning(f"Dance move '{name}' failed: {e}")
            time.sleep(0.3)
        idx += 1


def _take_a_bow(reachy_mini: ReachyMini):
    """Quick bow animation to punctuate the end of the song."""
    from reachy_mini.utils import create_head_pose
    reachy_mini.goto_target(head=create_head_pose(z=-15, mm=True), duration=0.6)
    time.sleep(0.7)
    reachy_mini.goto_target(head=create_head_pose(), duration=0.5)
