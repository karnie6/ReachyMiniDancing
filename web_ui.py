"""
Web UI
======
Simple Gradio interface served at http://localhost:7860
Acts as the fallback when voice input isn't used.

The user types a prompt → it goes into a shared queue →
main.py picks it up on the next loop iteration.

To launch alongside the main app:
    from .web_ui import launch_web_ui
    threading.Thread(target=launch_web_ui, args=(prompt_queue,), daemon=True).start()
"""

import queue
import logging
import threading

logger = logging.getLogger(__name__)


def launch_web_ui(prompt_queue: queue.Queue, stop_event: threading.Event) -> None:
    """
    Spins up a Gradio UI. Blocks until stop_event is set.
    Run this in a daemon thread.
    """
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed — web UI unavailable. pip install gradio")
        stop_event.wait()
        return

    def submit_prompt(prompt: str):
        if not prompt.strip():
            return "⚠️ Please enter a prompt."
        prompt_queue.put(prompt.strip())
        return f"✅ Queued: \"{prompt.strip()}\"\nRobot will start generating shortly..."

    with gr.Blocks(title="Reachy DJ") as demo:
        gr.Markdown("# 🤖🎵 Reachy DJ")
        gr.Markdown(
            "Describe the music you want. Reachy will generate it with AI and dance to it.\n\n"
            "**Examples:**\n"
            "- `upbeat funky groove with bass`\n"
            "- `slow jazz with piano and trumpet`\n"
            "- `heavy metal with fast drums`\n"
            "- `lo-fi hip hop chill beats`"
        )

        with gr.Row():
            prompt_box = gr.Textbox(
                placeholder="Describe your song...",
                label="Music Prompt",
                scale=4,
            )
            submit_btn = gr.Button("🎵 Generate + Dance", variant="primary", scale=1)

        status = gr.Textbox(label="Status", interactive=False)

        submit_btn.click(fn=submit_prompt, inputs=prompt_box, outputs=status)
        prompt_box.submit(fn=submit_prompt, inputs=prompt_box, outputs=status)

    # Launch non-blocking so we can watch stop_event
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True,
        quiet=True,
    )
    logger.info("Web UI available at http://localhost:7860")

    # Keep alive until app stops
    stop_event.wait()
    demo.close()
