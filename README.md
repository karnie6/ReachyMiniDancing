# Reachy DJ 🤖🎵

Describe any song you want. Reachy generates it with AI, plays it through its speaker, and dances — with moves chosen by an LLM to match the vibe.

## Full flow

```
You (voice or web UI): "heavy funky bass groove"
         │
         ▼
Whisper STT (local, no cloud)
         │
         ▼
Reachy: "On it. Give me a moment to cook."
         │
         ├──► Waiting beat loops from speaker
         │    + slow thinking head bob + antenna wiggle
         │
         ├──► Music generation API call (background thread)
         │    Phase 1: udioapi.pro  (~20-40s)
         │    Phase 2: MusicGen local (~15-60s depending on hardware)
         │
         ▼  (generation done)
LLM picks dance moves for "heavy funky bass groove"
→ ["groovy_sway_and_roll", "chicken_peck", "polyrhythm_combo", "neck_recoil"]
         │
         ▼
Reachy: "Here we go!"
         │
         ├──► MP3 decoded → PCM chunks → pushed to Reachy speaker (USB)
         └──► Dance loop cycles through selected moves
         │
         ▼  (song ends)
Reachy takes a bow 🙇
"What else should I make?"
```

## Setup

### 1. System dependency — ffmpeg (for MP3 decode)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Install Python deps
```bash
pip install -e .
```

### 3. Configure API keys
```bash
cp reachy_dj/.env.example .env
# Edit .env — add your UDIO_API_KEY (and optionally OPENAI_API_KEY)
```

Get a free udioapi.pro key at https://udioapi.pro (free tier: ~10 songs/day).

`OPENAI_API_KEY` is optional — used for LLM dance selection (~$0.001/call).
If not set, falls back to keyword heuristics (works fine).

### 4. Add a waiting beat (optional but recommended)
Drop a short drum loop WAV into `assets/waiting_beat.wav`.
Any 2-4 bar loop works — ~2-4 seconds, 44100 Hz.
Free sources: freesound.org, looperman.com.
Without it, Reachy still moves but plays no audio while generating.

### 5. Run
```bash
# Start Reachy daemon first
python -m reachy_mini.daemon.app.main

# In another terminal
python -c "
import threading
from reachy_mini import ReachyMini
from reachy_dj import ReachyDJ

stop = threading.Event()
with ReachyMini() as mini:
    ReachyDJ().run(mini, stop)
"
```

Web UI available at **http://localhost:7860** once running.

---

## Switching to Phase 2 (fully open source, no API keys)

When you're ready to publish to the HF app store:

1. `pip install -e ".[musicgen]"`
2. In `music_generator.py`, uncomment the `MusicGenGenerator` implementation
3. In `main.py`, change:
   ```python
   generator = UdioApiGenerator()
   # → becomes:
   generator = MusicGenGenerator()
   ```
4. Remove `UDIO_API_KEY` from `.env` — no keys needed
5. Ship it ✅

MusicGen runs locally on your laptop (CPU: ~60-120s, Mac MPS: ~15-30s).
For Reachy Wireless, generation still runs on the laptop — RPi 5 can't run it.

---

## Architecture

```
reachy_dj/
├── main.py             # App entry point, orchestration loop
├── music_generator.py  # Abstraction: UdioApiGenerator (P1) + MusicGenGenerator (P2)
├── dance_selector.py   # LLM picks moves, keyword fallback
├── audio_player.py     # MP3 → PCM → Reachy speaker streaming
├── waiting_behavior.py # "Thinking" motion + beat loop during generation
├── voice_listener.py   # Reachy mic → Whisper STT
└── web_ui.py           # Gradio typed prompt fallback (localhost:7860)

assets/
└── waiting_beat.wav    # Pre-bundled drum loop (add your own)
```
