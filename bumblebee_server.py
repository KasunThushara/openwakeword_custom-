"""
bumblebee_server.py
===================
Two-stage wake-word + command detection server.

  1. WAITING  — listens for "bumblebee" (wake word)
  2. COMMAND  — listens for command phrases for 6 seconds
     (activate party mode, back to normal mode, play spanish music,
      wave your antenna)

All .onnx models in model/ are loaded at startup.
Broadcasts wake/intent/DOA to WebSocket clients on ws://localhost:8767.

Requirements:
    pip install openwakeword onnxruntime sounddevice websockets pyusb

Usage:
    python bumblebee_server.py
    python bumblebee_server.py --device respeaker --threshold 0.75
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading 
import time
import traceback

import numpy as np
import sounddevice as sd
import websockets

import openwakeword
from openwakeword.model import Model

import config

WS_HOST = "localhost"
WS_PORT = 8767
CLIENTS = set()
_current_sound = None

# Map ONNX filename (without .onnx) → intent + optional sound effect
# The wake word "bumblebee" is NOT in this dict — it's the gatekeeper.
COMMANDS = {
    "activate_party_mode": {
        "intent": {"type": "intent", "intent": "lights",
                   "slots": {"command": "activate", "mode": "party"}},
        "sound": None,
    },
    "back_to_normal_mode": {
        "intent": {"type": "intent", "intent": "lights",
                   "slots": {"command": "deactivate", "mode": "party"}},
        "sound": None,
    },
    "play_spanish_music": {
        "intent": {"type": "intent", "intent": "sounds",
                   "slots": {"genre": "spanish"}},
        "sound": "spanish.mp3",
    },
    "wave_your_antenna": {
        "intent": {"type": "intent", "intent": "gesture",
                   "slots": {"command": "shake", "part": "antenna"}},
        "sound": None,
    },
}


def log(msg: str) -> None:
    """Print with timestamp so docker logs are useful."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def play_sound(path: str) -> None:
    """Play a WAV or MP3 file non-blocking via hardware ALSA device."""
    full = str(config.MUSIC_DIR / path)
    if not os.path.exists(full):
        log(f"Sound not found: {full}")
        return
    global _current_sound
    if _current_sound and _current_sound.poll() is None:
        _current_sound.terminate()
        _current_sound.wait()
    try:
        if full.endswith(".mp3"):
            _current_sound = subprocess.Popen(
                ["mpg123", "-q", full],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        else:
            _current_sound = subprocess.Popen(
                ["aplay", "-q", full],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
    except FileNotFoundError:
        log("aplay / mpg123 not found — install alsa-utils mpg123")
    except Exception as exc:
        log(f"Sound playback error: {exc}")

    def _check():
        if _current_sound:
            rc = _current_sound.wait()
            if rc != 0:
                err = _current_sound.stderr.read().decode(errors="replace")
                log(f"Sound via PulseAudio failed (exit {rc}), trying direct hardware ...")
                # Fallback: direct ALSA hardware (works in Docker / root, no PulseAudio needed)
                try:
                    if full.endswith(".mp3"):
                        fallback = subprocess.Popen(
                            ["mpg123", "-q", "-o", "alsa", "-a", "hw:1,0", full],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        )
                    else:
                        fallback = subprocess.Popen(
                            ["aplay", "-D", "hw:1,0", full],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        )
                    fc = fallback.wait()
                    if fc != 0:
                        ferr = fallback.stderr.read().decode(errors="replace")
                        log(f"Sound via hw:2,0 also failed (exit {fc}): {ferr.strip()}")
                    else:
                        log("Sound playback OK (direct hardware)")
                except Exception as exc2:
                    log(f"Sound fallback error: {exc2}")
            else:
                log("Sound playback OK")

    threading.Thread(target=_check, daemon=True).start()


def list_devices() -> None:
    devices = sd.query_devices()
    print("\n  Audio input devices:\n")
    print(f"  {'IDX':>4}  {'NAME':<45}  {'IN CH':>6}")
    print("  " + "─" * 65)
    default_in = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            tag = ""
            if i == default_in:
                tag = " ← default"
            print(f"  {i:>4}  {d['name']:<45}  {d['max_input_channels']:>6}{tag}")
    print()


async def ws_handler(websocket):
    CLIENTS.add(websocket)
    log(f"WebSocket client connected (total: {len(CLIENTS)})")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        log(f"WebSocket client disconnected (total: {len(CLIENTS)})")


async def broadcast(message: str) -> None:
    if not CLIENTS:
        return
    await asyncio.gather(
        *[client.send(message) for client in list(CLIENTS)],
        return_exceptions=True,
    )


def resolve_device(device_arg):
    """Resolve user-supplied --device arg (index or name substring) to a PortAudio index."""
    if device_arg is None:
        log("No --device specified, using system default input device")
        return None

    if isinstance(device_arg, str) and not device_arg.lstrip("-").isdigit():
        devices = sd.query_devices()
        matches = [
            i for i, d in enumerate(devices)
            if device_arg.lower() in d["name"].lower()
            and d["max_input_channels"] > 0
        ]
        if not matches:
            log(f"ERROR: No input device matching '{device_arg}'")
            log("Available input devices:")
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    log(f"  [{i}] {d['name']}")
            raise SystemExit(1)
        idx = matches[0]
        log(f"Resolved '{device_arg}' → device index {idx} ({devices[idx]['name']})")
        return idx

    return int(device_arg)


def load_all_models():
    """Load bumblebee (wake word) + all command models into one OWW Model."""
    model_dir = config.MODEL_DIR
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"FATAL: No .onnx models found in {model_dir}")

    openwakeword.utils.download_models()

    log(f"Loading {len(onnx_files)} model(s) from {model_dir}:")
    paths = []
    for p in onnx_files:
        log(f"  {p.name}")
        paths.append(str(p))

    model = Model(
        wakeword_models=paths,
        inference_framework="onnx",
        vad_threshold=0.0,
    )
    log("All models loaded successfully")
    return model


async def audio_loop(device, threshold, debounce_ms):
    loop = asyncio.get_running_loop()
    model = load_all_models()
    last_trigger = 0.0
    chunk_size = 1280
    debounce_seconds = debounce_ms / 1000.0
    report_interval = 1.5

    # State machine
    state = "WAITING"
    cmd_start = 0.0
    cmd_high = ("", 0.0)  # (stem, score)

    if device is not None:
        dev_info = sd.query_devices(device)
        num_channels = dev_info["max_input_channels"]
    else:
        dev_info = sd.query_devices(kind="input")
        num_channels = dev_info["max_input_channels"]

    log(f"Microphone: [{device}] {dev_info['name']}")
    log(f"Channels: {num_channels}")
    log(f"Threshold: {threshold}, debounce: {debounce_ms} ms")
    log("Starting audio stream ...")

    doa_dev = None
    doa_task = None

    dbg_chunks = 0
    dbg_rms_sum = 0.0
    dbg_rms_peak = 0.0
    dbg_score_peak = 0.0
    dbg_last_report = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger, state, cmd_start, cmd_high
        nonlocal dbg_chunks, dbg_rms_sum, dbg_rms_peak, dbg_score_peak, dbg_last_report
        if status:
            log(f"Audio input warning: {status}")
        if frames != chunk_size:
            return

        chunk = (indata[:, 0] * 32768).astype("int16")
        rms = float(np.sqrt(np.mean(chunk.astype("float64") ** 2)))

        try:
            pred = model.predict(chunk)
        except Exception as exc:
            log(f"Prediction error: {exc}")
            return

        now = time.time()

        # Extract per-model scores
        bumblebee_score = float(pred.get("bumblebee", 0.0))
        peak_score = bumblebee_score

        # Debug
        dbg_chunks += 1
        dbg_rms_sum += rms
        if rms > dbg_rms_peak:
            dbg_rms_peak = rms

        # ── State: WAITING ───────────────────────────────────────────
        if state == "WAITING":
            if bumblebee_score >= threshold and (now - last_trigger) >= debounce_seconds:
                last_trigger = now
                model.reset()
                play_sound("robot.wav")
                asyncio.run_coroutine_threadsafe(
                    broadcast(json.dumps({"type": "wake"})), loop)
                log(f"WAKE  bumblebee  score={bumblebee_score:.4f}")
                state = "COMMAND_LISTENING"
                cmd_start = now
                cmd_high = ("", 0.0)
            if bumblebee_score > dbg_score_peak:
                dbg_score_peak = bumblebee_score

        # ── State: COMMAND_LISTENING ─────────────────────────────────
        elif state == "COMMAND_LISTENING":
            for stem, info in COMMANDS.items():
                s = float(pred.get(stem, 0.0))
                if s > cmd_high[1]:
                    cmd_high = (stem, s)
                if s > peak_score:
                    peak_score = s

            if cmd_high[1] >= threshold and (now - cmd_start) > 0.5:
                cmd = COMMANDS.get(cmd_high[0])
                if cmd:
                    asyncio.run_coroutine_threadsafe(
                        broadcast(json.dumps(cmd["intent"])), loop)
                    log(f"COMMAND  {cmd_high[0]}  score={cmd_high[1]:.4f}")
                    if cmd["sound"]:
                        play_sound(cmd["sound"])
                state = "WAITING"
                cmd_high = ("", 0.0)
                return

            if now - cmd_start >= config.COMMAND_TIMEOUT:
                asyncio.run_coroutine_threadsafe(
                    broadcast(json.dumps({"type": "timeout"})), loop)
                log("COMMAND timeout — back to standby")
                state = "WAITING"
                cmd_high = ("", 0.0)

        if peak_score > dbg_score_peak:
            dbg_score_peak = peak_score

        if now - dbg_last_report >= report_interval:
            avg_rms = dbg_rms_sum / max(dbg_chunks, 1)
            st = f"state={state}"
            if state == "COMMAND_LISTENING":
                st += f"  top_cmd={cmd_high[0]}:{cmd_high[1]:.3f}  timeout={max(0, config.COMMAND_TIMEOUT - (now - cmd_start)):.0f}s"
            log(f"DEBUG  {st}  avg_RMS={avg_rms:.0f}  peak_RMS={dbg_rms_peak:.0f}  peak_score={dbg_score_peak:.4f}")
            dbg_chunks = 0
            dbg_rms_sum = 0.0
            dbg_rms_peak = 0.0
            dbg_score_peak = 0.0
            dbg_last_report = now

    try:
        with sd.InputStream(
            device=device,
            channels=num_channels,
            samplerate=config.SAMPLE_RATE,
            dtype="float32",
            blocksize=chunk_size,
            callback=callback,
        ):
            log("Audio stream started — listening for wake word")

            # ── DOA reader (ReSpeaker USB control) — start AFTER audio stream is open ──
            try:
                import doa_reader
                doa_dev = doa_reader.DOAReader()
                ver = doa_dev.version()
                log(f"ReSpeaker DOA reader ready (firmware {ver})")
            except Exception as exc:
                log(f"DOA reader not available: {exc}")
                log("Wake-word detection will work; DOA display will use demo mode.")

            async def poll_doa():
                """Broadcast DOA data to WebSocket clients at ~20 Hz."""
                if doa_dev is None:
                    return
                fail_count = 0
                while True:
                    try:
                        data = doa_dev.read()
                        if data is not None:
                            payload = json.dumps({"type": "doa", **data})
                            await broadcast(payload)
                            fail_count = 0
                        else:
                            fail_count += 1
                            if fail_count == 1 or fail_count % 50 == 0:
                                log(f"DOA read returned None (×{fail_count})")
                    except Exception as exc:
                        fail_count += 1
                        if fail_count == 1 or fail_count % 50 == 0:
                            log(f"DOA read error (×{fail_count}): {exc}")
                    await asyncio.sleep(0.05)

            if doa_dev is not None:
                doa_task = asyncio.create_task(poll_doa())

            await asyncio.Future()  # run forever
    except sd.PortAudioError as exc:
        log(f"FATAL: PortAudio error — cannot open audio device: {exc}")
        log("Check that the device is connected and not in use by another process.")
        log(f"Try running: python3 bumblebee_server.py --list-devices")
        raise
    except OSError as exc:
        log(f"FATAL: OS error accessing audio device: {exc}")
        log("Container may lack access to /dev/snd. Check docker-compose devices: section.")
        raise
    finally:
        if doa_task is not None:
            doa_task.cancel()
        if doa_dev is not None:
            doa_dev.close()


async def main(device, threshold, debounce_ms):
    log(f"Starting WebSocket server at ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await audio_loop(device, threshold, debounce_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bumblebee ONNX wake-word WebSocket server",
        epilog="Run --list-devices first to find the correct microphone index.",
    )
    parser.add_argument("--device", default=None,
                        help="Audio input device index or partial name")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Detection threshold 0–1 (default 0.75)")
    parser.add_argument("--debounce", type=int, default=800,
                        help="Milliseconds between detected wake events (default 800)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        raise SystemExit(0)

    log(f"Bumblebee Wake Word Server starting")
    log(f"Python: {sys.version}")
    log(f"openWakeWord version: {getattr(openwakeword, '__version__', 'unknown')}")

    device_idx = resolve_device(args.device)
    try:
        asyncio.run(main(device_idx, args.threshold, args.debounce))
    except KeyboardInterrupt:
        log("Stopped by user (SIGINT)")
    except SystemExit:
        pass
    except Exception:
        log("FATAL: Unhandled exception:")
        traceback.print_exc()
        raise SystemExit(1)
