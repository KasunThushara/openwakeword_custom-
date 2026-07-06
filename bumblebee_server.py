"""
bumblebee_server.py
===================
WebSocket bridge for your custom bumblebee ONNX wake-word model.

Run this on the same machine where your mic and model are available.
Then open bumblebee_face.html in a browser.

Requirements:
    pip install openwakeword onnxruntime sounddevice websockets

Usage:
    python bumblebee_server.py
    python bumblebee_server.py --device 9
    python bumblebee_server.py --threshold 0.55
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import sounddevice as sd
import websockets

import openwakeword
from openwakeword.model import Model

import config
import doa_reader

WS_HOST = "localhost"
WS_PORT = 8767
CLIENTS = set()


def log(msg: str) -> None:
    """Print with timestamp so docker logs are useful."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def build_model() -> Model:
    """Load the custom ONNX model via openWakeWord."""
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    log(f"Looking for model: {onnx_path}")

    if not onnx_path.exists():
        default_path = config.DEFAULT_MODEL_DIR / f"{config.MODEL_NAME}.onnx"
        log(f"Model not found in {config.MODEL_DIR}, checking fallback: {default_path}")
        if default_path.exists():
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(default_path, onnx_path)
            log(f"Copied fallback model to {onnx_path}")
        else:
            log(f"FATAL: Model not found at {onnx_path} or {default_path}")
            log("Run step4_train.py first to generate the model.")
            raise SystemExit(1)

    # Download openWakeWord's internal pre-trained models (VAD + embedding backbone).
    # These are cached so subsequent restarts skip the download.
    log("Checking openWakeWord pre-trained models (downloaded if missing) ...")
    try:
        openwakeword.utils.download_models()
        log("Pre-trained models ready")
    except Exception as exc:
        log(f"WARNING: download_models() failed: {exc}")
        log("The server may still work if models were already cached.")
        log("If it crashes, check network connectivity inside the container.")

    log(f"Loading ONNX model: {onnx_path.name}")
    try:
        model = Model(
            wakeword_models=[str(onnx_path)],
            inference_framework="onnx",
            vad_threshold=0.0,
        )
    except Exception as exc:
        log(f"FATAL: Failed to load model: {exc}")
        traceback.print_exc()
        raise SystemExit(1)

    log("Model loaded successfully")
    return model


async def audio_loop(device, threshold, debounce_ms):
    loop = asyncio.get_running_loop()
    model = build_model()
    last_trigger = 0.0
    chunk_size = 1280
    debounce_seconds = debounce_ms / 1000.0
    report_interval = 1.5  # print debug stats every N seconds

    if device is not None:
        dev_info = sd.query_devices(device)
        num_channels = dev_info["max_input_channels"]
    else:
        dev_info = sd.query_devices(kind="input")
        num_channels = dev_info["max_input_channels"]

    log(f"Microphone: [{device}] {dev_info['name']}")
    log(f"Channels: {num_channels}, default sample rate: {dev_info['default_samplerate']}")
    log(f"Chunk size: {chunk_size} samples ({chunk_size / config.SAMPLE_RATE * 1000:.0f} ms)")
    log(f"Threshold: {threshold}, debounce: {debounce_ms} ms")
    log("Starting audio stream ...")

    doa_dev = None
    doa_task = None

    # Debug accumulators
    dbg_chunks = 0
    dbg_rms_sum = 0.0
    dbg_rms_peak = 0.0
    dbg_score_peak = 0.0
    dbg_last_report = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger, dbg_chunks, dbg_rms_sum, dbg_rms_peak, dbg_score_peak, dbg_last_report
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

        score = float(list(pred.values())[0])

        dbg_chunks += 1
        dbg_rms_sum += rms
        if rms > dbg_rms_peak:
            dbg_rms_peak = rms
        if score > dbg_score_peak:
            dbg_score_peak = score

        now = time.time()
        if now - dbg_last_report >= report_interval:
            avg_rms = dbg_rms_sum / max(dbg_chunks, 1)
            log(f"DEBUG  chunks={dbg_chunks}  avg_RMS={avg_rms:.0f}  peak_RMS={dbg_rms_peak:.0f}  peak_score={dbg_score_peak:.4f}  threshold={threshold}")
            dbg_chunks = 0
            dbg_rms_sum = 0.0
            dbg_rms_peak = 0.0
            dbg_score_peak = 0.0
            dbg_last_report = now

        if score >= threshold and (now - last_trigger) >= debounce_seconds:
            last_trigger = now
            model.reset()
            payload = json.dumps({
                "type": "wake",
                "keyword": "bumblebee",
                "score": round(score, 4),
            })
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)
            log(f"DETECTED  bumblebee  score={score:.4f}")

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
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold 0–1 (default 0.5)")
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
