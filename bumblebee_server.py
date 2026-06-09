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
import time

import sounddevice as sd
import websockets

import openwakeword
from openwakeword.model import Model

import config

WS_HOST = "localhost"
WS_PORT = 8767
CLIENTS = set()


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
    print(f"[WS] Client connected ({len(CLIENTS)})")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"[WS] Client disconnected ({len(CLIENTS)})")


async def broadcast(message: str) -> None:
    if not CLIENTS:
        return
    await asyncio.gather(*[client.send(message) for client in list(CLIENTS)], return_exceptions=True)


def resolve_device(device_arg):
    if device_arg is None:
        return None
    if isinstance(device_arg, str) and not device_arg.lstrip("-").isdigit():
        devices = sd.query_devices()
        matches = [i for i, d in enumerate(devices)
                   if device_arg.lower() in d["name"].lower()
                   and d["max_input_channels"] > 0]
        if not matches:
            raise SystemExit(f"✗ No input device matching '{device_arg}' found.")
        return matches[0]
    return int(device_arg)


def build_model() -> Model:
    onnx_path = config.MODEL_DIR / f"{config.MODEL_NAME}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"✗ Model not found: {onnx_path}\n  Run step4_train.py first.")

    openwakeword.utils.download_models()
    print(f"[MODEL] Loading ONNX model: {onnx_path.name}")
    model = Model(
        wakeword_models=[str(onnx_path)],
        inference_framework="onnx",
        vad_threshold=0.0,
    )
    print("[MODEL] Loaded successfully")
    return model


async def audio_loop(device, threshold, debounce_ms):
    loop = asyncio.get_running_loop()
    model = build_model()
    last_trigger = 0.0
    chunk_size = 1280
    debounce_seconds = debounce_ms / 1000.0

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger
        if status:
            print(f"[MIC] Input warning: {status}")
        if frames != chunk_size:
            return

        chunk = (indata[:, 0] * 32768).astype("int16")
        try:
            pred = model.predict(chunk)
        except Exception as exc:
            print(f"[MODEL] Prediction error: {exc}")
            return

        score = float(list(pred.values())[0])
        now = time.time()
        if score >= threshold and (now - last_trigger) >= debounce_seconds:
            last_trigger = now
            model.reset()
            payload = json.dumps({"event": "wake", "keyword": "bumblebee", "score": score})
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)
            print(f"[DETECT] bumblebee detected (score={score:.3f})")

    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"[MIC] Listening on device {device}: {dev_info['name']}")
    else:
        dev_info = sd.query_devices(kind="input")
        print(f"[MIC] Listening on default input device: {dev_info['name']}")

    print(f"[MIC] Sample rate: {config.SAMPLE_RATE}, frame size: {chunk_size}")
    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=config.SAMPLE_RATE,
        dtype="float32",
        blocksize=chunk_size,
        callback=callback,
    ):
        print("[MIC] Audio stream started")
        await asyncio.Future()  # run forever


async def main(device, threshold, debounce_ms):
    print(f"[WS] Starting server ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await audio_loop(device, threshold, debounce_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bumblebee ONNX wake-word WebSocket server")
    parser.add_argument("--device", default=None,
                        help="Audio input device index or partial name")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold 0–1")
    parser.add_argument("--debounce", type=int, default=800,
                        help="Milliseconds between detected wake events")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        raise SystemExit()

    device_idx = resolve_device(args.device)
    try:
        asyncio.run(main(device_idx, args.threshold, args.debounce))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
