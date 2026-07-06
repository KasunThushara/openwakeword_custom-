#!/usr/bin/env python3
"""
app.py  —  Wake Word Trainer Web UI
=====================================
pip install flask
python app.py
Open  http://localhost:5000
"""

import collections
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from flask import Flask, Response, jsonify, request, stream_with_context

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH  = PROJECT_ROOT / "config.py"
FACE_PATH    = PROJECT_ROOT / "bumblebee_face.html"

app = Flask(__name__)

# ── State ────────────────────────────────────────────────────────
_steps       = {}   # {int: {process, log_queue, status, started_at}}
_server_proc = None
_server_lock = threading.Lock()
_server_log  = collections.deque(maxlen=200)  # last 200 lines of server output

STEP_SCRIPTS = {
    1: "step1_generate_clips.py",
    2: "step2_get_negatives.py",
    3: "step3_extract_features.py",
    4: "step4_train.py",
}

STEP_META = {
    1: {"title": "GENERATE POSITIVE CLIPS",  "desc": "Piper TTS · wake-word samples",        "eta": "~5–15 min"},
    2: {"title": "GENERATE NEGATIVE CLIPS",   "desc": "Piper TTS · non-wake-word samples",    "eta": "~3–10 min"},
    3: {"title": "EXTRACT FEATURES",          "desc": "openWakeWord backbone · all clips",    "eta": "~45 min ☕"},
    4: {"title": "TRAIN & EXPORT MODEL",      "desc": "Classifier training · ONNX export",   "eta": "~5–15 min"},
}


# ══════════════════════════════════════════════════════════════════
#  Config helpers
# ══════════════════════════════════════════════════════════════════

def read_config() -> dict:
    ns = {}
    try:
        exec(compile(CONFIG_PATH.read_text(), str(CONFIG_PATH), "exec"), ns)
    except Exception as exc:
        print(f"[config] read error: {exc}")
    return ns


def patch_config(updates: dict) -> None:
    text = CONFIG_PATH.read_text()

    # Integer fields (may have _ separators like 5_000)
    for k in ["N_POSITIVE_CLIPS", "N_NEGATIVE_CLIPS", "HIDDEN_SIZE", "BATCH_SIZE", "EPOCHS"]:
        if k in updates:
            text = re.sub(
                rf"^({re.escape(k)}\s*=\s*)[\d_]+",
                rf"\g<1>{int(updates[k])}",
                text, flags=re.MULTILINE,
            )

    # Float fields (may be 1e-3 style)
    for k in ["LEARNING_RATE", "VAL_SPLIT"]:
        if k in updates:
            text = re.sub(
                rf"^({re.escape(k)}\s*=\s*)[\d.e+\-]+",
                rf"\g<1>{float(updates[k])}",
                text, flags=re.MULTILINE,
            )

    # String fields
    for k in ["WAKE_WORD", "MODEL_NAME"]:
        if k in updates:
            text = re.sub(
                rf'^({re.escape(k)}\s*=\s*)["\'].*?["\']',
                rf'\g<1>"{updates[k]}"',
                text, flags=re.MULTILINE,
            )

    # PHRASES list (multi-line block)
    if "PHRASES" in updates:
        lines = "".join(f'    "{p.strip()}",\n' for p in updates["PHRASES"])
        new_block = f"PHRASES = [\n{lines}]"
        text = re.sub(r"PHRASES\s*=\s*\[.*?\]", new_block, text, flags=re.DOTALL)

    CONFIG_PATH.write_text(text)


# ══════════════════════════════════════════════════════════════════
#  Process helpers
# ══════════════════════════════════════════════════════════════════

def _run_thread(step_num: int, script: str) -> None:
    entry = _steps[step_num]
    entry["status"] = "running"
    entry["started_at"] = time.time()
    q: Queue = entry["log_queue"]
    try:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        entry["process"] = proc
        for line in proc.stdout:
            q.put(line.rstrip())
        proc.wait()
        rc = proc.returncode
        entry["status"] = "done" if rc == 0 else "error"
        entry["process"] = None
        entry["elapsed"] = round(time.time() - entry["started_at"])
        q.put(f"__EXIT__{rc}__")
    except Exception as exc:
        entry["status"] = "error"
        entry["process"] = None
        q.put(f"Launch error: {exc}")
        q.put("__EXIT__1__")


def start_step(n: int, script: str) -> tuple[bool, str]:
    if _steps.get(n, {}).get("status") == "running":
        return False, "already running"
    _steps[n] = {"process": None, "log_queue": Queue(), "status": "pending",
                 "started_at": None, "elapsed": None}
    threading.Thread(target=_run_thread, args=(n, script), daemon=True).start()
    return True, "started"


# ══════════════════════════════════════════════════════════════════
#  Audio devices
# ══════════════════════════════════════════════════════════════════

def list_devices() -> list:
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        dflt = sd.default.device[0]
        return [
            {"index": i, "name": d["name"],
             "channels": d["max_input_channels"], "default": i == dflt}
            for i, d in enumerate(devs)
            if d["max_input_channels"] > 0
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


# ══════════════════════════════════════════════════════════════════
#  Routes — Config
# ══════════════════════════════════════════════════════════════════

@app.route("/api/config")
def api_config_get():
    try:
        c = read_config()
        return jsonify({
            "wake_word":     c.get("WAKE_WORD", "bumblebee"),
            "phrases":       c.get("PHRASES", ["bumblebee"]),
            "n_positive":    c.get("N_POSITIVE_CLIPS", 5000),
            "n_negative":    c.get("N_NEGATIVE_CLIPS", 3000),
            "hidden_size":   c.get("HIDDEN_SIZE", 128),
            "epochs":        c.get("EPOCHS", 60),
            "learning_rate": c.get("LEARNING_RATE", 0.001),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/config", methods=["POST"])
def api_config_set():
    d = request.get_json() or {}
    try:
        ww = (d.get("wake_word") or "bumblebee").strip().lower()
        phrases = [p.strip() for p in (d.get("phrases") or [ww]) if str(p).strip()]
        if not phrases:
            phrases = [ww]
        patch_config({
            "WAKE_WORD":        ww,
            "MODEL_NAME":       ww,
            "PHRASES":          phrases,
            "N_POSITIVE_CLIPS": int(d.get("n_positive", 5000)),
            "N_NEGATIVE_CLIPS": int(d.get("n_negative", 3000)),
            "HIDDEN_SIZE":      int(d.get("hidden_size", 128)),
            "EPOCHS":           int(d.get("epochs", 60)),
            "LEARNING_RATE":    float(d.get("learning_rate", 0.001)),
        })
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════
#  Routes — Pipeline
# ══════════════════════════════════════════════════════════════════

@app.route("/api/run/<int:n>", methods=["POST"])
def api_run(n):
    if n not in STEP_SCRIPTS:
        return jsonify({"ok": False}), 404
    ok, msg = start_step(n, STEP_SCRIPTS[n])
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/run/<int:n>/stream")
def api_stream(n):
    if n not in _steps:
        def _empty():
            yield "data: [DONE:0]\n\n"
        return Response(_empty(), mimetype="text/event-stream")

    def generate():
        q: Queue = _steps[n]["log_queue"]
        while True:
            try:
                line = q.get(timeout=20)
                if line.startswith("__EXIT__"):
                    rc = int(line.split("__")[2])
                    yield f"data: [DONE:{rc}]\n\n"
                    return
                safe = line.replace("\r", "").replace("\n", " ")
                yield f"data: {safe}\n\n"
            except Empty:
                yield "data: [KA]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/run/<int:n>/stop", methods=["POST"])
def api_stop(n):
    e = _steps.get(n, {})
    proc = e.get("process")
    if not proc:
        return jsonify({"ok": False, "message": "not running"})
    try:
        proc.terminate()
        e["status"] = "stopped"
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


@app.route("/api/status")
def api_status():
    c    = read_config()
    mn   = c.get("MODEL_NAME", c.get("WAKE_WORD", "bumblebee"))
    pos  = PROJECT_ROOT / "data" / "positive"
    neg  = PROJECT_ROOT / "data" / "negative"
    feat = PROJECT_ROOT / "features"
    ff   = ["positive_train.npy", "positive_val.npy",
            "negative_train.npy", "negative_val.npy"]

    step_info = {}
    for s in STEP_SCRIPTS:
        e = _steps.get(s, {})
        step_info[str(s)] = {
            "status":  e.get("status", "idle"),
            "elapsed": e.get("elapsed"),
        }

    return jsonify({
        "steps":          step_info,
        "model_exists":   (PROJECT_ROOT / "model" / f"{mn}.onnx").exists(),
        "positive_count": len(list(pos.glob("*.wav"))) if pos.exists() else 0,
        "negative_count": len(list(neg.glob("*.wav"))) if neg.exists() else 0,
        "features_exist": all((feat / f).exists() for f in ff),
    })


# ══════════════════════════════════════════════════════════════════
#  Routes — Cleanup
# ══════════════════════════════════════════════════════════════════

@app.route("/api/clean/<target>", methods=["DELETE"])
def api_clean(target):
    targets = {
        "data":     [PROJECT_ROOT / "data" / "positive",
                     PROJECT_ROOT / "data" / "negative"],
        "features": [PROJECT_ROOT / "features"],
        "model":    [PROJECT_ROOT / "model"],
    }
    if target not in targets:
        return jsonify({"ok": False}), 404
    try:
        deleted = 0
        for p in targets[target]:
            if p.exists():
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                deleted += 1
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════
#  Routes — Wake-word server
# ══════════════════════════════════════════════════════════════════

@app.route("/api/devices")
def api_devices():
    return jsonify(list_devices())


@app.route("/api/server/start", methods=["POST"])
def api_srv_start():
    global _server_proc, _server_log
    with _server_lock:
        if _server_proc and _server_proc.poll() is None:
            return jsonify({"ok": False, "message": "already running"})
        d   = request.get_json() or {}
        dev = d.get("device")
        thr = float(d.get("threshold", 0.5))
        cmd = [sys.executable, str(PROJECT_ROOT / "bumblebee_server.py"),
               "--threshold", str(thr)]
        if dev is not None:
            cmd += ["--device", str(dev)]

        _server_log.clear()
        _server_log.append(f"[CMD] {' '.join(cmd)}")

        try:
            _server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
        except Exception as exc:
            _server_log.append(f"[ERROR] Launch failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)})

        def _drain():
            try:
                for line in _server_proc.stdout:
                    _server_log.append(line.rstrip())
                _server_proc.wait()
                rc = _server_proc.returncode
                _server_log.append(f"[EXIT] code={rc}")
            except Exception:
                pass

        threading.Thread(target=_drain, daemon=True).start()

        time.sleep(1.5)
        if _server_proc.poll() is not None:
            # Server exited during startup — capture output for diagnosis
            tail = list(_server_log)[-15:]
            detail = "\n".join(tail)
            _server_log.append("[DIED] Server process exited immediately after launch")
            return jsonify({
                "ok": False,
                "message": "Server crashed at startup. See server-log for details.",
                "log": tail,
            })
        return jsonify({"ok": True, "pid": _server_proc.pid, "log": list(_server_log)[:3]})


@app.route("/api/server/stop", methods=["POST"])
def api_srv_stop():
    global _server_proc, _server_log
    with _server_lock:
        if _server_proc and _server_proc.poll() is None:
            _server_proc.terminate()
            try:
                _server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _server_proc.kill()
            _server_log.append("[STOPPED] Server terminated by user")
        _server_proc = None
        return jsonify({"ok": True})


@app.route("/api/server/status")
def api_srv_status():
    return jsonify({
        "running": bool(_server_proc and _server_proc.poll() is None),
    })


@app.route("/api/server/log")
def api_srv_log():
    return jsonify({"lines": list(_server_log)})


# ══════════════════════════════════════════════════════════════════
#  Route — Face (with keyword substitution)
# ══════════════════════════════════════════════════════════════════

@app.route("/face")
def face():
    if not FACE_PATH.exists():
        return "bumblebee_face.html not found in project root", 404
    c  = read_config()
    kl = c.get("WAKE_WORD", "bumblebee").lower()
    ku = kl.upper()
    kt = kl.capitalize()
    html = FACE_PATH.read_text()
    for old, new in [
        ("Bumblebee · Wake Word",                   f"{kt} · Wake Word"),
        ("Bumblebee · DOA Monitor",                 f"{kt} · DOA Monitor"),
        ("BUMBLEBEE <em>· WAKE WORD MONITOR</em>",  f"{ku} <em>· WAKE WORD MONITOR</em>"),
        ("BUMBLEBEE <em>· DOA MONITOR</em>",        f"{ku} <em>· DOA MONITOR</em>"),
        (">BUMBLEBEE<",                              f">{ku}<"),
        ("'BUMBLEBEE!'",                             f"'{ku}!'"),
        ('"BUMBLEBEE!"',                             f'"{ku}!"'),
        ("say \"bumblebee\" to activate",            f'say "{kl}" to activate'),
        ("'bumblebee'",                              f"'{kl}'"),
        ('"bumblebee"',                              f'"{kl}"'),
        ("· bumblebee ·",                            f"· {kl} ·"),
        ("OPENWAKEWORD · bumblebee",                 f"OPENWAKEWORD · {kl}"),
    ]:
        html = html.replace(old, new)
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ══════════════════════════════════════════════════════════════════
#  Route — Main UI
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ══════════════════════════════════════════════════════════════════
#  UI HTML
# ══════════════════════════════════════════════════════════════════

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wake Word Trainer</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Black+Ops+One&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root {
  --y:    #FFD000;
  --y2:   #FFA500;
  --y3:   #FFE566;
  --bg:   #080600;
  --card: rgba(255,208,0,0.028);
  --bdr:  rgba(255,208,0,0.14);
  --bdr2: rgba(255,208,0,0.28);
  --glow: rgba(255,208,0,0.55);
  --gsft: rgba(255,208,0,0.10);
  --txt:  rgba(255,208,0,0.88);
  --dim:  rgba(255,208,0,0.32);
  --ok:   #44FF88;
  --err:  #FF4444;
  --run:  #33AAFF;
}
*{margin:0;padding:0;box-sizing:border-box}

body {
  background:var(--bg);
  color:var(--txt);
  font-family:'Share Tech Mono',monospace;
  min-height:100vh;
  overflow-x:hidden;
}

/* ── Hex grid background ── */
body::before {
  content:'';
  position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 0 L56 16 L56 50 L28 66 L0 50 L0 16Z' fill='none' stroke='rgba(255,208,0,0.032)' stroke-width='1'/%3E%3Cpath d='M28 66 L56 82 L56 116 L28 132 L0 116 L0 82Z' fill='none' stroke='rgba(255,208,0,0.032)' stroke-width='1'/%3E%3C/svg%3E");
  background-size:56px 100px;
  pointer-events:none;z-index:0;
}

/* ── Header ── */
.hdr {
  position:sticky;top:0;z-index:200;
  background:rgba(8,6,0,0.96);
  border-bottom:1px solid var(--bdr);
  backdrop-filter:blur(10px);
  height:58px;
  display:flex;align-items:center;
  padding:0 32px;gap:18px;
}
.hdr-logo {
  font-family:'Black Ops One',sans-serif;
  font-size:17px;letter-spacing:4px;
  color:var(--y);text-shadow:0 0 18px var(--glow);
  white-space:nowrap;
}
.hdr-logo span {
  font-family:'Share Tech Mono',monospace;
  font-size:10px;letter-spacing:3px;
  color:var(--dim);margin-left:10px;
}
.hdr-kw {
  margin-left:auto;
  font-size:9px;letter-spacing:3px;color:var(--dim);
  display:flex;align-items:center;gap:8px;
}
.hdr-kw strong {
  font-family:'Black Ops One',sans-serif;
  font-size:14px;letter-spacing:4px;
  color:var(--y);text-shadow:0 0 10px var(--glow);
}
.hdr-model-dot {
  width:7px;height:7px;border-radius:50%;
  background:rgba(255,208,0,0.15);
  border:1px solid rgba(255,208,0,0.25);
  transition:all 0.4s;
}
.hdr-model-dot.ready {
  background:var(--ok);
  border-color:var(--ok);
  box-shadow:0 0 6px var(--ok);
}

/* ── Tab bar ── */
.tabs {
  position:sticky;top:58px;z-index:199;
  background:rgba(8,6,0,0.93);
  border-bottom:1px solid var(--bdr);
  backdrop-filter:blur(8px);
  display:flex;
}
.tab {
  padding:0 28px;height:44px;
  font-size:10px;letter-spacing:4px;
  color:var(--dim);
  background:none;border:none;
  border-bottom:2px solid transparent;
  cursor:pointer;
  transition:all 0.18s;
  display:flex;align-items:center;gap:8px;
}
.tab:hover{color:var(--txt);}
.tab.on {
  color:var(--y);
  border-bottom-color:var(--y);
  text-shadow:0 0 12px var(--glow);
}
.tab .t-icon{font-size:12px;}

/* ── Content ── */
.pane{display:none;padding:32px;max-width:940px;margin:0 auto;position:relative;z-index:1;}
.pane.on{display:block;}

/* ── Section card ── */
.sec {
  background:var(--card);
  border:1px solid var(--bdr);
  padding:24px;margin-bottom:18px;
}
.sec-title {
  font-size:9px;letter-spacing:5px;color:var(--dim);
  margin-bottom:22px;
  display:flex;align-items:center;gap:12px;
}
.sec-title::after{content:'';flex:1;height:1px;background:var(--bdr);}

/* ── Field ── */
.field{margin-bottom:16px;}
.field label {
  display:block;font-size:9px;letter-spacing:3px;
  color:var(--dim);margin-bottom:7px;
}
.field input,.field textarea,.field select {
  width:100%;
  background:rgba(255,208,0,0.04);
  border:1px solid var(--bdr);
  color:var(--txt);
  font-family:'Share Tech Mono',monospace;
  font-size:13px;
  padding:9px 13px;
  outline:none;
  transition:border-color 0.2s,box-shadow 0.2s;
}
.field input:focus,.field textarea:focus,.field select:focus {
  border-color:rgba(255,208,0,0.55);
  box-shadow:0 0 0 1px rgba(255,208,0,0.12);
}
.field textarea{resize:vertical;min-height:110px;line-height:1.7;}
.field select option{background:#111008;color:var(--txt);}
.field .hint{font-size:9px;color:var(--dim);margin-top:5px;line-height:1.6;}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}

/* Big keyword input */
.kw-input {
  font-family:'Black Ops One',sans-serif !important;
  font-size:30px !important;
  letter-spacing:8px !important;
  text-align:center;
  padding:20px !important;
  color:var(--y) !important;
}
.kw-input:focus{box-shadow:0 0 0 1px rgba(255,208,0,0.3),0 0 24px rgba(255,208,0,0.08) !important;}

/* ── Buttons ── */
.btn {
  font-family:'Share Tech Mono',monospace;
  font-size:10px;letter-spacing:3px;
  padding:9px 18px;
  border:1px solid;
  cursor:pointer;
  transition:all 0.18s;
  background:none;
  white-space:nowrap;
}
.btn-y   {color:var(--y);   border-color:rgba(255,208,0,0.35); background:rgba(255,208,0,0.07);}
.btn-y:hover{background:rgba(255,208,0,0.16);border-color:rgba(255,208,0,0.6);box-shadow:0 0 14px var(--gsft);}
.btn-ok  {color:var(--ok);  border-color:rgba(68,255,136,0.3); background:rgba(68,255,136,0.06);}
.btn-ok:hover{background:rgba(68,255,136,0.14);}
.btn-err {color:var(--err); border-color:rgba(255,68,68,0.3);  background:rgba(255,68,68,0.06);}
.btn-err:hover{background:rgba(255,68,68,0.14);}
.btn-run {color:var(--run); border-color:rgba(51,170,255,0.3); background:rgba(51,170,255,0.06);}
.btn-run:hover{background:rgba(51,170,255,0.14);}
.btn-lg{padding:13px 30px;font-size:11px;letter-spacing:4px;}
.btn-sm{padding:5px 11px;font-size:9px;letter-spacing:2px;}
.btn:disabled{opacity:0.28;cursor:not-allowed;}

/* ── Stats row ── */
.stats {display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px;}
.stat {
  border:1px solid var(--bdr);
  padding:14px 10px;text-align:center;
  background:var(--card);
}
.stat-n {
  font-family:'Black Ops One',sans-serif;
  font-size:24px;color:var(--y);
  text-shadow:0 0 10px rgba(255,208,0,0.35);
}
.stat-l{font-size:8px;letter-spacing:3px;color:var(--dim);margin-top:4px;}

/* ── Step cards ── */
.step {
  border:1px solid var(--bdr);
  margin-bottom:10px;
  overflow:hidden;
  transition:border-color 0.25s;
}
.step.running{border-color:rgba(51,170,255,0.4);}
.step.done   {border-color:rgba(68,255,136,0.3);}
.step.error  {border-color:rgba(255,68,68,0.35);}

.step-hdr {
  display:flex;align-items:center;gap:14px;
  padding:14px 18px;
  cursor:pointer;
  background:var(--card);
  user-select:none;
}
.step-hdr:hover{background:rgba(255,208,0,0.04);}

.step-n {
  width:30px;height:30px;flex-shrink:0;
  border:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:center;
  font-family:'Black Ops One',sans-serif;
  font-size:13px;color:var(--dim);
  transition:all 0.25s;
}
.step.running .step-n{border-color:var(--run);color:var(--run);}
.step.done    .step-n{border-color:var(--ok); color:var(--ok);}
.step.error   .step-n{border-color:var(--err);color:var(--err);}

.step-info{flex:1;min-width:0;}
.step-title{font-size:12px;letter-spacing:2px;}
.step-desc {font-size:9px;color:var(--dim);margin-top:3px;}

.step-tag {
  font-size:8px;letter-spacing:3px;
  padding:3px 8px;border:1px solid;
  flex-shrink:0;
}
.tag-idle   {color:var(--dim);border-color:var(--bdr);}
.tag-running{color:var(--run);border-color:rgba(51,170,255,0.4);animation:blink 1.4s ease-in-out infinite;}
.tag-done   {color:var(--ok); border-color:rgba(68,255,136,0.35);}
.tag-error  {color:var(--err);border-color:rgba(255,68,68,0.35);}
.tag-stopped{color:var(--dim);border-color:var(--bdr);}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.45}}

.step-btns{display:flex;gap:6px;flex-shrink:0;}

/* Progress bar */
.step-prog{height:2px;background:rgba(255,208,0,0.08);}
.step-prog-fill{
  height:100%;width:0%;
  background:var(--y);
  transition:width 0.3s,background 0.3s;
}
.step-prog-fill.ind{
  animation:ind 1.6s ease-in-out infinite;
  width:25%;background:var(--run);
}
.step-prog-fill.full{width:100%;}
@keyframes ind{
  0%{transform:translateX(-100%)}
  100%{transform:translateX(500%)}
}

/* Log area */
.step-log {
  display:none;
  max-height:240px;overflow-y:auto;
  background:#040302;
  border-top:1px solid var(--bdr);
  padding:10px 14px;
  font-size:11px;line-height:1.65;
}
.step-log.open,.step.running .step-log,.step.done .step-log,.step.error .step-log{display:block;}
.log-ln{font-family:'Share Tech Mono',monospace;white-space:pre-wrap;word-break:break-all;color:rgba(255,208,0,0.5);}
.log-ln.ok {color:rgba(68,255,136,0.85);}
.log-ln.err{color:rgba(255,100,100,0.85);}
.log-ln.hi {color:rgba(255,208,0,0.85);}

/* Elapsed badge */
.elapsed{font-size:9px;color:var(--dim);margin-left:6px;}

/* ── Test pane ── */
.test-ctl {
  display:flex;gap:12px;align-items:flex-end;
  flex-wrap:wrap;margin-bottom:18px;
}
.test-ctl .field{margin:0;flex:1;min-width:180px;}
.test-ctl .btn-row{display:flex;gap:8px;padding-bottom:1px;}

.srv-status {
  display:flex;align-items:center;gap:10px;
  font-size:9px;letter-spacing:3px;color:var(--dim);
  margin-bottom:16px;
}
.srv-dot {
  width:8px;height:8px;border-radius:50%;
  background:rgba(255,208,0,0.12);border:1px solid rgba(255,208,0,0.2);
  transition:all 0.3s;flex-shrink:0;
}
.srv-dot.on{background:var(--ok);border-color:var(--ok);box-shadow:0 0 7px var(--ok);animation:blink 1.2s steps(1) infinite;}

.face-frame {
  width:100%;height:640px;
  border:1px solid var(--bdr);
  background:#050400;
  display:block;
}

/* ── Warn banner ── */
.warn {
  border:1px solid rgba(255,208,0,0.25);
  background:rgba(255,208,0,0.05);
  padding:11px 16px;font-size:11px;
  display:flex;align-items:center;gap:10px;
  margin-bottom:16px;
  color:rgba(255,220,50,0.85);
}
.warn.hidden{display:none;}

/* ── Danger row ── */
.danger-row {
  display:flex;gap:10px;flex-wrap:wrap;
  padding-top:4px;
}

/* ── Toast ── */
#toast {
  position:fixed;bottom:22px;right:22px;
  background:rgba(8,6,0,0.97);
  border:1px solid var(--bdr);
  padding:11px 18px;
  font-size:10px;letter-spacing:1px;
  z-index:999;max-width:320px;
  opacity:0;pointer-events:none;
  transition:opacity 0.25s;
  border-left-width:3px;
}
#toast.show{opacity:1;pointer-events:auto;}
#toast.ok {border-left-color:var(--ok);color:var(--ok);}
#toast.err{border-left-color:var(--err);color:var(--err);}
#toast.info{border-left-color:var(--y);color:var(--txt);}

/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(255,208,0,0.18);border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:rgba(255,208,0,0.35);}

/* Run-all row */
.run-all-row{display:flex;align-items:center;gap:16px;margin-bottom:20px;}
.run-all-hint{font-size:9px;color:var(--dim);letter-spacing:1px;line-height:1.6;}

/* Collapsible advanced */
.adv-toggle {
  font-size:9px;letter-spacing:3px;color:var(--dim);
  background:none;border:none;cursor:pointer;
  padding:6px 0;
  display:flex;align-items:center;gap:6px;
}
.adv-toggle:hover{color:var(--txt);}
.adv-body{overflow:hidden;transition:max-height 0.3s;max-height:0;}
.adv-body.open{max-height:600px;}
.adv-arrow{display:inline-block;transition:transform 0.25s;font-size:8px;}
.adv-body.open + .adv-toggle .adv-arrow,
.adv-toggle.open .adv-arrow{transform:rotate(90deg);}

/* Model info row */
.model-info {
  display:flex;align-items:center;gap:10px;
  font-size:10px;margin-bottom:16px;
  padding:10px 14px;
  border:1px solid var(--bdr);
  background:var(--card);
}
.model-info .mi-dot{
  width:8px;height:8px;border-radius:50%;
  flex-shrink:0;
  background:rgba(255,68,68,0.5);
  border:1px solid rgba(255,68,68,0.3);
}
.model-info.ready .mi-dot{background:var(--ok);border-color:var(--ok);box-shadow:0 0 5px var(--ok);}
.model-info .mi-path{color:var(--dim);font-size:9px;margin-left:auto;letter-spacing:1px;}
</style>
</head>
<body>
<div id="toast"></div>

<!-- ══ HEADER ══════════════════════════════════════════════════ -->
<header class="hdr">
  <div class="hdr-logo">⬡ WAKEWORD<span>· TRAINER</span></div>
  <div class="hdr-kw">
    KEYWORD&nbsp;
    <strong id="hdr-kw">—</strong>
    <div class="hdr-model-dot" id="hdr-dot" title="Model status"></div>
  </div>
</header>

<!-- ══ TABS ════════════════════════════════════════════════════ -->
<nav class="tabs">
  <button class="tab on"  onclick="showTab('cfg')"   id="t-cfg"><span class="t-icon">⚙</span>CONFIGURE</button>
  <button class="tab"     onclick="showTab('train')"  id="t-train"><span class="t-icon">◆</span>TRAIN</button>
  <button class="tab"     onclick="showTab('test')"   id="t-test"><span class="t-icon">▶</span>TEST</button>
</nav>

<!-- ══════════════════════════════════════════════════════════════
     PANE: CONFIGURE
══════════════════════════════════════════════════════════════ -->
<div class="pane on" id="pane-cfg">

  <div class="sec">
    <div class="sec-title">WAKE WORD</div>

    <div class="field">
      <label>KEYWORD — what you will speak to activate your device</label>
      <input type="text" id="kw" class="kw-input" placeholder="bumblebee"
             autocomplete="off" spellcheck="false" oninput="onKwChange()">
    </div>

    <div class="warn hidden" id="regen-warn">
      ⚠ &nbsp;Keyword changed — delete <code>data/</code> and <code>features/</code>
      before re-running the pipeline, or the old clips will be reused.
    </div>

    <div class="field">
      <label>TRAINING PHRASES — all variants the model should recognise (one per line)</label>
      <textarea id="phrases" placeholder="bumblebee&#10;hey bumblebee&#10;ok bumblebee&#10;bumble bee"></textarea>
      <div class="hint">Include phonetic variants and natural prefixes. The bare keyword should always be listed.</div>
    </div>
  </div>

  <!-- Advanced settings (collapsible) -->
  <div class="sec">
    <div class="sec-title">
      <button class="adv-toggle open" id="adv-btn" onclick="toggleAdv()">
        <span class="adv-arrow">▶</span>&nbsp;DATA &amp; TRAINING SETTINGS
      </button>
    </div>
    <div class="adv-body open" id="adv-body">
      <div class="grid2" style="margin-bottom:14px">
        <div class="field">
          <label>POSITIVE CLIPS</label>
          <input type="number" id="n-pos" value="5000" min="500" max="20000" step="500">
          <div class="hint">More clips = better accuracy. 5000 is a good minimum.</div>
        </div>
        <div class="field">
          <label>NEGATIVE CLIPS</label>
          <input type="number" id="n-neg" value="3000" min="500" max="15000" step="500">
          <div class="hint">Non-wake-word TTS samples for the classifier to reject.</div>
        </div>
      </div>
      <div class="grid3">
        <div class="field">
          <label>EPOCHS</label>
          <input type="number" id="epochs" value="60" min="10" max="300">
          <div class="hint">Increase if val loss still falling at end.</div>
        </div>
        <div class="field">
          <label>HIDDEN SIZE</label>
          <input type="number" id="hsize" value="128" min="32" max="512" step="32">
          <div class="hint">Try 64 or 256.</div>
        </div>
        <div class="field">
          <label>LEARNING RATE</label>
          <input type="number" id="lr" value="0.001" min="0.0001" max="0.01" step="0.0001">
          <div class="hint">Default 0.001 works for most cases.</div>
        </div>
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:center;gap:16px;margin-bottom:28px">
    <button class="btn btn-y btn-lg" onclick="saveConfig()">💾 SAVE CONFIGURATION</button>
    <div id="save-msg" style="font-size:9px;letter-spacing:2px;opacity:0;transition:opacity 0.3s;color:var(--ok)"></div>
  </div>

  <!-- Pipeline reset -->
  <div class="sec">
    <div class="sec-title">PIPELINE RESET</div>
    <div style="font-size:10px;color:var(--dim);line-height:1.8;margin-bottom:16px;">
      Use these when you change the keyword or want to regenerate data from scratch.
      Steps that depend on deleted data will need to be re-run.
    </div>
    <div class="danger-row">
      <button class="btn btn-err btn-sm" onclick="cleanData()">🗑 DELETE DATA CLIPS</button>
      <button class="btn btn-err btn-sm" onclick="cleanFeatures()">🗑 DELETE FEATURES</button>
      <button class="btn btn-err btn-sm" onclick="cleanModel()">🗑 DELETE MODEL</button>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════
     PANE: TRAIN
══════════════════════════════════════════════════════════════ -->
<div class="pane" id="pane-train">

  <!-- Stats -->
  <div class="stats" id="stats">
    <div class="stat"><div class="stat-n" id="s-pos">—</div><div class="stat-l">POSITIVE CLIPS</div></div>
    <div class="stat"><div class="stat-n" id="s-neg">—</div><div class="stat-l">NEGATIVE CLIPS</div></div>
    <div class="stat"><div class="stat-n" id="s-feat">—</div><div class="stat-l">FEATURES</div></div>
    <div class="stat"><div class="stat-n" id="s-model">—</div><div class="stat-l">MODEL</div></div>
  </div>

  <div class="run-all-row">
    <button class="btn btn-y btn-lg" id="btn-all" onclick="runAll()">▶▶ RUN ALL STEPS</button>
    <div class="run-all-hint">
      Runs steps 1 → 4 sequentially<br>
      ~1–2 hours on CPU · downloads voice models on first run
    </div>
  </div>

  <!-- Step cards (generated by JS) -->
  <div id="steps-container"></div>

</div>

<!-- ══════════════════════════════════════════════════════════════
     PANE: TEST
══════════════════════════════════════════════════════════════ -->
<div class="pane" id="pane-test">

  <div class="sec">
    <div class="sec-title">LIVE DETECTION</div>

    <div class="model-info" id="mi">
      <div class="mi-dot"></div>
      <span id="mi-txt">Model not found — run training first</span>
      <span class="mi-path" id="mi-path"></span>
    </div>

    <div class="test-ctl">
      <div class="field">
        <label>MICROPHONE DEVICE</label>
        <select id="dev-sel">
          <option value="">— default device —</option>
        </select>
      </div>
      <div class="field" style="max-width:220px">
        <label>THRESHOLD &nbsp;<span id="thr-lbl" style="color:var(--y)">0.50</span></label>
        <input type="range" id="thr" min="0.1" max="0.9" step="0.05" value="0.5"
               style="accent-color:var(--y);width:100%"
               oninput="document.getElementById('thr-lbl').textContent=parseFloat(this.value).toFixed(2)">
      </div>
      <div class="btn-row">
        <button class="btn btn-ok btn-lg" id="btn-start" onclick="srvStart()">▶ START</button>
        <button class="btn btn-err btn-lg" id="btn-stop"  onclick="srvStop()" disabled>■ STOP</button>
      </div>
    </div>

    <div class="srv-status">
      <div class="srv-dot" id="srv-dot"></div>
      <span id="srv-txt">OFFLINE — configure microphone and click START</span>
      <button class="btn btn-sm" style="margin-left:auto;color:var(--dim);border-color:var(--bdr);" 
              onclick="toggleSrvLog()">📋 LOG</button>
    </div>
    <div class="step-log" id="srv-log" style="display:none;max-height:200px;margin-bottom:16px;"></div>
  </div>

  <iframe class="face-frame" id="face" src="about:blank" frameborder="0"></iframe>
</div>

<!-- ══════════════════════════════════════════════════════════════
     JAVASCRIPT
══════════════════════════════════════════════════════════════ -->
<script>
// ── Tab management ─────────────────────────────────────────────
const TABS = ['cfg','train','test'];
function showTab(id) {
  TABS.forEach(t => {
    document.getElementById('pane-'+t).classList.toggle('on', t===id);
    document.getElementById('t-'+t).classList.toggle('on', t===id);
  });
  if (id==='train') refreshStats();
  if (id==='test')  { loadDevices(); checkModel(); }
}

// ── Toast ───────────────────────────────────────────────────────
let _toastT;
function toast(msg, type='info', ms=3200) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(_toastT);
  _toastT = setTimeout(()=>el.classList.remove('show'), ms);
}

// ── Collapsible advanced ────────────────────────────────────────
function toggleAdv() {
  const body = document.getElementById('adv-body');
  const btn  = document.getElementById('adv-btn');
  body.classList.toggle('open');
  btn.classList.toggle('open');
}

// ── Config ─────────────────────────────────────────────────────
let _origKw = '';

async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const c = await r.json();
    document.getElementById('kw').value      = c.wake_word || '';
    document.getElementById('phrases').value = (c.phrases||[]).join('\n');
    document.getElementById('n-pos').value   = c.n_positive  ?? 5000;
    document.getElementById('n-neg').value   = c.n_negative  ?? 3000;
    document.getElementById('epochs').value  = c.epochs      ?? 60;
    document.getElementById('hsize').value   = c.hidden_size ?? 128;
    document.getElementById('lr').value      = c.learning_rate ?? 0.001;
    _origKw = (c.wake_word||'').toLowerCase();
    document.getElementById('hdr-kw').textContent = _origKw.toUpperCase() || '—';
  } catch(e) { toast('Failed to load config: '+e, 'err'); }
}

function onKwChange() {
  const v = document.getElementById('kw').value.trim().toLowerCase();
  document.getElementById('regen-warn').classList.toggle('hidden', v === _origKw || !v);
}

async function saveConfig() {
  const kw = document.getElementById('kw').value.trim().toLowerCase();
  const phrases = document.getElementById('phrases').value
    .split('\n').map(s=>s.trim()).filter(Boolean);
  if (!kw) { toast('Keyword cannot be empty', 'err'); return; }
  if (!phrases.length) phrases.push(kw);

  try {
    const r = await fetch('/api/config', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        wake_word: kw, phrases,
        n_positive: +document.getElementById('n-pos').value,
        n_negative: +document.getElementById('n-neg').value,
        epochs:     +document.getElementById('epochs').value,
        hidden_size:+document.getElementById('hsize').value,
        learning_rate:+document.getElementById('lr').value,
      })
    });
    const d = await r.json();
    if (d.ok) {
      _origKw = kw;
      document.getElementById('hdr-kw').textContent = kw.toUpperCase();
      document.getElementById('regen-warn').classList.add('hidden');
      const msg = document.getElementById('save-msg');
      msg.textContent = '✓ SAVED'; msg.style.opacity = '1';
      setTimeout(()=>msg.style.opacity='0', 2400);
      toast('Configuration saved', 'ok');
    } else {
      toast('Save failed: '+(d.error||d.message), 'err');
    }
  } catch(e) { toast('Save error: '+e, 'err'); }
}

// ── Cleanup ─────────────────────────────────────────────────────
async function cleanData() {
  if (!confirm('Delete all positive and negative WAV clips?\nThis will require re-running steps 1–4.')) return;
  const r = await fetch('/api/clean/data', {method:'DELETE'});
  const d = await r.json();
  toast(d.ok ? '🗑 Data clips deleted' : 'Error: '+d.error, d.ok?'ok':'err');
  if (d.ok) refreshStats();
}
async function cleanFeatures() {
  if (!confirm('Delete feature .npy files?\nSteps 3–4 will need to re-run.')) return;
  const r = await fetch('/api/clean/features', {method:'DELETE'});
  const d = await r.json();
  toast(d.ok ? '🗑 Features deleted' : 'Error: '+d.error, d.ok?'ok':'err');
  if (d.ok) refreshStats();
}
async function cleanModel() {
  if (!confirm('Delete the ONNX model?\nStep 4 will need to re-run.')) return;
  const r = await fetch('/api/clean/model', {method:'DELETE'});
  const d = await r.json();
  toast(d.ok ? '🗑 Model deleted' : 'Error: '+d.error, d.ok?'ok':'err');
  if (d.ok) refreshStats();
}

// ── Step cards builder ──────────────────────────────────────────
const STEP_META = {
  1:{title:'GENERATE POSITIVE CLIPS', desc:'Piper TTS · wake-word samples',       eta:'~5–15 min'},
  2:{title:'GENERATE NEGATIVE CLIPS',  desc:'Piper TTS · non-wake-word samples',   eta:'~3–10 min'},
  3:{title:'EXTRACT FEATURES',         desc:'openWakeWord backbone · all clips',   eta:'~45 min ☕'},
  4:{title:'TRAIN &amp; EXPORT MODEL', desc:'Classifier training · ONNX export',  eta:'~5–15 min'},
};

function buildStepCards() {
  const c = document.getElementById('steps-container');
  c.innerHTML = '';
  for (let n = 1; n <= 4; n++) {
    const m = STEP_META[n];
    c.insertAdjacentHTML('beforeend', `
    <div class="step" id="step-${n}">
      <div class="step-hdr" onclick="toggleLog(${n})">
        <div class="step-n" id="sn-${n}">${n}</div>
        <div class="step-info">
          <div class="step-title">${m.title}</div>
          <div class="step-desc">${m.desc} &nbsp;·&nbsp; <span style="color:rgba(51,170,255,0.6)">${m.eta}</span></div>
        </div>
        <span class="step-tag tag-idle" id="tag-${n}">IDLE</span>
        <span class="elapsed" id="ela-${n}"></span>
        <div class="step-btns">
          <button class="btn btn-y btn-sm" id="rbtn-${n}"
                  onclick="event.stopPropagation();runStep(${n})">▶ RUN</button>
          <button class="btn btn-err btn-sm" id="sbtn-${n}"
                  onclick="event.stopPropagation();stopStep(${n})" style="display:none">■ STOP</button>
        </div>
      </div>
      <div class="step-prog"><div class="step-prog-fill" id="prog-${n}"></div></div>
      <div class="step-log" id="log-${n}"></div>
    </div>`);
  }
}

// ── Step state management ───────────────────────────────────────
function setStepState(n, state) {
  const card = document.getElementById('step-'+n);
  card.className = 'step ' + state;

  const tag   = document.getElementById('tag-'+n);
  const lbl   = {idle:'IDLE',pending:'PENDING',running:'RUNNING',done:'DONE',error:'ERROR',stopped:'STOPPED'};
  tag.textContent = lbl[state] || state.toUpperCase();
  tag.className = 'step-tag tag-'+(state==='pending'?'idle':state);

  document.getElementById('rbtn-'+n).style.display = state==='running' ? 'none' : '';
  document.getElementById('sbtn-'+n).style.display = state==='running' ? ''     : 'none';

  const p = document.getElementById('prog-'+n);
  p.className = 'step-prog-fill';
  if (state==='running') {
    p.classList.add('ind');
  } else if (state==='done') {
    p.classList.add('full'); p.style.background='var(--ok)';
  } else if (state==='error') {
    p.classList.add('full'); p.style.background='var(--err)';
  } else {
    p.style.width='0%'; p.style.background='var(--y)';
  }
}

function setProgress(n, pct) {
  const p = document.getElementById('prog-'+n);
  p.classList.remove('ind');
  p.style.width = pct+'%';
  p.style.background = 'var(--y)';
}

function appendLog(n, line) {
  const el = document.getElementById('log-'+n);
  const cls =
    line.match(/✓|Done|✗.*ok|Completed|Loaded|saved|success/i) ? 'ok' :
    line.match(/✗|Error|error|Traceback|failed|FAILED/i)        ? 'err' :
    line.match(/==|STEP|Step|Loading|Training|Epoch/i)          ? 'hi'  : '';
  const div = Object.assign(document.createElement('div'), {className:`log-ln ${cls}`, textContent:line});
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;

  // Parse epoch progress for step 4
  const ep = line.match(/Epoch\s+(\d+)\/(\d+)/);
  if (ep && n===4) setProgress(4, Math.round(+ep[1] / +ep[2] * 100));
}

function clearLog(n) {
  document.getElementById('log-'+n).innerHTML = '';
  document.getElementById('ela-'+n).textContent = '';
}

function toggleLog(n) {
  document.getElementById('log-'+n).classList.toggle('open');
}

// ── Run step ────────────────────────────────────────────────────
let _autoChain = false;
const _es = {};

async function runStep(n) {
  clearLog(n);
  document.getElementById('log-'+n).classList.add('open');
  setStepState(n, 'running');
  appendLog(n, `▶ Starting step ${n}…`);

  const r = await fetch(`/api/run/${n}`, {method:'POST'});
  const d = await r.json();
  if (!d.ok) {
    appendLog(n, '✗ ' + (d.message||d.error), 'err');
    setStepState(n, 'error');
    _autoChain = false;
    return;
  }

  if (_es[n]) { _es[n].close(); }
  const es = new EventSource(`/api/run/${n}/stream`);
  _es[n] = es;

  es.onmessage = e => {
    const line = e.data;
    if (line === '[KA]') return;
    if (line.startsWith('[DONE:')) {
      es.close(); delete _es[n];
      const rc = parseInt(line.match(/\d+/)[0]);
      if (rc === 0) {
        setStepState(n, 'done');
        appendLog(n, `✓ Step ${n} completed`);
        refreshStats();
        if (_autoChain && n < 4) {
          setTimeout(()=>runStep(n+1), 600);
        } else {
          _autoChain = false;
        }
      } else {
        setStepState(n, 'error');
        appendLog(n, `✗ Exited with code ${rc}`);
        _autoChain = false;
      }
      return;
    }
    appendLog(n, line);
  };

  es.onerror = () => { es.close(); delete _es[n]; };
}

async function stopStep(n) {
  if (_es[n]) { _es[n].close(); delete _es[n]; }
  _autoChain = false;
  const r = await fetch(`/api/run/${n}/stop`, {method:'POST'});
  setStepState(n, 'stopped');
  appendLog(n, '■ Stopped');
  toast(`Step ${n} stopped`, 'info');
}

function runAll() {
  _autoChain = true;
  runStep(1);
}

// ── Stats ───────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();

    document.getElementById('s-pos').textContent   = s.positive_count.toLocaleString();
    document.getElementById('s-neg').textContent   = s.negative_count.toLocaleString();
    document.getElementById('s-feat').textContent  = s.features_exist ? '✓' : '—';
    document.getElementById('s-model').textContent = s.model_exists   ? '✓' : '—';

    document.getElementById('hdr-dot').classList.toggle('ready', s.model_exists);

    // Restore step states for finished steps (page reload resilience)
    for (const [sn, info] of Object.entries(s.steps||{})) {
      const st = info.status || info;
      if (st && st !== 'idle' && st !== 'pending' && st !== 'running') {
        setStepState(parseInt(sn), st);
        if (info.elapsed) {
          document.getElementById('ela-'+sn).textContent =
            `${Math.floor(info.elapsed/60)}m ${info.elapsed%60}s`;
        }
      }
    }
  } catch(e) {}
}

// ── Test: devices & model check ──────────────────────────────────
async function loadDevices() {
  try {
    const r = await fetch('/api/devices');
    const devs = await r.json();
    const sel = document.getElementById('dev-sel');
    sel.innerHTML = '<option value="">— default device —</option>';
    for (const d of devs) {
      if (d.error) { sel.innerHTML += `<option disabled>${d.error}</option>`; continue; }
      const tag = d.default ? ' ← default' : '';
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = `[${d.index}]  ${d.name}${tag}`;
      if (d.name.toLowerCase().includes('respeaker') || d.name.toLowerCase().includes('array'))
        opt.textContent += ' 🎤';
      sel.appendChild(opt);
    }
  } catch(e) {}
}

async function checkModel() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const mi = document.getElementById('mi');
    const kw = document.getElementById('hdr-kw').textContent.toLowerCase();
    if (s.model_exists) {
      mi.classList.add('ready');
      document.getElementById('mi-txt').textContent = 'Model ready';
      document.getElementById('mi-path').textContent = `model/${kw}.onnx`;
      document.getElementById('btn-start').disabled = false;
    } else {
      mi.classList.remove('ready');
      document.getElementById('mi-txt').textContent = 'Model not found — run training first';
      document.getElementById('mi-path').textContent = '';
      document.getElementById('btn-start').disabled = true;
    }
  } catch(e) {}
}

function toggleSrvLog() {
  const el = document.getElementById('srv-log');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
  fetchSrvLog();
}

async function fetchSrvLog() {
  try {
    const r = await fetch('/api/server/log');
    const d = await r.json();
    const el = document.getElementById('srv-log');
    el.innerHTML = (d.lines||[]).map(l => `<div class="log-ln">${esc(l)}</div>`).join('');
    el.scrollTop = el.scrollHeight;
  } catch(e) {}
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Server ──────────────────────────────────────────────────────
async function srvStart() {
  const dev = document.getElementById('dev-sel').value;
  const thr = parseFloat(document.getElementById('thr').value);
  document.getElementById('srv-txt').textContent = 'LAUNCHING…';
  const r = await fetch('/api/server/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({device: dev ? parseInt(dev) : null, threshold: thr})
  });
  const d = await r.json();
  if (d.ok) {
    setSrvState(true, `LISTENING · threshold ${thr.toFixed(2)} · PID ${d.pid}`);
    reloadFace();
    toast('Detection server started', 'ok');
  } else {
    setSrvState(false, 'OFFLINE — server failed to start');
    const el = document.getElementById('srv-log');
    el.style.display = 'block';
    el.innerHTML = '<div class="log-ln err">✗ SERVER CRASHED AT STARTUP</div>';
    if (d.log) {
      d.log.forEach(l => {
        el.innerHTML += `<div class="log-ln${l.toLowerCase().includes('error')||l.toLowerCase().includes('traceback')?' err':''}">${esc(l)}</div>`;
      });
    }
    el.innerHTML += `<div class="log-ln" style="color:var(--dim)">${esc(d.message||'')}</div>`;
    el.scrollTop = el.scrollHeight;
    toast('Server error — check log panel', 'err', 8000);
  }
  fetchSrvLog();
}

async function srvStop() {
  await fetch('/api/server/stop', {method:'POST'});
  setSrvState(false, 'OFFLINE — click START to begin listening');
  toast('Server stopped', 'info');
  fetchSrvLog();
}

function setSrvState(on, txt) {
  document.getElementById('srv-dot').classList.toggle('on', on);
  document.getElementById('srv-txt').textContent = txt;
  document.getElementById('btn-start').disabled = on;
  document.getElementById('btn-stop').disabled  = !on;
}

function reloadFace() {
  document.getElementById('face').src = '/face?' + Date.now();
}

// Poll server status every 3 s
setInterval(async () => {
  try {
    const r = await fetch('/api/server/status');
    const s = await r.json();
    document.getElementById('srv-dot').classList.toggle('on', s.running);
    document.getElementById('btn-start').disabled = s.running;
    document.getElementById('btn-stop').disabled  = !s.running;
    if (document.getElementById('srv-log').style.display !== 'none') {
      fetchSrvLog();
    }
  } catch(e) {}
}, 3000);

// ── Init ────────────────────────────────────────────────────────
window.onload = async () => {
  buildStepCards();
  await loadConfig();
  refreshStats();
  setInterval(refreshStats, 12000);
};
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │   Wake Word Trainer  ·  web UI               │")
    print("  │   http://localhost:5000                       │")
    print("  │                                               │")
    print("  │   Ctrl+C to stop                             │")
    print("  └─────────────────────────────────────────────┘")
    print()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)