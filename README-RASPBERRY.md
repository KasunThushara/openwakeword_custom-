# Bumblebee Wake Word — Raspberry Pi 5 Deployment

Two-stage wake-word + command detection with OpenWakeWord, ReSpeaker DOA/VAD/beamforming, and an animated face dashboard — all in Docker.

---

## Prerequisites

- Raspberry Pi 5 (4 GB+ recommended)
- USB ReSpeaker Flex XVF3800 C16K6Ch connected
- Headphones or speakers plugged into the ReSpeaker's **3.5 mm audio jack**
- Internet connection (for image pull and model downloads on first run)

---

## 1. Install Docker on the Pi

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, or run:
newgrp docker
```

Verify:

```bash
docker --version
```

---

## 2. Deploy (2-minute setup)

```bash
mkdir -p ~/bumblebee && cd ~/bumblebee
```

### Create asound.conf — enable dmix (software mixing) on the ReSpeaker

The ReSpeaker is card 2 on the Pi 5.  dmix lets the audio capture stream and
playback share the same USB hardware.

```bash
cat > asound.conf << 'EOF'
pcm.!default {
    type dmix
    ipc_key 1024
    slave {
        pcm "hw:2,0"
        rate 16000
        channels 2
        period_time 0
        period_size 1024
        buffer_size 4096
    }
}
ctl.!default {
    type hw
    card 2
}
EOF
```

### Create docker-compose.yml

```bash
cat > docker-compose.yml << 'EOF'
services:
  server:
    image: kasunt96/bumblebee-wakeword:latest
    container_name: bumblebee-server
    command: python3 bumblebee_server.py --device respeaker --threshold 0.75
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb:/dev/bus/usb
    group_add:
      - audio
    volumes:
      - ./asound.conf:/etc/asound.conf:ro
    environment:
      - PYTHONUNBUFFERED=1
    network_mode: host
    restart: unless-stopped

  trainer:
    image: kasunt96/bumblebee-wakeword:latest
    container_name: bumblebee-trainer
    command: python3 app.py
    ports:
      - "5000:5000"
    restart: unless-stopped
EOF
```

### Start everything

```bash
docker compose up -d
```

---

## 3. Verify

Check both containers are running:

```bash
docker ps
# Should show: bumblebee-server AND bumblebee-trainer
```

Check the server started correctly:

```bash
docker logs bumblebee-server 2>&1 | head -20
```

Look for:
```
Loading 5 model(s) from /app/model:
  activate_party_mode.onnx
  back_to_normal_mode.onnx
  bumblebee.onnx
  play_spanish_music.onnx
  wave_your_antenna.onnx
Audio stream started — listening for wake word
```

Open the face dashboard — from any device on the same network:

```
http://<pi-ip-address>:5000/face
```

Replace `<pi-ip-address>` with your Pi's IP (run `hostname -I` to find it).

---

## 4. Usage

| Wake word / command | Action |
|---------------------|--------|
| **"bumblebee"** | Wake word — plays robot sound, face animates, activates command listening (6 seconds) |
| **"activate party mode"** | Lights → party mode (face goes rainbow) |
| **"back to normal mode"** | Lights → normal (stops party/relax mode) |
| **"play spanish music"** | Sound → plays spanish.mp3 through ReSpeaker audio jack |
| **"wave your antenna"** | Gesture → face shakes its antenna horns |

### Workflow

1. Say **"bumblebee"** → robot sound plays → face shows **BUMBLEBEE!** → 6-second command window opens
2. Within 6 seconds, say one of the command phrases above
3. If no command is spoken within 6 seconds → timeout → back to standby

### Dashboard

The face page (`http://<pi-ip>:5000/face`) shows:
- **DOA Angle** — direction of arrival from the ReSpeaker's 4-mic array
- **Voice Activity** — SILENCE / SPEECH indicator
- **4 Beam energy bars** — per-beam azimuth and energy level
- **Compass** — polar DOA display with radar sweep
- **DOA Histogram** — last 200 readings distribution
- **Bumblebee face** — animates on wake word detection

---

## 5. View logs

```bash
# Server (detection) logs
docker logs -f bumblebee-server

# Trainer (web UI) logs
docker logs -f bumblebee-trainer
```

---

## 6. Stop / Restart / Update

```bash
# Stop
docker compose down

# Start
docker compose up -d

# Update to latest image
docker compose down
docker pull kasunt96/bumblebee-wakeword:latest
docker compose up -d
```

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| No DOA data on the dashboard | Run `docker compose down && docker compose up -d` — USB devices sometimes need a clean restart |
| No sound from wake word | Check headphones are in ReSpeaker's jack (NOT the Pi's jack) |
| Sound crackling or choppy | The dmix slave rate must be 16000. Verify `asound.conf` is correct |
| `aplay: audio open error` | ReSpeaker must be card 2. Run `docker compose run --rm server aplay -l` to check |
| Face page blank | Wait 10 seconds after startup — openWakeWord pre-trained models download on first run |
| `No ReSpeaker found` | USB cable loose? Run `docker compose run --rm server lsusb` — should show `2886:001e` |
| Low wake-word detection | Lower threshold: change `--threshold 0.75` → `--threshold 0.5` in `docker-compose.yml` |
| Port 5000 already in use | The Pi may have another web server. Change `"5000:5000"` → `"8080:5000"` in the compose file |

---

## 8. How it works (architecture)

```
┌──────────────────────┐
│ bumblebee_server.py  │  WebSocket on :8767
│                      │
│  WAITING state:      │  Listens for "bumblebee" (wake word ONNX model)
│  COMMAND state:      │  6-second window for command ONNX models
│                      │
│  DOA reader:         │  USB control transfers → ReSpeaker registers
│  Sound playback:     │  aplay (WAV) / mpg123 (MP3)
└──────────────────────┘
         │
         │  {"type":"wake"} / {"type":"doa":{...}} / {"type":"intent":{...}}
         ▼
┌──────────────────────┐
│ bumblebee_face.html  │  Browser dashboard
│                      │
│  Animated Autobot    │  Compass, DOA histogram, beam energy bars
│  Party/Relax modes   │  Demo mode when server disconnected
└──────────────────────┘
```

All 5 ONNX models are baked into the Docker image.  The image is multi-arch
(linux/amd64 + linux/arm64) — the same `docker pull` works on both desktop
PCs and Raspberry Pi.

---

## 9. Building the image yourself (optional)

If you want to build from source instead of pulling from Docker Hub:

```bash
git clone <your-repo-url> ~/OpenWakeword_1
cd ~/OpenWakeword_1
docker compose build --no-cache
docker compose --profile detection up server
```

For cross-platform builds (push to your own Docker Hub):

```bash
docker buildx create --use --name multiarch
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <your-user>/bumblebee-wakeword:latest --push .
```
