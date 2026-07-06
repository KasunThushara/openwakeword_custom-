"""
doa_reader.py
=============
Reads Direction-of-Arrival, voice activity, beam azimuths, and
speech energy from the Seeed ReSpeaker XVF3800 4-Mic Array via
raw USB control transfers (VID 0x2886, PID 0x001A).

USB control (endpoint 0, this module) and USB audio (isochronous,
ALSA / PortAudio) are separate endpoints — they coexist.  This module
does NOT touch the kernel driver, so the audio stream keeps working.

Usage:
    reader = DOAReader()
    data = reader.read()
    # → {"doa": 45, "vad": 1, "azimuths": [10, 80, 45, 45],
    #    "energy": [0.12, 0.08, 0.55, 0.78]}
    reader.close()
"""

import math
import struct
import threading
import time

import usb.core
import usb.util

VID_RESPEAKER = 0x2886
PID_XVF3800   = 0x001E       # reSpeaker Flex XVF3800 C16K6Ch
PID_FALLBACKS = [0x001A]     # older XVF3800 firmware variants
TIMEOUT       = 100_000

CONTROL_SUCCESS        = 0
SERVICER_COMMAND_RETRY = 0x40
MAX_RETRIES            = 100

# (resid, cmdid, count, access, dtype)
PARAMS = {
    "VERSION":            (48,  0, 3, "ro", "uint8"),
    "DOA_VALUE":          (20, 18, 2, "ro", "uint16"),    # [doa_angle, vad_flag]
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians"),   # [b1, b2, free, auto] azimuth
    "AEC_SPENERGY_VALUES":(33, 80, 4, "ro", "float"),     # [b1, b2, free, auto] energy
}


def _rad2deg(r: float) -> float:
    if math.isnan(r) or math.isinf(r):
        return 0.0
    return round(math.degrees(r) % 360, 1)


class DOAReader:
    """USB control reader for ReSpeaker XVF3800 DOA / VAD / beam data."""

    def __init__(self, vid=VID_RESPEAKER, pid=PID_XVF3800):
        dev = usb.core.find(idVendor=vid, idProduct=pid)
        if dev is None:
            for fallback_pid in PID_FALLBACKS:
                dev = usb.core.find(idVendor=vid, idProduct=fallback_pid)
                if dev is not None:
                    break
        if dev is None:
            raise RuntimeError(
                f"No ReSpeaker found (VID=0x{vid:04x}, PID=0x{pid:04x}). "
                f"Ensure the device is connected and /dev/bus/usb is mounted."
            )
        self._dev = dev
        self._lock = threading.Lock()

    def read(self):
        """
        Read DOA + VAD + beam azimuths + speech energy.
        Returns a dict or None on error.

        {"doa": int, "vad": int, "azimuths": [float,...], "energy": [float,...]}
        """
        try:
            doa_raw = self._read("DOA_VALUE")
            doa = int(doa_raw[0])
            vad = int(doa_raw[1])

            az_raw = self._read("AEC_AZIMUTH_VALUES")
            azimuths = [_rad2deg(r) for r in az_raw]

            try:
                en_raw = self._read("AEC_SPENERGY_VALUES")
                energy = [max(0.0, min(1.0, float(e))) for e in en_raw]
            except Exception:
                energy = [0.0, 0.0, 0.0, 0.0]

            return {
                "doa":      doa,
                "vad":      vad,
                "azimuths": azimuths,
                "energy":   energy,
            }
        except (usb.core.USBError, struct.error, IndexError, TimeoutError) as exc:
            return None

    def _read(self, name: str):
        p     = PARAMS[name]
        resid = p[0]
        cmdid = 0x80 | p[1]
        count = p[2]
        dtype = p[4]

        if dtype in ("uint8", "char"):
            length = count + 1
        elif dtype in ("float", "radians", "uint32", "int32"):
            length = count * 4 + 1
        elif dtype == "uint16":
            length = count * 2 + 1
        else:
            length = count * 4 + 1

        with self._lock:
            for _ in range(MAX_RETRIES):
                resp = self._dev.ctrl_transfer(
                    usb.util.CTRL_IN |
                    usb.util.CTRL_TYPE_VENDOR |
                    usb.util.CTRL_RECIPIENT_DEVICE,
                    0, cmdid, resid, length, TIMEOUT,
                )
                status = resp[0]
                if status == CONTROL_SUCCESS:
                    break
                if status == SERVICER_COMMAND_RETRY:
                    time.sleep(0.01)
                    continue
                raise IOError(f"Status 0x{status:02X} for '{name}'")
            else:
                raise TimeoutError(f"RETRY loop exhausted for '{name}'")

        raw = resp.tobytes()  # raw[0] = status byte already consumed

        if dtype == "uint8":
            return list(struct.unpack("<" + "B" * count, raw[1:1 + count]))
        elif dtype in ("float", "radians"):
            return list(struct.unpack("<" + "f" * count, raw[1:1 + count * 4]))
        elif dtype == "uint16":
            return list(struct.unpack("<" + "H" * count, raw[1:1 + count * 2]))
        elif dtype == "uint32":
            return list(struct.unpack("<" + "I" * count, raw[1:1 + count * 4]))
        elif dtype == "int32":
            return list(struct.unpack("<" + "i" * count, raw[1:1 + count * 4]))
        return list(raw[1:])

    def version(self):
        try:
            ver = self._read("VERSION")
            return f"{ver[0]}.{ver[1]}.{ver[2]}"
        except Exception:
            return "unknown"

    def close(self):
        usb.util.dispose_resources(self._dev)
        self._dev = None
