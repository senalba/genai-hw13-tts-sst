#!/usr/bin/env python3
# tts_script.py — StyleTTS2 (Ukrainian) via Hugging Face Spaces / gradio_client
# Usage examples:
#   # 1) Simple (one sentence)
#   python tts_script.py --text "Привіт, світ" --outdir out
#
#   # 2) Batch from CSV (columns: id?, text, voice_name?, model_name?, speed?)
#   python tts_script.py --in data.csv --outdir out
#
#   # 3) Batch from JSON (list of objects with the same fields)
#   python tts_script.py --in data.json --outdir out
#
# Optional flags:
#   --model-name multi            (default used for /synthesize)
#   --voice-name "Артем Окороков" (default voice; depends on space)
#   --speed 1.0                   (float, 0.5..2.0 usually)
#   --api verbalize|synthesize    (auto: uses 'synthesize' if any voice/model/speed is set)
#   --endpoint patriotyk/styletts2-ukrainian  (HF space id)
#
# Dependencies:
#   pip install gradio_client requests soundfile

import argparse
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from gradio_client import Client


# ---------- utils ----------


def slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s]+", "_", s.strip())
    s = s[:maxlen]
    return s or "utt"


def read_items(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                items.append(
                    {
                        "id": (row.get("id") or "").strip(),
                        "text": text,
                        "voice_name": (row.get("voice_name") or "").strip() or None,
                        "model_name": (row.get("model_name") or "").strip() or None,
                        "speed": float(row["speed"]) if row.get("speed") else None,
                    }
                )
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of objects or have key 'items'.")
        for obj in data:
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            items.append(
                {
                    "id": (obj.get("id") or "").strip(),
                    "text": text,
                    "voice_name": (obj.get("voice_name") or None),
                    "model_name": (obj.get("model_name") or None),
                    "speed": (
                        float(obj["speed"]) if obj.get("speed") is not None else None
                    ),
                }
            )
    else:
        raise ValueError("Only .csv or .json are supported")
    return items


def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:audio/")


def _maybe_decode_data_uri_to_bytes(s: str) -> Optional[bytes]:
    if not _is_data_uri(s):
        return None
    try:
        # expected format: data:audio/wav;base64,<payload>
        b64 = s.split(",", 1)[1]
        return base64.b64decode(b64)
    except Exception:
        return None


def _maybe_download(url: str) -> Optional[bytes]:
    if not (isinstance(url, str) and url.startswith("http")):
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def save_audio_result(result: Any, out_path: Path) -> Path:
    """
    Attempts to persist whatever the Space returns:
      - data URI base64 (string)
      - HTTP URL (string)
      - local temp file path (string)
      - dict with "audio" (data URI) or "path"/"filepath"
    Returns the final path written.
    """
    # 1) dict variants
    if isinstance(result, dict):
        # common: {"audio": "data:audio/wav;base64,...."}
        if "audio" in result and isinstance(result["audio"], str):
            data = _maybe_decode_data_uri_to_bytes(result["audio"])
            if data:
                out_path.write_bytes(data)
                return out_path
        # some spaces: {"filepath": "/tmp/xyz.wav"} or {"path": "/tmp/xyz.wav"}
        for k in ("filepath", "path"):
            p = result.get(k)
            if isinstance(p, str) and Path(p).exists():
                return _copy_file(Path(p), out_path)

    # 2) string variants
    if isinstance(result, str):
        # data URI
        data = _maybe_decode_data_uri_to_bytes(result)
        if data:
            out_path.write_bytes(data)
            return out_path
        # remote URL
        data = _maybe_download(result)
        if data:
            out_path.write_bytes(data)
            return out_path
        # local path
        p = Path(result)
        if p.exists() and p.is_file():
            return _copy_file(p, out_path)

    # 3) unsupported
    raise ValueError("Unrecognized result format from space; cannot save audio.")


def _copy_file(src: Path, dst: Path) -> Path:
    if src.resolve() == dst.resolve():
        return dst
    dst.write_bytes(src.read_bytes())
    return dst


# ---------- TTS call(s) ----------


def call_verbalize(client: Client, text: str) -> Any:
    # Minimal endpoint that just takes text
    return client.predict(text=text, api_name="/verbalize")


def call_synthesize(
    client: Client,
    text: str,
    model_name: Optional[str],
    voice_name: Optional[str],
    speed: Optional[float],
) -> Any:
    # The space expects at least model_name + text; voice_name/speed optional
    payload = {
        "model_name": model_name or "multi",
        "text": text,
        "speed": speed if speed is not None else 1.0,
        "voice_name": voice_name or "Артем Окороков",
    }
    return client.predict(**payload, api_name="/synthesize")


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="One-shot text to synthesize")
    ap.add_argument(
        "--in", dest="in_path", type=Path, help="CSV/JSON with rows of text/params"
    )
    ap.add_argument(
        "--outdir", type=Path, required=True, help="Output directory for .wav files"
    )
    ap.add_argument(
        "--endpoint",
        default="patriotyk/styletts2-ukrainian",
        help="Hugging Face Space id",
    )
    ap.add_argument(
        "--api",
        choices=["verbalize", "synthesize"],
        help="Force specific API; default=auto",
    )
    ap.add_argument(
        "--model-name", help="Default model_name for synthesize (e.g., 'multi')"
    )
    ap.add_argument("--voice-name", help="Default voice_name")
    ap.add_argument("--speed", type=float, help="Default speed (e.g., 1.0)")
    args = ap.parse_args()

    if not args.text and not args.in_path:
        ap.error("Provide --text or --in")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # init client once
    client = Client(args.endpoint)

    def run_one(
        text: str,
        out_file: Path,
        model_name: Optional[str] = None,
        voice_name: Optional[str] = None,
        speed: Optional[float] = None,
    ):
        # choose API
        use_api = args.api
        if not use_api:
            # auto: if any of voice/model/speed provided (globally or locally) -> synthesize, else verbalize
            any_adv = (
                model_name
                or args.model_name
                or voice_name
                or args.voice_name
                or speed is not None
                or args.speed is not None
            )
            use_api = "synthesize" if any_adv else "verbalize"

        if use_api == "verbalize":
            result = call_verbalize(client, text=text)
        else:
            result = call_synthesize(
                client,
                text=text,
                model_name=model_name or args.model_name or "multi",
                voice_name=voice_name or args.voice_name or "Артем Окороков",
                speed=(
                    speed
                    if speed is not None
                    else (args.speed if args.speed is not None else 1.0)
                ),
            )

        save_audio_result(result, out_file)
        print(out_file)

    # single text mode
    if args.text:
        name = slugify(args.text[:50])
        out = args.outdir / f"{name}.wav"
        run_one(args.text, out)
        return

    # batch mode
    items = read_items(args.in_path)
    if not items:
        print("No valid rows in input.", file=sys.stderr)
        sys.exit(1)

    for i, it in enumerate(items, 1):
        name = it.get("id") or slugify(it["text"][:50])
        out = args.outdir / f"{i:03d}_{name}.wav"
        run_one(
            text=it["text"],
            out_file=out,
            model_name=it.get("model_name"),
            voice_name=it.get("voice_name"),
            speed=it.get("speed"),
        )


if __name__ == "__main__":
    main()
