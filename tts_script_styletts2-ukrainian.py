#!/usr/bin/env python3
# tts_script_styletts2-ukrainian.py
# Unified TTS:
#  - UKR via HF Space "patriotyk/styletts2-ukrainian" (gradio_client)
#  - EN (British) via Parler-TTS (local)
#
# Examples:
#   # Single line (Space, UA)
#   python tts_script_styletts2-ukrainian.py --text "Привіт! Тест." --outdir out
#   # Single line (Parler, EN)
#   python tts_script_styletts2-ukrainian.py --text "Good evening." --outdir out --engine parler
#   # Batch JSON/CSV mixed (per-row routing by "engine" or "language")
#   python tts_script_styletts2-ukrainian.py --in franko.json --outdir franko-modern

import argparse, base64, csv, json, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------- common utils ---------
def slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s]+", "_", s.strip())
    return s[:maxlen] or "utt"


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
                        "voice_name": (row.get("voice_name") or None),
                        "model_name": (row.get("model_name") or None),
                        "speed": float(row["speed"]) if row.get("speed") else None,
                        "language": (row.get("language") or None),
                        "engine": (row.get("engine") or None),  # "space" | "parler"
                    }
                )
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise ValueError("JSON must be a list or have key 'items'.")
        for obj in data:
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            items.append(
                {
                    "id": (obj.get("id") or "").strip(),
                    "text": text,
                    "voice_name": obj.get("voice_name"),
                    "model_name": obj.get("model_name"),
                    "speed": (
                        float(obj["speed"]) if obj.get("speed") is not None else None
                    ),
                    "language": obj.get("language"),
                    "engine": obj.get("engine"),
                }
            )
    else:
        raise ValueError("Only .csv or .json are supported")
    return items


def write_bytes(out_path: Path, data: bytes) -> Path:
    out_path.write_bytes(data)
    return out_path


# --------- Space (UA) helpers ---------
def _is_data_uri(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:audio/")


def _decode_data_uri(s: str) -> Optional[bytes]:
    try:
        b64 = s.split(",", 1)[1]
        import base64 as _b

        return _b.b64decode(b64)
    except Exception:
        return None


def _maybe_download(url: str) -> Optional[bytes]:
    if not (isinstance(url, str) and url.startswith("http")):
        return None
    try:
        import requests

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def space_save_result(result: Any, out_path: Path) -> Path:
    """
    Persist Space outputs:
    - bytes / bytearray
    - dict with audio/url/path/filepath/data
    - list/tuple -> first valid
    - objects with .url/.path/.filepath/.name
    - string:
        * data:audio/...;base64,...
        * http(s)://...
        * local path
        * JSON-like string (parse then recurse)
        * strings containing 'file=/...wav' or embedded URL
    """
    # 0) bytes
    if isinstance(result, (bytes, bytearray)):
        out_path.write_bytes(result)
        return out_path

    # 1) dict
    if isinstance(result, dict):
        val = (
            result.get("audio")
            or result.get("url")
            or result.get("path")
            or result.get("filepath")
        )
        if isinstance(val, str):
            if _is_data_uri(val):
                data = _decode_data_uri(val)
                if data:
                    out_path.write_bytes(data)
                    return out_path
            data = _maybe_download(val)
            if data:
                out_path.write_bytes(data)
                return out_path
            p = Path(val)
            if p.exists() and p.is_file():
                out_path.write_bytes(p.read_bytes())
                return out_path
        if "data" in result:
            return space_save_result(result["data"], out_path)

    # 2) list/tuple
    if isinstance(result, (list, tuple)):
        for item in result:
            try:
                return space_save_result(item, out_path)
            except Exception:
                pass

    # 3) object with attrs
    for attr in ("url", "path", "filepath", "name"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, str):
                if _is_data_uri(val):
                    data = _decode_data_uri(val)
                    if data:
                        out_path.write_bytes(data)
                        return out_path
                data = _maybe_download(val)
                if data:
                    out_path.write_bytes(data)
                    return out_path
                p = Path(val)
                if p.exists() and p.is_file():
                    out_path.write_bytes(p.read_bytes())
                    return out_path

    # 4) string
    if isinstance(result, str):
        s = result.strip()

        # 4a) data URI
        if _is_data_uri(s):
            data = _decode_data_uri(s)
            if data:
                out_path.write_bytes(data)
                return out_path

        # 4b) looks like JSON? try parse & recurse
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                return space_save_result(json.loads(s), out_path)
            except Exception:
                pass

        # 4c) embedded URL
        import re as _re

        m = _re.search(r"https?://\S+", s)
        if m:
            url = m.group(0).rstrip("',\") ]}")
            data = _maybe_download(url)
            if data:
                out_path.write_bytes(data)
                return out_path

        # 4d) file=/… or path=/…
        m2 = _re.search(r"(?:file|path|filepath)\s*=\s*([^\s,'\")]+)", s)
        if m2:
            p = Path(m2.group(1))
            if p.exists() and p.is_file():
                out_path.write_bytes(p.read_bytes())
                return out_path

        # 4e) treat as local path
        p = Path(s)
        if p.exists() and p.is_file():
            out_path.write_bytes(p.read_bytes())
            return out_path

    raise ValueError(
        f"Unrecognized Space result; cannot save audio. Got type={type(result)} and value preview={str(result)[:160]!r}"
    )


def space_call(
    endpoint: str,
    api: str,
    text: str,
    model_name: Optional[str],
    voice_name: Optional[str],
    speed: Optional[float],
) -> Any:
    from gradio_client import Client

    client = Client(endpoint)
    if api == "verbalize":
        return client.predict(text=text, api_name="/verbalize")
    payload = {
        "model_name": model_name or "multi",
        "text": text,
        "speed": speed if speed is not None else 1.0,
        "voice_name": voice_name or "Артем Окороков",
    }
    return client.predict(**payload, api_name="/synthesize")


# --------- Parler (EN British) helpers ---------
class ParlerBritish:
    _model = None
    _tok = None
    _sr = None

    @classmethod
    def ensure_loaded(cls, model_id: str = "parler-tts/parler_tts_mini_v0.1"):
        if cls._model is None:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import torch

            cls._model = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
            cls._model = cls._model.to("cuda" if torch.cuda.is_available() else "cpu")
            cls._tok = AutoTokenizer.from_pretrained(model_id)
            cls._sr = cls._model.config.sampling_rate

    @classmethod
    def synth(cls, text: str, description: Optional[str] = None) -> bytes:
        import torch, numpy as np, soundfile as sf, io

        cls.ensure_loaded()
        desc = description or (
            "A British male speaker with a clear, neutral accent, natural pacing, "
            "calm tone, very clear audio, minimal room reverb."
        )
        input_ids = cls._tok(desc, return_tensors="pt").input_ids.to(cls._model.device)
        prompt_ids = cls._tok(text, return_tensors="pt").input_ids.to(cls._model.device)
        with torch.no_grad():
            audio = cls._model.generate(
                input_ids=input_ids, prompt_input_ids=prompt_ids
            )
        audio_np = audio.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_np, cls._sr, format="WAV")
        return buf.getvalue()


# --------- routing ---------
def pick_engine(
    global_engine: Optional[str], language: Optional[str], row_engine: Optional[str]
) -> str:
    # explicit per-row engine wins
    if row_engine in ("space", "parler"):
        return row_engine
    # global override
    if global_engine in ("space", "parler"):
        return global_engine
    # heuristic by language
    if language and language.lower().startswith(("en", "en-")):
        return "parler"
    return "space"


# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Single text to synthesize")
    ap.add_argument("--in", dest="in_path", type=Path, help="CSV/JSON with rows")
    ap.add_argument("--outdir", type=Path, required=True, help="Directory for .wav output")
    # Space (UA)
    ap.add_argument("--endpoint", default="patriotyk/styletts2-ukrainian", help="HF Space id")
    ap.add_argument("--api", choices=["verbalize", "synthesize"], help="Force Space API (default auto)")
    ap.add_argument("--model-name", help="Space model_name (default multi)")
    ap.add_argument("--voice-name", help="Space voice_name (default Артем Окороков)")
    ap.add_argument("--speed", type=float, help="Space speed (default 1.0)")
    # Global engine override
    ap.add_argument("--engine", choices=["space", "parler"], help="Force engine for all rows")
    args = ap.parse_args()

    if not args.text and not args.in_path:
        ap.error("Provide --text or --in")

    args.outdir.mkdir(parents=True, exist_ok=True)

    def run_one(
        text: str,
        out_file: Path,
        language: Optional[str] = None,
        row_engine: Optional[str] = None,
        voice_name: Optional[str] = None,
        model_name: Optional[str] = None,
        speed: Optional[float] = None,
    ):
        eng = pick_engine(args.engine, language, row_engine)
        if eng == "parler":
            wav_bytes = ParlerBritish.synth(text)
            write_bytes(out_file, wav_bytes)
        else:
            # Space: auto-pick API
            api = args.api or (
                "synthesize"
                if (voice_name or model_name or speed or args.voice_name or args.model_name or args.speed)
                else "verbalize"
            )
            res = space_call(
                endpoint=args.endpoint,
                api=api,
                text=text,
                model_name=(model_name or args.model_name or "multi"),
                voice_name=(voice_name or args.voice_name or "Артем Окороков"),
                speed=(speed if speed is not None else (args.speed if args.speed is not None else 1.0)),
            )
            space_save_result(res, out_file)
        print(out_file)

    # single
    if args.text:
        out = args.outdir / f"{slugify(args.text[:50])}.wav"
        run_one(args.text, out)
        return

    # batch
    rows = read_items(args.in_path)
    if not rows:
        print("No valid rows in input.", file=sys.stderr)
        sys.exit(1)

    seen = set()
    for r in rows:
        if not r.get("id"):
            raise ValueError("Each row must have a non-empty 'id' to name the output file.")
        fname = slugify(r["id"])
        if fname in seen:
            raise ValueError(f"Duplicate id after slugify: {fname}. Make ids unique.")
        seen.add(fname)

        out = args.outdir / f"{fname}.wav"
        run_one(
            text=r["text"],
            out_file=out,
            language=r.get("language"),
            row_engine=r.get("engine"),
            voice_name=r.get("voice_name"),
            model_name=r.get("model_name"),
            speed=r.get("speed"),
        )



if __name__ == "__main__":
    main()
