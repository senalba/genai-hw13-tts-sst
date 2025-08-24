#!/usr/bin/env python3
"""
TTS: читання тексту з --text або з CSV/JSON (--in) і збереження .wav у --outdir.
Підтримка кількох голосів/мов (якщо це вміє обрана модель, напр. XTTS v2).

Приклади:
  # Простий варіант (одне речення)
  python tts_script.py --text "Hello world" --outdir out \
      --model tts_models/en/ljspeech/tacotron2-DDC

  # Деталізований (CSV з колонками: id?, text, speaker?, language?)
  python tts_script.py --in data.csv --outdir out \
      --model tts_models/multilingual/multi-dataset/xtts_v2 --language en

  # Мультимова/голоси (XTTS v2) із референтним голосом
  python tts_script.py --text "Привіт, світ" --outdir out \
      --model tts_models/multilingual/multi-dataset/xtts_v2 --language uk \
      --speaker-wav path/to/voice_sample.wav

Залежності:
  pip install TTS==0.22.0 soundfile
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

def slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s]+", "_", s.strip())
    return s[:maxlen] or "utt"

def read_items(path: Path):
    items = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("text"):
                    continue
                items.append({
                    "id": row.get("id") or "",
                    "text": row["text"],
                    "speaker": row.get("speaker"),
                    "language": row.get("language"),
                })
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise ValueError("JSON має бути списком об'єктів або мати ключ 'items'.")
        for obj in data:
            if not obj.get("text"):
                continue
            items.append({
                "id": obj.get("id") or "",
                "text": obj["text"],
                "speaker": obj.get("speaker"),
                "language": obj.get("language"),
            })
    else:
        raise ValueError("Підтримуються лише .csv або .json")
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Текст для простого варіанту")
    ap.add_argument("--in", dest="in_path", type=Path, help="CSV/JSON із переліком текстів")
    ap.add_argument("--outdir", type=Path, required=True, help="Директорія для .wav")
    ap.add_argument("--model", default="tts_models/en/ljspeech/tacotron2-DDC",
                    help="Назва моделі Coqui TTS (напр., xtts_v2 для мультимови)")
    ap.add_argument("--speaker", help="Ім'я спікера (для багатоголосих моделей)")
    ap.add_argument("--speaker-wav", dest="speaker_wav", help="Шлях до зразка голосу (XTTS v2)")
    ap.add_argument("--language", help="Код мови (en, uk, ...), якщо підтримується моделлю")
    args = ap.parse_args()

    if not args.text and not args.in_path:
        ap.error("Вкажіть --text або --in")

    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        from TTS.api import TTS as CoquiTTS
    except Exception as e:
        print("Помилка імпорту Coqui TTS. Встановіть пакет: pip install TTS==0.22.0", file=sys.stderr)
        raise

    tts = CoquiTTS(model_name=args.model)

    def synth_one(text: str, file_path: Path, speaker=None, language=None):
        kw = {"text": text, "file_path": str(file_path)}
        # Параметри залежать від моделі: передаємо те, що підтримується.
        if args.speaker_wav:
            kw["speaker_wav"] = args.speaker_wav
        if speaker:
            kw["speaker"] = speaker
        if language:
            kw["language"] = language
        try:
            tts.tts_to_file(**kw)
        except TypeError:
            # Якщо модель не підтримує speaker/language/speaker_wav — пробуємо базовий виклик
            kw = {"text": text, "file_path": str(file_path)}
            tts.tts_to_file(**kw)

    if args.text:
        base = slugify(args.text[:50])
        out = args.outdir / f"{base}.wav"
        synth_one(args.text, out, speaker=args.speaker, language=args.language)
        print(out)
        return

    # Пакетна обробка з CSV/JSON
    items = read_items(args.in_path)
    if not items:
        print("У вхідному файлі немає валідних записів.", file=sys.stderr)
        sys.exit(1)

    for i, it in enumerate(items, 1):
        name = it["id"] or slugify(it["text"][:50])
        out = args.outdir / f"{i:03d}_{name}.wav"
        synth_one(it["text"], out,
                  speaker=it.get("speaker") or args.speaker,
                  language=it.get("language") or args.language)
        print(out)

if __name__ == "__main__":
    main()
