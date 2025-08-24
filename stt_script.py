#!/usr/bin/env python3
"""
STT: розпізнає один файл (--file) або всю папку (--dir) і зберігає результат у CSV/JSON.
Додатково: оцінка WER за jiwer при наявності ground truth (--gt).

Приклади:
  # Простий варіант (один .wav, друк у stdout)
  python stt_script.py --file audio.wav

  # Деталізований (папка з .wav) -> CSV
  python stt_script.py --dir wavs --out transcribed.csv

  # Додатковий: WER (gt.csv має колонки: filename, text)
  python stt_script.py --dir wavs --out transcribed.csv --gt gt.csv

Залежності:
  pip install faster-whisper==1.0.3 jiwer==3.0.4 soundfile==0.12.1
"""
import argparse, csv, json, sys
from pathlib import Path
from typing import List, Dict, Tuple
from jiwer import (
    process_words,
    Compose,
    ToLowerCase,
    RemovePunctuation,
    RemoveMultipleSpaces,
    Strip,
)


TEXT_NORM = Compose(
    [ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()]
)


def list_wavs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*.wav") if p.is_file()])


def transcribe_one(
    model, path: Path, language: str = None, beam_size: int = 5, vad: bool = False
) -> Tuple[str, Dict]:
    segments, info = model.transcribe(
        str(path), beam_size=beam_size, language=language, vad_filter=vad
    )
    text = "".join(seg.text for seg in segments).strip()
    meta = {
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }
    return text, meta


def save_table(rows: List[Dict], out_path: Path):
    if out_path.suffix.lower() == ".json":
        out_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    # CSV
    keys = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_gt(gt_path: Path) -> Dict[str, str]:
    if gt_path.suffix.lower() == ".json":
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        # очікуємо список об'єктів {filename, text}
        return {
            row["filename"]: row["text"]
            for row in data
            if row.get("filename") and row.get("text")
        }
    # CSV
    out = {}
    with gt_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("filename") and row.get("text"):
                out[row["filename"]] = row["text"]
    return out


def compute_wer(
    gt_map: Dict[str, str], hyp_map: Dict[str, str]
) -> Tuple[List[Dict], float]:
    file_rows = []
    total_sub = total_del = total_ins = total_ref = 0
    for fn, gt in gt_map.items():
        hyp = hyp_map.get(fn, "")
        m = process_words(
            reference=gt,
            hypothesis=hyp,
            reference_transform=TEXT_NORM,
            hypothesis_transform=TEXT_NORM,
        )
        file_rows.append(
            {
                "filename": fn,
                "wer": m["wer"],
                "hits": m["hits"],
                "substitutions": m["substitutions"],
                "deletions": m["deletions"],
                "insertions": m["insertions"],
                "ref_len": m["reference_length"],
            }
        )
        total_sub += m["substitutions"]
        total_del += m["deletions"]
        total_ins += m["insertions"]
        total_ref += m["reference_length"]
    overall = (total_sub + total_del + total_ins) / max(total_ref, 1)
    return file_rows, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="Окремий .wav для простого варіанту")
    ap.add_argument("--dir", type=Path, help="Папка з .wav для пакетного варіанту")
    ap.add_argument(
        "--beam-size", type=int, default=5, help="Beam size для декодування (дефолт 5)"
    )
    ap.add_argument(
        "--vad", action="store_true", help="Увімкнути VAD-фільтр для шумних записів"
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Файл результатів (.csv або .json). Для --file не обов'язковий",
    )
    ap.add_argument(
        "--model-size",
        default="small",
        help="Розмір/назва моделі faster-whisper (tiny, base, small, medium, large-v3, або шлях)",
    )
    ap.add_argument("--language", help="Код мови (автовизначення, якщо не задано)")
    ap.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Пристрій для моделі"
    )
    ap.add_argument(
        "--gt",
        type=Path,
        help="Ground truth (CSV/JSON з колонками filename,text) для WER",
    )
    args = ap.parse_args()

    if not args.file and not args.dir:
        ap.error("Вкажіть --file або --dir")

    try:
        from faster_whisper import WhisperModel
    except Exception:
        print(
            "Встановіть пакети: pip install faster-whisper jiwer soundfile",
            file=sys.stderr,
        )
        raise

    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model_size, device=args.device, compute_type=compute_type)

    rows = []
    if args.file:
        text, meta = transcribe_one(
            model,
            args.file,
            language=args.language,
            beam_size=args.beam_size,
            vad=args.vad,
        )
        row = {
            "filename": args.file.name,
            "text": text,
            **{k: v for k, v in meta.items() if v is not None},
        }
        if args.out:
            save_table([row], args.out)
        else:
            print(text)
        rows = [row]
    else:
        wavs = list_wavs(args.dir)
        if not wavs:
            print("У папці немає .wav", file=sys.stderr)
            sys.exit(1)
        for i, p in enumerate(wavs, 1):
            print(f"[{i}/{len(wavs)}] Transcribing {p.name} ...")
            text, meta = transcribe_one(
                model, p, language=args.language, beam_size=args.beam_size, vad=args.vad
            )
            rows.append(
                {
                    "filename": p.name,
                    "text": text,
                    **{k: v for k, v in meta.items() if v is not None},
                }
            )
        if not args.out:
            args.out = Path("transcribed.csv")
        save_table(rows, args.out)
        print(args.out)

    # Оцінка WER (за бажанням)
    if args.gt:
        gt_map = load_gt(args.gt)
        hyp_map = {r["filename"]: r["text"] for r in rows}
        file_wer_rows, overall = compute_wer(gt_map, hyp_map)
        # Записуємо окремий файл з WER
        wer_out = Path(args.out.stem + "_wer.csv") if args.out else Path("wer.csv")
        save_table(
            file_wer_rows + [{"filename": "__OVERALL__", "wer": overall}], wer_out
        )
        print(f"OVERALL WER: {overall:.4f}")
        print(wer_out)


if __name__ == "__main__":
    main()



# python stt_script.py --file british_demo.wav --out stt_british_demo.json  --model-size small

# python stt_script.py  --dir art-of-war   --out art-of-war/stt_transcribed.csv --model-size small --beam-size 5 --vad

# python stt_script.py --dir 130732 --out 130732/transcribed.csv --gt 130732/gt.csv --model-size small --beam-size 5 --vad


