#!/usr/bin/env python3
# make_gt_from_librispeech.py
# -----------------------------------------------------------------------------
# CLI-скрипт: перетворює LibriSpeech *.trans.txt у GT-файл (CSV/JSON) формату:
#   filename,text
# і кладе його ПОРУЧ (у той самий каталог, де лежать *.flac та *.trans.txt).
#
# Приклади:
#   # 1) Один каталог (де є *.trans.txt і *.flac)
#   python make_gt_from_librispeech.py "/path/LibriSpeech/test-other/367/130732"
#
#   # 2) Рекурсивно по всій гілці (створить gt.csv у кожній папці з *.trans.txt)
#   python make_gt_from_librispeech.py "/path/LibriSpeech/test-other" --recursive
#
#   # 3) Вихід JSON та з розширенням .wav замість .flac
#   python make_gt_from_librispeech.py "/path/LibriSpeech/test-other/367/130732" --ext wav --out-format json
#
# Залежності: стандартна бібліотека Python.
# -----------------------------------------------------------------------------

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict


TRANS_GLOB = "*.trans.txt"


def parse_trans_file(trans_path: Path) -> List[Tuple[str, str]]:
    """
    Зчитує LibriSpeech *.trans.txt
    Рядок формату: "<ID> <TEXT...>"
    Повертає список (id, text)
    """
    rows: List[Tuple[str, str]] = []
    for line in trans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s+(.*)$", line)
        if not m:
            continue
        utt_id, txt = m.group(1), m.group(2).strip()
        rows.append((utt_id, txt))
    return rows


def write_gt(
    rows: List[Tuple[str, str]], out_path: Path, ext: str, out_format: str
) -> None:
    """
    Записує GT у out_path (gt.csv|gt.json) з колонками:
      filename,text
    filename формуємо як "<utt_id>.<ext>"
    """
    mapped: List[Dict[str, str]] = [
        {"filename": f"{utt_id}.{ext}", "text": text} for utt_id, text in rows
    ]

    if out_format == "json":
        out_path.write_text(
            json.dumps(mapped, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    # CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "text"])
        w.writeheader()
        w.writerows(mapped)


def make_gt_in_dir(dir_path: Path, ext: str, out_format: str) -> bool:
    """
    Шукає *.trans.txt у каталозі dir_path.
    Якщо знайшов — створює gt.csv (або gt.json) поруч.
    Повертає True, якщо файл створено.
    """
    found = False
    for trans_path in dir_path.glob(TRANS_GLOB):
        rows = parse_trans_file(trans_path)
        if not rows:
            continue
        out_name = "gt.json" if out_format == "json" else "gt.csv"
        out_path = dir_path / out_name
        write_gt(rows, out_path, ext=ext, out_format=out_format)
        print(f"[OK] {out_path}  ({len(rows)} рядків)  ← з {trans_path.name}")
        found = True
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Створює GT (CSV/JSON) з LibriSpeech *.trans.txt у кожній цільовій папці."
    )
    ap.add_argument(
        "path", type=Path, help="Каталог із *.trans.txt (або корінь для --recursive)"
    )
    ap.add_argument(
        "--recursive", action="store_true", help="Обійти підкаталоги рекурсивно"
    )
    ap.add_argument(
        "--ext",
        choices=["flac", "wav"],
        default="flac",
        help="Яке розширення ставити у полі filename (дефолт flac)",
    )
    ap.add_argument(
        "--out-format",
        choices=["csv", "json"],
        default="csv",
        help="Формат GT-файлу (дефолт csv)",
    )
    args = ap.parse_args()

    root: Path = args.path
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Немає каталогу: {root}")

    if args.recursive:
        made_any = False
        for d in sorted(p for p in root.rglob("*") if p.is_dir()):
            if make_gt_in_dir(d, ext=args.ext, out_format=args.out_format):
                made_any = True
        if not made_any:
            print("[INFO] Не знайдено жодного *.trans.txt у підкаталогах.")
    else:
        if not make_gt_in_dir(root, ext=args.ext, out_format=args.out_format):
            print("[INFO] У каталозі немає *.trans.txt")


if __name__ == "__main__":
    main()
