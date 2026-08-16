"""OmniFall 标注 → 统一片段级事件标注 CSV。

输入: data/omnifall/labels/*.csv (path,label,start,end,subject,cam,dataset,...)
输出: data/annotations/events.csv
  clip_id, dataset, label_id, label_name, t_start, t_end, is_fall_event, is_hard_negative

事件语义约定（对应 eval/metrics.py 的 GT Event）:
- is_fall_event=1: label==fall(1)，即"跌倒过程"时间段——评测的 TP 对象
- is_hard_negative=1: 易混日常动作 fall_en(2)/sit_down(3)/sitting(4)/lie_down(5)/lying(6)
  /kneel_down(10)/kneeling(11)/squat_down(12)/squatting(13)——误报分析的重点
- 其余（walk/stand_up/standing/other/crawl/jump）：普通背景类

用法: python tools/prepare_omnifall_events.py [--labels-dir data/omnifall/labels] [--out data/annotations/events.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABEL_NAMES = {
    0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
    5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other",
    10: "kneel_down", 11: "kneeling", 12: "squat_down", 13: "squatting",
    14: "crawl", 15: "jump",
}
FALL_LABEL = 1
HARD_NEGATIVES = {2, 3, 4, 5, 6, 10, 11, 12, 13}


def convert(labels_dir: Path, out_path: Path) -> dict:
    rows_out = []
    stats = {"files": 0, "segments": 0, "fall_events": 0, "hard_negatives": 0,
             "by_dataset": {}}
    for csv_path in sorted(labels_dir.glob("*.csv")):
        if csv_path.stem == "label2id":
            continue
        stats["files"] += 1
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = int(row["label"])
                ds = row["dataset"]
                rec = {
                    "clip_id": row["path"],
                    "dataset": ds,
                    "label_id": label,
                    "label_name": LABEL_NAMES.get(label, f"unk{label}"),
                    "t_start": float(row["start"]),
                    "t_end": float(row["end"]),
                    "is_fall_event": int(label == FALL_LABEL),
                    "is_hard_negative": int(label in HARD_NEGATIVES),
                }
                rows_out.append(rec)
                stats["segments"] += 1
                stats["fall_events"] += rec["is_fall_event"]
                stats["hard_negatives"] += rec["is_hard_negative"]
                d = stats["by_dataset"].setdefault(ds, {"segments": 0, "falls": 0})
                d["segments"] += 1
                d["falls"] += rec["is_fall_event"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="data/omnifall/labels")
    ap.add_argument("--out", default="data/annotations/events.csv")
    args = ap.parse_args()
    stats = convert(Path(args.labels_dir), Path(args.out))
    print(f"标注文件: {stats['files']} | 总片段: {stats['segments']} | "
          f"跌倒事件: {stats['fall_events']} | 易混负例: {stats['hard_negatives']}")
    for ds, d in sorted(stats["by_dataset"].items()):
        print(f"  {ds:12s} segments={d['segments']:6d} falls={d['falls']:5d}")
    print(f"输出: {args.out}")
