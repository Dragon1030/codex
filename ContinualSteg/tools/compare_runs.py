"""Compare final-stage metrics and forgetting across training runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_run(text: str):
    if "=" not in text:
        raise argparse.ArgumentTypeError("runs must use NAME=PATH")
    name, path = text.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("runs must use NAME=PATH")
    return name, Path(path)


def read_run(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    for row in rows:
        row["stage"] = int(row["stage"])
        for key in ("cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"):
            row[key] = float(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="run name and metrics.csv path, e.g. full=runs/full/metrics.csv",
    )
    parser.add_argument("--out", default="runs/experiment_comparison.csv")
    args = parser.parse_args()

    output_rows = []
    for run_name, path in args.run:
        rows = read_run(path)
        final_stage = max(row["stage"] for row in rows)
        history = defaultdict(list)
        for row in rows:
            history[row["task"]].append(row)
        for task, task_rows in sorted(history.items()):
            task_rows.sort(key=lambda row: row["stage"])
            final = next(row for row in task_rows if row["stage"] == final_stage)
            best = max(row["secret_psnr"] for row in task_rows)
            forgetting = best - final["secret_psnr"] if len(task_rows) > 1 else ""
            output_rows.append({
                "run": run_name,
                "final_stage": final_stage,
                "task": task,
                "cover_psnr": final["cover_psnr"],
                "cover_ssim": final["cover_ssim"],
                "secret_psnr": final["secret_psnr"],
                "secret_ssim": final["secret_ssim"],
                "cover_apd": final["cover_apd"],
                "forgetting_secret_psnr": forgetting,
            })

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        fields = list(output_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"saved comparison to {output_path}")
    for row in output_rows:
        forgetting = row["forgetting_secret_psnr"]
        forgetting_text = "n/a" if forgetting == "" else f"{float(forgetting):.4f}"
        print(
            f"{row['run']:>18} {row['task']:<8} "
            f"secret_psnr={row['secret_psnr']:.4f} "
            f"secret_ssim={row['secret_ssim']:.4f} "
            f"forgetting={forgetting_text}"
        )


if __name__ == "__main__":
    main()
