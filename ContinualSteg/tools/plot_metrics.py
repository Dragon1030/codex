"""Plot stage-wise metrics from a ContinualSteg metrics.csv file."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out_dir", default="runs/plots")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.metrics).open(newline="")))
    if not rows:
        raise ValueError("metrics.csv is empty")
    for row in rows:
        row["stage"] = int(row["stage"])
        for key in ("cover_psnr", "cover_ssim", "secret_psnr", "secret_ssim", "cover_apd"):
            row[key] = float(row[key])

    series = defaultdict(list)
    for row in rows:
        series[row["task"]].append(row)
    for values in series.values():
        values.sort(key=lambda row: row["stage"])

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "secret_psnr": ("Secret recovery PSNR (dB)", "secret_psnr.png"),
        "secret_ssim": ("Secret recovery SSIM", "secret_ssim.png"),
        "cover_psnr": ("Cover-container PSNR (dB)", "cover_psnr.png"),
        "cover_ssim": ("Cover-container SSIM", "cover_ssim.png"),
        "cover_apd": ("Cover-container APD", "cover_apd.png"),
    }
    for metric, (ylabel, filename) in plots.items():
        figure, axis = plt.subplots(figsize=(8, 5))
        for task, values in sorted(series.items()):
            axis.plot(
                [row["stage"] for row in values],
                [row[metric] for row in values],
                marker="o",
                label=task,
            )
        axis.set_xlabel("Continual-learning stage")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel + " by task")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    print(f"saved plots to {output_dir}")


if __name__ == "__main__":
    main()
