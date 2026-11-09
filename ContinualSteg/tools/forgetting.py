"""Summarize a metrics.csv produced by train.py."""

import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    history = defaultdict(list)
    with open(args.csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            history[task].append((int(row["stage"]), float(row["secret_psnr"])))
    for task, values in sorted(history.items()):
        values.sort()
        best = max(score for _, score in values)
        final = values[-1][1]
        print(f"{task}: best_secret_psnr={best:.4f} final={final:.4f} forgetting={best-final:.4f}")


if __name__ == "__main__":
    main()
