"""Create qualitative hiding/recovery panels from a ContinualSteg checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from continual_steg.data import PairedImageDataset
from continual_steg.metrics import apd, psnr, ssim
from continual_steg.models import ContinualHidingModel
from continual_steg.tasks import TASKS, apply_attack


def parse_args():
    parser = argparse.ArgumentParser(description="Save qualitative ContinualSteg results")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_dir", default="runs/visuals")
    parser.add_argument("--tasks", default="clean,jpeg,gaussian,resize,crop")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--diff_scale", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_tasks(text: str):
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; available={sorted(TASKS)}")
    return [TASKS[name] for name in names]


def infer_model_shape(state):
    model_state = state["model"]
    prompt_keys = [
        key for key in model_state
        if key.startswith("prompt_bank.prompts.") and key.rsplit(".", 1)[-1].isdigit()
    ]
    if not prompt_keys:
        raise KeyError("checkpoint does not contain prompt_bank.prompts")
    num_tasks = max(int(key.rsplit(".", 1)[-1]) for key in prompt_keys) + 1
    prompt_dim = int(model_state["prompt_bank.prompts.0"].shape[0])
    adapter_bottleneck = int(model_state["hiding.adapters.0.down.weight"].shape[0])
    return num_tasks, adapter_bottleneck, prompt_dim


def amplified_diff(left: torch.Tensor, right: torch.Tensor, scale: float) -> torch.Tensor:
    return ((left - right).abs() * scale).clamp(0.0, 1.0)


@torch.no_grad()
def main():
    args = parse_args()
    task_list = get_tasks(args.tasks)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    state = torch.load(args.checkpoint, map_location=device)
    num_tasks, adapter_bottleneck, prompt_dim = infer_model_shape(state)
    model = ContinualHidingModel(num_tasks, adapter_bottleneck, prompt_dim).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dataset = PairedImageDataset(args.val_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_index, (cover, secret) in enumerate(loader):
        if sample_index >= args.max_samples:
            break
        cover, secret = cover.to(device), secret.to(device)
        for task in task_list:
            if task.task_id >= num_tasks:
                raise ValueError(
                    f"checkpoint has {num_tasks} tasks but task {task.name} has id {task.task_id}"
                )
            container, _ = model.hide_reveal(cover, secret, task.task_id)
            attacked = apply_attack(container, task)
            recovered = model.reveal_container(attacked, task.task_id)
            cover_diff = amplified_diff(cover, container, args.diff_scale)
            secret_diff = amplified_diff(secret, recovered, args.diff_scale)

            # Columns: cover, container, amplified cover diff, secret,
            # recovered secret, amplified secret diff.
            panel = torch.cat(
                [cover, container, cover_diff, secret, recovered, secret_diff], dim=0
            )
            image_path = output_dir / f"sample_{sample_index:04d}_{task.name}.png"
            save_image(make_grid(panel, nrow=6, padding=4), image_path)

            rows.append({
                "sample": sample_index,
                "task": task.name,
                "cover_psnr": psnr(container, cover),
                "cover_ssim": ssim(container, cover),
                "secret_psnr": psnr(recovered, secret),
                "secret_ssim": ssim(recovered, secret),
                "cover_apd": apd(container, cover),
                "image": str(image_path),
            })

    metrics_path = output_dir / "visualization_metrics.csv"
    if rows:
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"saved {len(rows)} panels to {output_dir}")
    if rows:
        print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
