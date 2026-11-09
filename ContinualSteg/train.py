from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from continual_steg.data import PairedImageDataset
from continual_steg.metrics import apd, psnr, ssim
from continual_steg.models import ContinualHidingModel
from continual_steg.replay import ReplayBuffer
from continual_steg.tasks import TASKS, apply_attack


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual robust image hiding")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_dir", default="")
    parser.add_argument("--val_dir", default="")
    parser.add_argument("--checkpoint_dir", default="runs/continual")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--tasks", default="clean,jpeg,gaussian,resize,crop")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs_per_task", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--adapter_bottleneck", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=256)
    parser.add_argument("--replay_per_batch", type=int, default=2)
    parser.add_argument("--lambda_distill", type=float, default=0.2)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_tasks(text: str):
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; available={sorted(TASKS)}")
    return [TASKS[name] for name in names]


def make_loader(root, image_size, batch_size, workers, shuffle):
    dataset = PairedImageDataset(root, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=shuffle)


def train_one_task(model, old_model, task, loader, replay, args, device, optimizer,
                   start_epoch=0, checkpoint_callback=None):
    model.train()
    for epoch in range(start_epoch, args.epochs_per_task):
        cover_sum = secret_sum = distill_sum = 0.0
        batches = 0
        for batch_index, (cover, secret) in enumerate(tqdm(loader, desc=f"task={task.name} epoch={epoch + 1}")):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            cover, secret = cover.to(device), secret.to(device)
            container, _ = model.hide_reveal(cover, secret, task.task_id)
            attacked = apply_attack(container, task)
            recovered = model.reveal(attacked, task.task_id)

            cover_loss = F.l1_loss(container, cover)
            secret_loss = F.l1_loss(recovered, secret)
            distill_loss = torch.zeros((), device=device)

            memory = replay.sample(args.replay_per_batch)
            if old_model is not None and memory is not None:
                old_cover, old_secret, old_task = memory
                old_cover, old_secret = old_cover.to(device), old_secret.to(device)
                distill_terms = []
                for old_task_id in old_task.unique().tolist():
                    mask = old_task == old_task_id
                    with torch.no_grad():
                        old_container, old_recovered = old_model.hide_reveal(
                            old_cover[mask], old_secret[mask], int(old_task_id)
                        )
                    new_container, new_recovered = model.hide_reveal(
                        old_cover[mask], old_secret[mask], int(old_task_id)
                    )
                    distill_terms.append(
                        F.mse_loss(new_container, old_container)
                        + F.mse_loss(new_recovered, old_recovered)
                    )
                distill_loss = torch.stack(distill_terms).mean()

            loss = cover_loss + secret_loss + args.lambda_distill * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            cover_sum += cover_loss.item()
            secret_sum += secret_loss.item()
            distill_sum += distill_loss.item()
            batches += 1
        divisor = max(1, batches)
        print(f"task={task.name} epoch={epoch + 1}/{args.epochs_per_task} "
              f"cover={cover_sum / divisor:.6f} secret={secret_sum / divisor:.6f} "
              f"distill={distill_sum / divisor:.6f}")
        if checkpoint_callback is not None:
            checkpoint_callback(epoch + 1)


@torch.no_grad()
def evaluate(model, task, loader, device, max_batches=0):
    model.eval()
    values = {"cover_psnr": [], "cover_ssim": [], "secret_psnr": [], "secret_ssim": [], "cover_apd": []}
    for batch_index, (cover, secret) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        cover, secret = cover.to(device), secret.to(device)
        container, _ = model.hide_reveal(cover, secret, task.task_id)
        recovered = model.reveal(apply_attack(container, task), task.task_id)
        values["cover_psnr"].append(psnr(container, cover))
        values["cover_ssim"].append(ssim(container, cover))
        values["secret_psnr"].append(psnr(recovered, secret))
        values["secret_ssim"].append(ssim(recovered, secret))
        values["cover_apd"].append(apd(container, cover))
    return {key: sum(value) / max(1, len(value)) for key, value in values.items()}


def save_stage(path, model, optimizer, replay, task_id, epoch):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "replay": replay.state_dict(), "task_id": task_id, "epoch": epoch}, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tasks = get_tasks(args.tasks)
    model = ContinualHidingModel(max(task.task_id for task in tasks) + 1,
                                 args.adapter_bottleneck, args.prompt_dim).to(device)
    replay = ReplayBuffer(args.replay_capacity)
    start_task = 0
    start_epoch = 0
    resume_state = None
    checkpoint_path = args.checkpoint or args.resume
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        resume_state = state
        model.load_state_dict(state["model"])
        replay.load_state_dict(state["replay"])
        if int(state.get("epoch", args.epochs_per_task)) < args.epochs_per_task:
            start_task = int(state["task_id"])
            start_epoch = int(state.get("epoch", 0))
        else:
            start_task = int(state["task_id"]) + 1

    train_root = args.train_dir or os.path.join(args.data_root, "train")
    val_root = args.val_dir or os.path.join(args.data_root, "val")
    train_loader = make_loader(train_root, args.image_size, args.batch_size, args.workers, True)
    val_loader = make_loader(val_root, args.image_size, args.batch_size, args.workers, False)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        for task in tasks:
            print(task.name, evaluate(model, task, val_loader, device, args.max_val_batches))
        return

    old_model = None
    for task in tasks[start_task:]:
        if task.task_id == 0 and start_task == 0:
            for parameter in model.parameters():
                parameter.requires_grad = True
            for name, parameter in model.named_parameters():
                if "adapters." in name or "prompt_bias." in name:
                    parameter.requires_grad = ".0." in name
                elif "prompt_bank.prompts" in name:
                    parameter.requires_grad = name.endswith(".0")
        else:
            old_model = model.clone_frozen()
            model.train_only_task(task.task_id)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate, weight_decay=1e-4)
        if resume_state is not None and task.task_id == start_task and start_epoch > 0:
            optimizer.load_state_dict(resume_state["optimizer"])
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        def checkpoint_callback(epoch):
            save_stage(latest_path, model, optimizer, replay, task.task_id, epoch)
        train_one_task(model, old_model, task, train_loader, replay, args, device, optimizer,
                       start_epoch=start_epoch if task.task_id == start_task else 0,
                       checkpoint_callback=checkpoint_callback)
        replay_cover, replay_secret = next(iter(train_loader))
        with torch.no_grad():
            prototype_cover = replay_cover.to(device)
            prototype_secret = replay_secret.to(device)
            prototype_container, _ = model.hide_reveal(prototype_cover, prototype_secret, task.task_id)
            model.update_prototype(prototype_container, task.task_id)
        replay.add(replay_cover, replay_secret, task.task_id)

        rows = []
        for seen_task in tasks[:task.task_id + 1]:
            metrics = evaluate(model, seen_task, val_loader, device, args.max_val_batches)
            rows.append({"stage": task.task_id, "task": seen_task.name, **metrics})
            print(f"stage={task.name} eval={seen_task.name} {metrics}")
        save_stage(os.path.join(args.checkpoint_dir, f"stage_{task.task_id:02d}.pt"),
                   model, optimizer, replay, task.task_id, args.epochs_per_task)
        start_epoch = 0
        with open(os.path.join(args.checkpoint_dir, "metrics.csv"), "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
