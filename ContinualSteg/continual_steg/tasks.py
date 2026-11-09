from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Task:
    task_id: int
    name: str
    severity: float = 0.0


TASKS = {
    "clean": Task(0, "clean", 0.0),
    "jpeg": Task(1, "jpeg", 50.0),
    "gaussian": Task(2, "gaussian", 0.05),
    "resize": Task(3, "resize", 0.75),
    "crop": Task(4, "crop", 0.75),
}


def apply_attack(x: torch.Tensor, task: Task) -> torch.Tensor:
    """Apply a differentiable approximation of a container attack."""
    if task.name == "clean":
        return x.clamp(0, 1)
    if task.name == "gaussian":
        return (x + torch.randn_like(x) * task.severity).clamp(0, 1)
    if task.name == "jpeg":
        step = max(1.0, task.severity) / 255.0
        quantized = torch.round(x / step) * step
        return x + (quantized - x).detach()
    if task.name == "resize":
        h, w = x.shape[-2:]
        small = F.interpolate(x, scale_factor=task.severity, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    if task.name == "crop":
        h, w = x.shape[-2:]
        crop_h = max(1, int(h * task.severity))
        crop_w = max(1, int(w * task.severity))
        top, left = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = x[..., top:top + crop_h, left:left + crop_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    raise ValueError(f"unknown task {task.name}")
