import math

import torch
from skimage.metrics import structural_similarity


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.detach().float() - b.detach().float()) ** 2).item()
    return 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = a.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    b_np = b.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    values = [structural_similarity(x, y, channel_axis=-1, data_range=1.0) for x, y in zip(a_np, b_np)]
    return float(sum(values) / max(1, len(values)))


def apd(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().mean().item() * 255.0)
