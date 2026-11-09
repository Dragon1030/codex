from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_files(root: str) -> List[Path]:
    files = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return sorted(files)


class PairedImageDataset(Dataset):
    """Samples a cover and a secret independently from one image folder."""

    def __init__(self, root: str, image_size: int, length: int | None = None):
        self.files = image_files(root)
        self.length = length or len(self.files)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        cover_path = self.files[index % len(self.files)]
        secret_path = self.files[(index * 997 + 17) % len(self.files)]
        cover = self.transform(Image.open(cover_path).convert("RGB"))
        secret = self.transform(Image.open(secret_path).convert("RGB"))
        return cover, secret
