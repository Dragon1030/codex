from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ReplayItem:
    cover: torch.Tensor
    secret: torch.Tensor
    task_id: int


class ReplayBuffer:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.items: List[ReplayItem] = []
        self.seen = 0

    def add(self, cover: torch.Tensor, secret: torch.Tensor, task_id: int):
        for c, s in zip(cover.detach().cpu(), secret.detach().cpu()):
            self.seen += 1
            item = ReplayItem(c.clone(), s.clone(), int(task_id))
            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                slot = torch.randint(self.seen, (1,)).item()
                if slot < self.capacity:
                    self.items[slot] = item

    def sample(self, count: int):
        if not self.items:
            return None
        count = min(count, len(self.items))
        idx = torch.randperm(len(self.items))[:count].tolist()
        items = [self.items[i] for i in idx]
        return (
            torch.stack([x.cover for x in items]),
            torch.stack([x.secret for x in items]),
            torch.tensor([x.task_id for x in items], dtype=torch.long),
        )

    def state_dict(self):
        return {"capacity": self.capacity, "seen": self.seen, "items": self.items}

    def load_state_dict(self, state):
        self.capacity = int(state["capacity"])
        self.seen = int(state["seen"])
        self.items = list(state["items"])
