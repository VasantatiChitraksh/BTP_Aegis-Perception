"""Single-node torchrun support; plain python retains single-device behavior."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta


@contextmanager
def distributed_session(requested_device: str):
    import torch
    import torch.distributed as dist

    from .checkpoints import select_device

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = select_device(requested_device)
    if world_size > 1 and device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    owned = world_size > 1 and not dist.is_initialized()
    if owned:
        dist.init_process_group(
            "nccl" if device.startswith("cuda") else "gloo", timeout=timedelta(minutes=30)
        )
    try:
        yield device, rank, world_size
    finally:
        if owned:
            dist.destroy_process_group()


def gather_objects(value):
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, value)
    return results
