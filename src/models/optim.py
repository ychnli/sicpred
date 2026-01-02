"""
Optimizer + learning-rate scheduler helpers.

Conventions (all optional unless noted):

Optimizer:
- OPTIMIZER: str, default "adamw"
- OPTIMIZER_ARGS: dict, merged with required args
- LEARNING_RATE: float (required by current configs)
- WEIGHT_DECAY: float

Schedulers (optional):
- LR_SCHEDULER: str or None. Supported:
    - None / "constant" / "none"
    - "cosine_with_warmup" (linear warmup then cosine decay; step-based)
    - "cosine" (CosineAnnealingLR; epoch-based)
    - "warm_restarts" (CosineAnnealingWarmRestarts; epoch-based)
- LR_SCHEDULER_ARGS: dict

Note: epoch-based schedulers are stepped once per epoch, after validation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim import AdamW


def build_optimizer(config, model):
    """Build an optimizer from config.

    Defaults to AdamW to match the existing training script.
    """
    name = getattr(config, "OPTIMIZER", "adamw")
    name = str(name).lower()
    args = dict(getattr(config, "OPTIMIZER_ARGS", {}) or {})

    if name == "adamw":
        return AdamW(
            model.parameters(),
            lr=getattr(config, "LEARNING_RATE"),
            weight_decay=getattr(config, "WEIGHT_DECAY", 0.0),
            **args,
        )

    raise NotImplementedError(f"Optimizer {name!r} not implemented")


def _cosine_with_warmup_lambda(*, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = min(warmup_steps, total_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        t = min(step - warmup_steps, cosine_steps)
        progress = float(t) / float(cosine_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return lr_lambda


def build_lr_scheduler(
    config,
    optimizer,
    *,
    steps_per_epoch: int,
    total_epochs: int,
    global_step: int = 0,
):
    """Build an (optional) LR scheduler.

    Returns:
        (scheduler, interval)
        - scheduler: torch scheduler or None
        - interval: "step" or "epoch" (how often to call scheduler.step())

    `global_step` is used to set `last_epoch` for step-based schedulers.
    """
    name = getattr(config, "LR_SCHEDULER", None)
    if name is None:
        return None, None

    name = str(name).lower()
    if name in {"constant", "none"}:
        return None, None

    args = dict(getattr(config, "LR_SCHEDULER_ARGS", {}) or {})

    if name in {"cosine_with_warmup", "cosinewarmup", "warmup_cosine"}:
        total_steps = int(steps_per_epoch) * int(total_epochs)
        warmup_steps = int(args.get("warmup_steps", 0))
        if "warmup_epochs" in args:
            warmup_steps = int(args["warmup_epochs"]) * int(steps_per_epoch)
        min_lr_ratio = float(args.get("min_lr_ratio", 0.0))

        lr_lambda = _cosine_with_warmup_lambda(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
            last_epoch=int(global_step) - 1,
        )
        return scheduler, "step"

    if name in {"cosine", "cosine_annealing"}:
        # Epoch-based cosine annealing. Users can specify T_max (epochs).
        # If omitted, default to total_epochs.
        t_max = int(args.get("t_max", total_epochs))
        eta_min = float(args.get("eta_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        return scheduler, "epoch"

    if name in {"warm_restarts", "cosine_warm_restarts", "cosinewarmrestarts"}:
        # Epoch-based cosine warm restarts.
        t0 = int(args.get("t_0", 10))
        t_mult = int(args.get("t_mult", 1))
        eta_min = float(args.get("eta_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=eta_min,
        )
        return scheduler, "epoch"

    raise NotImplementedError(
        f"LR_SCHEDULER={name!r} not implemented. Supported: constant/none, cosine_with_warmup, cosine, warm_restarts."
    )
