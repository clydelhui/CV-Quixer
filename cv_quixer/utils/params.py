"""Parameter counting utilities for hybrid quantum-classical models."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in any nn.Module.

    Args:
        model:          Any nn.Module.
        trainable_only: If True (default), count only parameters with
                        requires_grad=True. Set False to count all parameters.

    Returns:
        Integer parameter count.
    """
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad or not trainable_only
    )


def parameter_summary(model: nn.Module) -> dict:
    """Return a structured breakdown of parameter counts.

    Counts are split at the top-level named children of the model, which gives
    the right granularity for a hybrid model with named blocks (e.g. pre_mlp,
    cv_block, post_mlp).

    Args:
        model: Any nn.Module.

    Returns:
        Dictionary with keys:
            "total"     — all parameters (trainable + frozen)
            "trainable" — parameters with requires_grad=True
            "frozen"    — parameters with requires_grad=False
            "by_module" — dict mapping each direct child's name to
                          {"total": int, "trainable": int, "frozen": int}

        Parameters that belong directly to the model (not to any child) are
        collected under the key "_own" inside "by_module".
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    by_module: dict[str, dict[str, int]] = {}

    # Own parameters (registered directly on the model, not in any child)
    own_params = list(model._parameters.values())  # noqa: SLF001
    own_total = sum(p.numel() for p in own_params if p is not None)
    own_trainable = sum(
        p.numel() for p in own_params if p is not None and p.requires_grad
    )
    if own_total > 0:
        by_module["_own"] = {
            "total": own_total,
            "trainable": own_trainable,
            "frozen": own_total - own_trainable,
        }

    # Direct children
    for name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        by_module[name] = {
            "total": child_total,
            "trainable": child_trainable,
            "frozen": child_total - child_trainable,
        }

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "by_module": by_module,
    }


def print_parameter_table(model: nn.Module, title: str = "") -> None:
    """Pretty-print a parameter count table to stdout.

    Example output:
        ┌─────────────────────────────────────────────┐
        │  HybridCV — parameter summary               │
        ├──────────────┬───────────┬──────────────────┤
        │  Module      │  Params   │  Trainable       │
        ├──────────────┼───────────┼──────────────────┤
        │  pre_mlp     │    4,416  │          4,416   │
        │  cv_block    │       12  │             12   │
        │  post_mlp    │    4,164  │          4,164   │
        ├──────────────┼───────────┼──────────────────┤
        │  TOTAL       │    8,592  │          8,592   │
        └──────────────┴───────────┴──────────────────┘

    Args:
        model: Any nn.Module.
        title: Optional header title (defaults to the model class name).
    """
    summary = parameter_summary(model)
    header = title or type(model).__name__

    col_w = [16, 12, 12]   # widths for module name, total, trainable columns

    def row(name: str, total: int, trainable: int) -> str:
        return (
            f"  {name:<{col_w[0]}} "
            f"{total:>{col_w[1]},}  "
            f"{trainable:>{col_w[1]},}"
        )

    sep = "─" * (col_w[0] + col_w[1] * 2 + 8)

    lines = [
        f"\n{header} — parameter summary",
        sep,
        f"  {'Module':<{col_w[0]}} {'Total':>{col_w[1]}}  {'Trainable':>{col_w[1]}}",
        sep,
    ]

    for name, counts in summary["by_module"].items():
        lines.append(row(name, counts["total"], counts["trainable"]))

    lines += [
        sep,
        row("TOTAL", summary["total"], summary["trainable"]),
        sep,
    ]

    print("\n".join(lines))
