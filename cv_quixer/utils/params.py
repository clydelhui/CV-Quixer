"""Parameter counting utilities for hybrid quantum-classical models."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable, TypeVar

import torch.nn as nn

_ConfigT = TypeVar("_ConfigT")


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


def autoscale_to_target(
    build_fn: Callable[[_ConfigT], nn.Module],
    config: _ConfigT,
    knob: str,
    target: int,
    *,
    step: int = 1,
    trigger_field: str = "target_params",
    hi_cap: int = 2 ** 20,
    tol: float = 0.10,
) -> _ConfigT:
    """Binary-search an integer config field so the built model hits a param budget.

    Model-agnostic: the only thing it knows about the model is how to build one
    from a config (``build_fn``) and how to count its trainable parameters
    (``count_parameters``). The parameter count is therefore read off the *real*
    ``nn.Parameter`` set of a throwaway instance — there is no hand-written
    formula to drift from the architecture, so any model with a monotonic
    scaling knob is auto-scalable for free.

    The search varies ``config.<knob> = step * k`` over positive integers ``k``
    and assumes the trainable-param count is non-decreasing in the knob (true for
    widths, head counts, mode counts, …).

    Args:
        build_fn:      ``(config) -> nn.Module``. Must build *without* re-entering
                       auto-scaling; the search guarantees this by forcing
                       ``trigger_field = -1`` in every trial config.
        config:        A (frozen or mutable) dataclass config instance.
        knob:          Name of the integer dataclass field to scale.
        target:        Desired trainable-parameter count.
        step:          Knob granularity. Trials use ``step * k`` so callers can
                       enforce constraints (e.g. ``step=2`` to keep a derived
                       dimension even).
        trigger_field: Field set to ``-1`` in trial configs to prevent recursive
                       auto-scaling (defaults to the ``target_params`` convention).
        hi_cap:        Upper bound on the trial knob value. If the count never
                       reaches ``target`` below this cap the knob cannot satisfy
                       the budget (e.g. it does not affect the param count) and a
                       ``ValueError`` is raised instead of looping forever.
        tol:           Relative tolerance; a warning (not error) is emitted if the
                       closest achievable count is further than this from target.

    Returns:
        A copy of ``config`` (via ``dataclasses.replace``) with ``knob`` set to
        the best-fit value. ``trigger_field`` is left unchanged on the result.
    """

    def count(value: int) -> int:
        trial = dataclasses.replace(config, **{trigger_field: -1, knob: value})
        return count_parameters(build_fn(trial))

    # Expand an upper bound until the budget is reachable (or proven out of reach).
    hi = 1
    while count(step * hi) < target:
        hi *= 2
        if step * hi > hi_cap:
            raise ValueError(
                f"scaling_knob={knob!r} cannot reach target_params={target} "
                f"(step*{hi} exceeds hi_cap={hi_cap}); the knob may not affect the "
                "parameter count or is bounded — choose a different scaling_knob."
            )

    # Smallest k with count(step*k) >= target.
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if count(step * mid) < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo

    # The step below the crossover may be the closer fit (under- vs over-shoot).
    if best > 1 and abs(count(step * (best - 1)) - target) < abs(count(step * best) - target):
        best -= 1

    best_value = step * best
    achieved = count(best_value)
    if abs(achieved - target) / max(target, 1) > tol:
        warnings.warn(
            f"target_params={target} but closest achievable is {achieved} "
            f"({knob}={best_value}). Adjust other architecture knobs for a tighter match.",
            stacklevel=2,
        )

    return dataclasses.replace(config, **{knob: best_value})
