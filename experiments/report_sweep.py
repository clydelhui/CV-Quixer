"""Aggregate a CV-Quixer hyperparameter sweep into thesis-ready outputs.

Scans a sweep directory (`results/sweeps/<sweep>_<ts>/`) produced by
`experiments/sweep.py`, reads each run's `history.json` (and `config.json` for
the resolved architecture), and emits a comparison table + plots into the sweep
directory:

    summary.csv / summary.md         one row per run (drop-in thesis table)
    figures/acc_vs_params.png        best test acc vs achieved params, one
                                     seed-averaged point per configuration
                                     (series chosen via --series-by)
    figures/acc_by_observable.png    grouped bars: acc by observable — by budget,
                                     or by configuration for manual sweeps
                                     (only when ≥2 observable presets)
    figures/acc_by_scaling_knob.png  grouped bars: acc by model/scaling knob,
                                     grouped by budget (only when ≥2 knobs swept)
    figures/acc_vs_trunc_lambda.png  acc vs truncation penalty weight λ
                                     (only when ≥2 λ values are swept)
    figures/acc_vs_<field>.png       one per architecture field that varies;
                                     points connected into all-else-equal trend
                                     lines (one line per combination of the
                                     other varying fields)

Cross-run figures group runs by *configuration identity* — every recorded sweep
coordinate except the training seed (``CONFIG_IDENTITY_FIELDS``) — so only seed
repeats are ever averaged together. Manual-mode runs, which all share the
``target_params=-1`` / default-knob placeholders, therefore get one point per
architecture instead of collapsing into a single averaged point. Every figure
renders only when its comparison axis takes ≥2 distinct values (a skip notice is
printed otherwise).

Epoch fairness: best/final accuracies span each run's whole training history,
so comparing runs trained to different lengths (a mid-top-up sweep, a
``--runs``-filtered top-up — see experiments/resume_sweep.py) is unfair; a
``RuntimeWarning`` fires whenever the compared runs' epoch counts differ. Pass
``--max-epoch N`` to derive best/best-epoch/final/n_epochs from each run's
first N epochs only (ignoring meta's all-epochs values) and compare at a common
horizon; runs with fewer than N epochs are kept, with a warning. The cap
applies to the table and cross-run figures only — the per-run
report_diagnostics suite always renders full history.

The cross-run aggregation reads only JSON (no torch / model rebuild), so it is
fast and runs on a partial or in-progress sweep. By default it then also renders
the full `report_diagnostics.py` figure suite into each run's own `figures/` dir
(one subprocess per run; report_diagnostics' default path is npz/JSON-based, so
heavy torch imports stay deferred). Pass --skip-per-run-figures for the fast
cross-run-only pass.

A *coordinate filter* (CONTEXT.md) restricts the tables + cross-run figures to a
subset of runs by configuration value — ``--num-modes 2 3 --num-heads 5 10`` keeps
runs with ``num_modes ∈ {2,3}`` *and* ``num_heads ∈ {5,10}`` (OR within a flag,
AND across; same flags + semantics as experiments/resume_sweep.py, shared via
experiments/_run_selection.py). A run missing a filtered coordinate is excluded
with a warning. When a filter is active and ``--out-dir`` is omitted, output
defaults to ``<sweep>/subsets/<filter-slug>/`` so the full-sweep artefacts at the
sweep root are left intact.

Usage:
    uv run python experiments/report_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/

    # only the num_modes∈{2,3} runs → results/sweeps/<sweep>_<ts>/subsets/num_modes-2-3/
    uv run python experiments/report_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/ \
        --num-modes 2 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _run_selection import (
    FILTERABLE_FIELDS,
    REROLL_PREFIX,
    add_filter_args,
    coords_from_meta,
    parse_filter_args,
    read_run_names_file,
    run_matches,
)

# Repo root (experiments/ -> repo root); used to invoke report_diagnostics.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIAGNOSTICS = "experiments/report_diagnostics.py"

# Above this many all-else-equal trend lines in one acc_vs_<field> figure, the
# legend is suppressed (it would dwarf the plot) — the lines + colours stay.
MAX_LEGEND_CHAINS = 12

# Resolved architecture knobs recorded in history["meta"] by full_experiment.py.
# Surfaced per-run in the table and used as the candidate axes for the
# acc-vs-hyperparam figures (any that vary across the runs). num_layers is loaded
# separately (it predates this list) but is included here as a figure candidate.
ARCH_META_FIELDS = [
    "num_modes", "num_heads", "cutoff_dim", "poly_degree",
    "cnn_channels_1", "cnn_channels_2", "cnn_kernel_size", "decoder_hidden_dim",
    "cnn_num_conv_layers", "hypernet_num_linear_layers", "decoder_num_layers",
    "decoder_hidden_mult", "cvqnn_num_layers",
    # Stacked-model axes (ADR-0003); absent on older runs → skipped.
    "num_seq2seq_blocks", "pooling", "block_residual",
    # Loss-weight knobs (query_trunc_lambda is a sweep axis, __qtl marker).
    "query_trunc_lambda", "cvqnn_trunc_lambda",
    # Symmetry-breaking poly-init perturbation (__pin marker); the axis a re-roll
    # varies vs its original, so it must be a config-identity coordinate (off and
    # on are different configurations, never seed-averaged together).
    "poly_init_noise",
    # Positional-encoding variant (__pe marker); none/1d/2d are distinct
    # configurations, never seed-averaged together. Defaulted to "2d" on
    # pre-knob runs just below.
    "positional_encoding",
    # Coefficient-ablation variant (__ca marker; ADR-0008); none/lcu/lcu_poly are
    # distinct configurations, never seed-averaged together. Defaulted to "none"
    # on pre-knob runs just below.
    "coeff_ablation",
]

# A run's *configuration identity*: every sweep coordinate except the training
# seed. Two runs are "the same experiment repeated" — and may be seed-averaged
# in cross-run figures — iff they agree on all of these fields. Keep this in
# sync with the axes sweep.py exposes; _check_identity_drift() warns when run
# names reveal an axis missing from this list.
CONFIG_IDENTITY_FIELDS = (
    "model", "observables", "scaling_knob", "target_params",
    "num_layers", "trunc_lambda", *ARCH_META_FIELDS,
)
_FIELD_INDEX = {f: i for i, f in enumerate(CONFIG_IDENTITY_FIELDS)}

# Columns for the summary table, in display order.
SUMMARY_COLUMNS = [
    "run_name", "model", "observables", "target_params", "achieved_params",
    "scaling_knob", "num_layers", "trunc_lambda", "seed",
    *ARCH_META_FIELDS,
    "best_test_acc", "best_epoch", "final_test_acc",
    "final_train_acc", "n_epochs", "total_runtime_sec", "device",
]


def _resolve_model(run_dir: Path, meta: dict) -> str:
    """Model variant for a run: meta → config.json fallback → "quantum".

    `meta["model"]` only exists on runs trained after that key was added; older
    runs (e.g. the finished quantum sweeps) carry it only in config.json, so fall
    back to the saved config and finally to the historic default "quantum".
    """
    model = meta.get("model")
    if model:
        return str(model)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                cfg_model = json.load(f).get("model")
            if cfg_model:
                return str(cfg_model)
        except (json.JSONDecodeError, OSError):
            pass
    return "quantum"


def _load_run(run_dir: Path, max_epoch: int | None = None) -> dict | None:
    """Read one run's history/config into a flat summary row, or None to skip.

    With ``max_epoch=N`` the per-epoch series are truncated to the first N
    epochs and best/best-epoch/final/n_epochs are derived from the slice —
    ignoring meta's all-epochs values — so runs topped up to different lengths
    can be compared fairly at a common epoch count.
    """
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        # A dir with config.json but no history.json is a run that started but
        # never finished an epoch (e.g. OOM / wall-time kill). Warn so it isn't
        # silently dropped from the table and figures.
        if (run_dir / "config.json").is_file():
            warnings.warn(
                f"skipping {run_dir.name}: no history.json — run did not complete "
                "an epoch (check its logs/slurm .err; re-run to populate).",
                RuntimeWarning, stacklevel=2,
            )
        return None
    with open(history_path) as f:
        history = json.load(f)

    meta = history.get("meta", {})
    epoch = history.get("epoch", {})
    test_acc = epoch.get("test_acc") or []
    train_acc = epoch.get("train_acc") or []

    if max_epoch is not None:
        if len(test_acc) < max_epoch:
            warnings.warn(
                f"{run_dir.name} has only {len(test_acc)} epoch(s), below the "
                f"--max-epoch {max_epoch} cap — it is kept, but compares a "
                "shorter training horizon (top it up with "
                "experiments/resume_sweep.py for a fair comparison).",
                RuntimeWarning, stacklevel=2,
            )
        test_acc = test_acc[:max_epoch]
        train_acc = train_acc[:max_epoch]
        best = max(test_acc) if test_acc else None
        best_epoch = test_acc.index(best) + 1 if test_acc else None
    else:
        best = meta.get("best_test_acc")
        if best is None and test_acc:
            best = max(test_acc)
        best_epoch = meta.get("best_epoch")

    row = {
        "run_name": run_dir.name,
        "model": _resolve_model(run_dir, meta),
        "observables": meta.get("observables_name"),
        "target_params": meta.get("target_params"),
        "achieved_params": meta.get("achieved_params") or meta.get("n_params"),
        # Default to the historic knob for runs predating the scaling_knob axis,
        # so old sweeps still group cleanly instead of being dropped as None.
        "scaling_knob": meta.get("scaling_knob") or "cnn_channels_2",
        "num_layers": meta.get("num_layers"),
        "trunc_lambda": meta.get("trunc_lambda"),
        "seed": meta.get("seed"),
        "best_test_acc": best,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc[-1] if test_acc else None,
        "final_train_acc": train_acc[-1] if train_acc else None,
        "n_epochs": len(test_acc),
        "total_runtime_sec": meta.get("total_runtime_sec"),
        "device": meta.get("device"),
    }
    # Resolved architecture knobs (present for runs from the manual-sweep-aware
    # full_experiment.py; None for older runs — kept as None in the identity key).
    # poly_init_noise is defaulted separately just below, so skip it here.
    for field in ARCH_META_FIELDS:
        if field in ("poly_init_noise", "positional_encoding", "coeff_ablation"):
            continue
        row[field] = meta.get(field)
    # poly_init_noise: a pre-feature run has no key — default to 0.0 (off), like
    # scaling_knob's pre-axis default, so a re-roll (on) pairs against an original
    # (off) instead of (None) in the compare figure.
    _pin = meta.get("poly_init_noise")
    row["poly_init_noise"] = 0.0 if _pin is None else float(_pin)
    # positional_encoding: a pre-knob run has no key — default to "2d" (the
    # historic hardcoded behaviour), so it groups as the 2d arm rather than None.
    _pe = meta.get("positional_encoding")
    row["positional_encoding"] = "2d" if _pe is None else str(_pe)
    # coeff_ablation: a pre-knob run has no key — default to "none" (the historic
    # fully-trained behaviour), so it groups as the none arm rather than None.
    _ca = meta.get("coeff_ablation")
    row["coeff_ablation"] = "none" if _ca is None else str(_ca)
    # Re-roll provenance (CONTEXT.md "Re-roll"): the original this run re-rolls,
    # or None for an ordinary run. Used by --rerolls to pair re-rolls to originals.
    row["reroll_of"] = meta.get("reroll_of")
    return row


# The ``low_accuracy_runs.txt`` parser is shared (the same format is an *exclude*
# list here and an *include* list for rerun_sweep) — single source in
# _run_selection so a format tweak can never make the two tools disagree.
read_exclude_file = read_run_names_file


def is_reroll(row: dict) -> bool:
    """True if this run is a re-roll (identified by its run-name prefix)."""
    return str(row.get("run_name", "")).startswith(REROLL_PREFIX)


def select_for_rerolls(rows: list[dict], mode: str) -> list[dict]:
    """Filter rows per the --rerolls mode (CONTEXT.md "Re-roll", ADR-0007).

    Pairing is by ``reroll_of`` reference (set from history meta), never by name
    stripping — a re-roll may change any knob, not just the one in its dir name.

      * ``ignore``  — drop every re-roll row (the full-sweep report is unchanged).
      * ``replace`` — substitute each re-roll for its original (drop the originals
        that have a re-roll; keep everything else).
      * ``compare`` — keep ONLY the paired configs: the re-rolls and the originals
        they reference. The escape-rate comparison is about exactly those.
    """
    if mode == "ignore":
        return [r for r in rows if not is_reroll(r)]
    rerolls = [r for r in rows if is_reroll(r)]
    paired_originals = {r.get("reroll_of") for r in rerolls}
    if mode == "replace":
        return [r for r in rows
                if is_reroll(r) or r["run_name"] not in paired_originals]
    if mode == "compare":
        return [r for r in rows
                if is_reroll(r) or r["run_name"] in paired_originals]
    raise ValueError(f"unknown --rerolls mode: {mode}")


def _common_max_epoch(rows: list[dict]) -> int | None:
    """The max epoch count common to all rows (min n_epochs); None if empty.

    The fair horizon for a re-roll compare: an original and its re-roll may have
    trained to different lengths, so best/final are compared at the shorter one.
    """
    counts = [r["n_epochs"] for r in rows if r.get("n_epochs")]
    return min(counts) if counts else None


def load_sweep(sweep_dir: Path, max_epoch: int | None = None) -> list[dict]:
    """Load every run row under `sweep_dir`, sorted for stable output.

    The ``subsets/`` directory (where coordinate-filtered runs write their own
    tables + figures) is skipped — it is this tool's own output, never a run.
    """
    rows: list[dict] = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if run_dir.name == "subsets":
            continue
        row = _load_run(run_dir, max_epoch=max_epoch)
        if row is not None:
            rows.append(row)
    rows.sort(
        key=lambda r: (
            str(r["observables"]),
            r["target_params"] if r["target_params"] is not None else -1,
            r["seed"] if r["seed"] is not None else -1,
        )
    )
    return rows


def write_table(rows: list[dict], sweep_dir: Path) -> None:
    """Write summary.csv and summary.md."""
    csv_path = sweep_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in SUMMARY_COLUMNS} for r in rows)
    print(f"  ✓ {csv_path}")

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    md_path = sweep_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(SUMMARY_COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in SUMMARY_COLUMNS) + " |\n")
    print(f"  ✓ {md_path}")


def _aggregate_by(
    rows: list[dict], fields: tuple[str, ...], skip_none_keys: bool = True
) -> dict[tuple, tuple[float, float]]:
    """Mean ± std of best_test_acc over runs, keyed by the given meta fields.

    Runs missing best_test_acc are always skipped. With ``skip_none_keys=True``
    (default), a None in any key field also drops the run — that is what
    restricts the λ figure to λ-swept runs. ``skip_none_keys=False`` keeps None
    as an ordinary key value instead (configuration-identity grouping: older
    runs simply miss some meta fields and must still group among themselves).
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        acc = r.get("best_test_acc")
        if acc is None:
            continue
        key = tuple(r.get(f) for f in fields)
        if skip_none_keys and any(k is None for k in key):
            continue
        buckets[key].append(float(acc))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in buckets.items()}


def _mean_value_by(
    rows: list[dict], fields: tuple[str, ...], value_field: str,
    skip_none_keys: bool = True,
) -> dict[tuple, float]:
    """Mean of ``value_field`` over rows, keyed by the given meta fields.

    Uses the same keying/skip rules as ``_aggregate_by`` (rows missing
    best_test_acc or — with ``skip_none_keys`` — any key field drop out), so the
    returned map lines up one-to-one with ``_aggregate_by``'s keys. Used to place
    points at the mean *achieved* parameter count of each seed-group rather than
    the target budget.
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("best_test_acc") is None:
            continue
        key = tuple(r.get(f) for f in fields)
        if skip_none_keys and any(k is None for k in key):
            continue
        val = r.get(value_field)
        if val is None:
            continue
        buckets[key].append(float(val))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _config_key(row: dict) -> tuple:
    """A run's configuration identity key (every sweep coordinate except seed)."""
    return tuple(row.get(f) for f in CONFIG_IDENTITY_FIELDS)


def _strip_seed(run_name: str) -> str:
    """Run name with the ``__seed{N}`` marker removed.

    sweep.py encodes every active grid axis in the run-dir name (it must, or two
    grid points would collide on disk), so the stripped name is unique per
    configuration by construction.
    """
    return re.sub(r"__seed\d+", "", run_name)


def _check_identity_drift(rows: list[dict]) -> None:
    """Warn when one configuration identity spans >1 seed-stripped run name.

    Two different stripped run names sharing one identity key means a sweep axis
    exists that ``CONFIG_IDENTITY_FIELDS`` does not list — those genuinely
    different configs would be silently seed-averaged together (the manual-sweep
    collapse bug all over again), so make it loud.
    """
    names_by_key: dict[tuple, set[str]] = defaultdict(set)
    for r in rows:
        names_by_key[_config_key(r)].add(_strip_seed(str(r.get("run_name"))))
    for names in names_by_key.values():
        if len(names) > 1:
            warnings.warn(
                "runs with different sweep coordinates share one configuration "
                f"identity and will be averaged together: {sorted(names)} — a "
                "sweep axis is probably missing from CONFIG_IDENTITY_FIELDS in "
                "report_sweep.py.",
                RuntimeWarning, stacklevel=2,
            )


def _check_epoch_heterogeneity(rows: list[dict]) -> None:
    """Warn when the compared runs' effective epoch counts differ.

    best/final acc over more epochs is advantaged, so cross-run comparisons of
    unequal-length runs are unfair — top the short runs up
    (experiments/resume_sweep.py) or cap the long ones (--max-epoch).
    """
    # Keyed per row, not per run_name: two compared sweeps may legitimately
    # contain the same run name (e.g. a re-run vs the original).
    counts = [(str(r["run_name"]), r["n_epochs"]) for r in rows]
    if len({n for _name, n in counts}) > 1:
        by_count: dict[int, list[str]] = defaultdict(list)
        for name, n in counts:
            by_count[n].append(name)
        detail = "; ".join(
            f"{n} epoch(s): {sorted(names)}" for n, names in sorted(by_count.items())
        )
        warnings.warn(
            "runs being compared have differing epoch count(s) — best/final "
            "accuracies span unequal training horizons. Top up the short runs "
            "(experiments/resume_sweep.py) or pass --max-epoch to compare at a "
            f"common epoch. {detail}",
            RuntimeWarning, stacklevel=2,
        )


def _config_groups(rows: list[dict]) -> dict[tuple, dict]:
    """Seed-average ``best_test_acc`` within each configuration identity.

    Returns ``{config_key: {"acc": (mean, std), "x": mean achieved_params}}``
    (``x`` is None when no run in the group recorded achieved_params). None
    identity-field values are kept as ordinary key components.
    """
    acc = _aggregate_by(rows, CONFIG_IDENTITY_FIELDS, skip_none_keys=False)
    xs = _mean_value_by(
        rows, CONFIG_IDENTITY_FIELDS, "achieved_params", skip_none_keys=False
    )
    return {key: {"acc": acc[key], "x": xs.get(key)} for key in acc}


def _varying_fields(keys: Iterable[tuple]) -> list[str]:
    """Identity fields taking ≥2 distinct values across the given config keys."""
    keys = list(keys)
    return [
        f for i, f in enumerate(CONFIG_IDENTITY_FIELDS)
        if len({k[i] for k in keys}) > 1
    ]


def _dependent_fields(rows: list[dict]) -> set[str]:
    """Identity fields that are *derived* from other coordinates in this sweep.

    A dependent field is a deterministic function of the independent sweep axes,
    so it co-varies with them and must NOT count toward "all else equal" in the
    acc_vs_<field> trend lines — otherwise it splits every chain into lone points
    (the reason num_modes / num_heads never connected: decoder_hidden_dim follows
    them, so no two runs differ in num_modes *alone*). Currently the only derived
    architecture field is ``decoder_hidden_dim``, which is sized as
    ``decoder_hidden_mult × (num_heads × readout_width)`` whenever
    ``decoder_hidden_mult`` is set — readout_width depending on num_modes and the
    observable preset, so the decoder width is slaved to those axes.
    """
    dependent: set[str] = set()
    if any(r.get("decoder_hidden_mult") is not None for r in rows):
        dependent.add("decoder_hidden_dim")
    return dependent


def _series_key(cfg_key: tuple, series_by: Sequence[str]) -> tuple:
    """Project a config key onto the --series-by fields (legend grouping)."""
    return tuple(cfg_key[_FIELD_INDEX[f]] for f in series_by)


def _series_label(skey: tuple) -> str:
    return "/".join(str(v) for v in skey) or "all"


def plot_acc_vs_params(
    rows: list[dict], fig_dir: Path, series_by: Sequence[str]
) -> None:
    """Best test acc vs *achieved* parameter count, one point per configuration.

    Each point is one configuration identity, seed-averaged, at its mean
    achieved parameter count — manual-mode configs (which all share the
    ``target_params=-1`` placeholder) get one point each instead of collapsing
    into a single average. Series (colour/legend) group by the --series-by
    fields. A series' points are connected only when exactly one identity field
    varies across them (a genuine 1-D trend, e.g. a budget sweep); otherwise
    they are drawn as markers + error bars only.
    """
    groups = {k: g for k, g in _config_groups(rows).items() if g["x"] is not None}
    if len(groups) < 2:
        print("  (need ≥2 distinct configurations — skipping acc_vs_params)")
        return
    by_series: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for key, g in groups.items():
        by_series[_series_key(key, series_by)][key] = g
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for skey in sorted(by_series, key=str):
        sgroups = by_series[skey]
        pts = sorted((g["x"], g["acc"][0], g["acc"][1]) for g in sgroups.values())
        linestyle = "-" if len(_varying_fields(sgroups)) == 1 else "none"
        ax.errorbar(
            [p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
            marker="o", linestyle=linestyle, capsize=3, label=_series_label(skey),
        )
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs achieved parameter count")
    ax.grid(alpha=0.3)
    ax.legend(title=" / ".join(series_by))
    fig.tight_layout()
    out = fig_dir / "acc_vs_params.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_observable(rows: list[dict], fig_dir: Path) -> None:
    """Grouped bars: best test acc by observable preset.

    Skipped unless ≥2 observable presets have completed runs. Budget sweeps keep
    the historic layout — (model, observable, knob) categories with one bar
    group per target budget. Manual sweeps (no positive budgets) group by
    configuration instead: one bar per config under its (model, observable)
    category, annotated with its achieved param count.
    """
    completed = [r for r in rows if r.get("best_test_acc") is not None]
    n_obs = len({
        r.get("observables") for r in completed if r.get("observables") is not None
    })
    if n_obs < 2:
        print("  (need ≥2 observable presets — skipping acc_by_observable)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if any((r.get("target_params") or 0) > 0 for r in completed):
        # Budget mode: historic layout, with `model` added to the category key.
        key_fields = ("model", "observables", "scaling_knob", "target_params")
        agg = _aggregate_by(rows, key_fields)
        achieved = _mean_value_by(rows, key_fields, "achieved_params")
        cats = sorted({(m, o, k) for (m, o, k, _tp) in agg})
        budgets = sorted({tp for (_m, _o, _k, tp) in agg})
        x = np.arange(len(cats))
        width = 0.8 / max(len(budgets), 1)
        for i, tp in enumerate(budgets):
            means = [agg.get((m, o, k, tp), (np.nan, 0.0))[0] for (m, o, k) in cats]
            errs = [agg.get((m, o, k, tp), (np.nan, 0.0))[1] for (m, o, k) in cats]
            bars = ax.bar(
                x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}"
            )
            for (m, o, k), bar in zip(cats, bars):
                n = achieved.get((m, o, k, tp))
                if n is not None and not np.isnan(bar.get_height()):
                    ax.annotate(
                        f"{int(round(n)):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks(x + width * (len(budgets) - 1) / 2)
        ax.set_xticklabels(
            [f"{m}/{o}/{k}" for (m, o, k) in cats], rotation=15, ha="right"
        )
        ax.set_xlabel("Model / observable preset / scaling knob")
        ax.legend(title="target param budget")
    else:
        # Manual mode: one bar per configuration under its (model, observable).
        groups = _config_groups(rows)
        mi, oi = _FIELD_INDEX["model"], _FIELD_INDEX["observables"]
        by_cat: dict[tuple, list[dict]] = defaultdict(list)
        for key, g in groups.items():
            by_cat[(key[mi], key[oi])].append(g)
        cats = sorted(by_cat, key=str)
        n_bars = max(len(v) for v in by_cat.values())
        width = 0.8 / n_bars
        x = np.arange(len(cats))
        for ci, cat in enumerate(cats):
            cfgs = sorted(by_cat[cat], key=lambda g: (g["x"] is None, g["x"]))
            for bi, g in enumerate(cfgs):
                bar = ax.bar(
                    ci + bi * width, g["acc"][0], width,
                    yerr=g["acc"][1], capsize=3, color="tab:blue",
                )[0]
                if g["x"] is not None:
                    ax.annotate(
                        f"{int(round(g['x'])):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks(x + width * (n_bars - 1) / 2)
        ax.set_xticklabels([f"{m}/{o}" for (m, o) in cats], rotation=15, ha="right")
        ax.set_xlabel("Model / observable preset (one bar per configuration)")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by observable (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = fig_dir / "acc_by_observable.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_scaling_knob(rows: list[dict], fig_dir: Path) -> None:
    """Grouped bars: best test acc by model/scaling knob, grouped by budget.

    The headline figure for a scaling-knob sweep — quantum-width (num_heads) vs
    classical (cnn_channels_2) at each budget. Skipped unless ≥2 knobs are
    present. Intended for single-observable sweeps; observables are averaged over.
    """
    key_fields = ("model", "scaling_knob", "target_params")
    agg = _aggregate_by(rows, key_fields)
    knobs = sorted({k for (_m, k, _tp) in agg})
    if len(knobs) < 2:
        print("  (need ≥2 scaling_knob values — skipping acc_by_scaling_knob)")
        return
    # Achieved param count per (model, knob, budget), averaged over observables.
    achieved = _mean_value_by(rows, key_fields, "achieved_params")
    cats = sorted({(m, k) for (m, k, _tp) in agg})
    budgets = sorted({tp for (_m, _k, tp) in agg})
    x = np.arange(len(cats))
    width = 0.8 / max(len(budgets), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tp in enumerate(budgets):
        means = [agg.get((m, k, tp), (np.nan, 0.0))[0] for (m, k) in cats]
        errs = [agg.get((m, k, tp), (np.nan, 0.0))[1] for (m, k) in cats]
        bars = ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}")
        for (m, k), bar in zip(cats, bars):
            n = achieved.get((m, k, tp))
            if n is not None and not np.isnan(bar.get_height()):
                ax.annotate(
                    f"{int(round(n)):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels([f"{m}/{k}" for (m, k) in cats])
    ax.set_xlabel("Model / scaling knob")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by scaling knob (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="target param budget")
    fig.tight_layout()
    out = fig_dir / "acc_by_scaling_knob.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def _plot_acc_vs_field(
    rows: list[dict], fig_dir: Path, field: str,
    xlabel: str, title: str, out_name: str,
) -> bool:
    """Best test acc vs one identity field ``field``, as all-else-equal trend lines.

    The plotted configurations (those with a non-None ``field``) are partitioned
    into *all-else-equal trend lines*: each line connects the configs that agree
    on every other identity coordinate and differ only in ``field``, so a
    multi-axis sweep yields one line per combination of the *other* varying
    fields (legend title = those field names, each line labelled by its values)
    rather than a single scatter. A chain with only one present x-value renders
    as a lone marker; ``--series-by`` does not affect this figure (the series are
    derived automatically). Above ``MAX_LEGEND_CHAINS`` lines the legend is
    suppressed (it would dwarf the plot) while the lines + colours remain.
    Returns False (and plots nothing) when ``field`` takes <2 distinct values
    across the runs.

    *Dependent fields* (``_dependent_fields`` — e.g. ``decoder_hidden_dim`` when
    sized by a multiplier) are excluded from the "all else equal" key: they are
    slaved to the independent axes, so counting them would split every chain into
    lone points. They are allowed to vary freely along a trend line (that is what
    lets num_modes / num_heads connect).
    """
    fi = _FIELD_INDEX[field]
    groups = {k: g for k, g in _config_groups(rows).items() if k[fi] is not None}
    if len({k[fi] for k in groups}) < 2:
        return False
    # Group into chains keyed by the *other* identity fields that vary across the
    # plotted configs, minus this sweep's dependent (derived) fields — within each
    # chain only `field` (and the slaved dependents) differs, so every chain is a
    # genuine 1-D trend in `field` and is always line-connected.
    dependent = _dependent_fields(rows) - {field}
    other_varying = [
        f for f in _varying_fields(groups) if f != field and f not in dependent
    ]
    ov_idx = [_FIELD_INDEX[f] for f in other_varying]
    by_chain: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for key, g in groups.items():
        by_chain[tuple(key[i] for i in ov_idx)][key] = g
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for chain_key in sorted(by_chain, key=str):
        chain = by_chain[chain_key]
        pts = sorted((k[fi], g["acc"][0], g["acc"][1]) for k, g in chain.items())
        xs = [p[0] for p in pts]
        if len(set(xs)) != len(xs):
            # Two distinct configs share a chain key and an x-value — only
            # possible if a dropped "dependent" field was in fact independent.
            warnings.warn(
                f"acc_vs_{field}: a trend line has duplicate {field} values after "
                f"excluding dependent fields {sorted(dependent)} — distinct "
                "configs overlap; the exclusion may be wrong for this sweep.",
                RuntimeWarning, stacklevel=2,
            )
        ax.errorbar(
            [p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
            marker="o", linestyle="-", capsize=3, label=_series_label(chain_key),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best test accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # Suppress the legend once there are too many trend lines to label legibly
    # (the lines + colours remain); otherwise title it with the varying fields.
    if len(by_chain) <= MAX_LEGEND_CHAINS:
        ax.legend(title=" / ".join(other_varying) or None)
    else:
        print(
            f"    ({len(by_chain)} trend lines > {MAX_LEGEND_CHAINS} — "
            f"legend suppressed for {out_name})"
        )
    fig.tight_layout()
    out = fig_dir / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")
    return True


def plot_acc_vs_trunc_lambda(rows: list[dict], fig_dir: Path) -> None:
    """Best test acc vs truncation penalty weight λ, as all-else-equal trend lines.

    Skipped unless ≥2 distinct λ values are present. Points are grouped into
    all-else-equal trend lines (see ``_plot_acc_vs_field``): each line connects
    the configs differing only in λ, legend-labelled by the other fields that
    vary across the sweep.
    """
    if not _plot_acc_vs_field(
        rows, fig_dir, "trunc_lambda",
        xlabel="Truncation penalty weight λ (trunc_lambda)",
        title="Accuracy vs truncation penalty weight",
        out_name="acc_vs_trunc_lambda.png",
    ):
        print("  (need ≥2 distinct trunc_lambda values — skipping acc_vs_trunc_lambda)")


def plot_acc_vs_hyperparam(rows: list[dict], fig_dir: Path) -> None:
    """One ``acc_vs_<field>.png`` per architecture field that varies across runs.

    The figure for manual-hyperparameter sweeps: for every candidate field
    (``ARCH_META_FIELDS`` + ``num_layers``) that takes ≥2 distinct non-None
    values, plot one seed-averaged point per configuration against that field's
    value, grouped into *all-else-equal trend lines* (see ``_plot_acc_vs_field``)
    — each line connects the configs that agree on every other identity
    coordinate and differ only in this field, so a multi-axis sweep gets one line
    per combination of the other varying fields. Fields constant across the
    sweep, or absent (older runs), are skipped.
    """
    candidates = ARCH_META_FIELDS + ["num_layers"]
    for field in candidates:
        _plot_acc_vs_field(
            rows, fig_dir, field,
            xlabel=field, title=f"Accuracy vs {field}",
            out_name=f"acc_vs_{field}.png",
        )


def render_per_run_figures(sweep_dir: Path, rows: list[dict]) -> None:
    """Render the full report_diagnostics figure suite for every run in the sweep.

    One `report_diagnostics.py --run-dir <run>` subprocess per run (its default
    path is npz/JSON-based, so heavy torch imports stay deferred). Each is wrapped
    so one failure doesn't abort the rest.
    """
    print(f"\nRendering per-run figure suites for {len(rows)} run(s)...")
    for r in rows:
        run_dir = sweep_dir / r["run_name"]
        if not run_dir.is_dir():
            warnings.warn(f"run dir missing, skipping: {run_dir}", RuntimeWarning)
            continue
        cmd = [sys.executable, "-u", REPORT_DIAGNOSTICS, "--run-dir", str(run_dir)]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"report_diagnostics failed for {r['run_name']}: exit {e.returncode}",
                RuntimeWarning,
            )


def _filter_slug(filters: dict[str, set]) -> str:
    """Filesystem-safe directory name encoding an active coordinate filter.

    Fields appear in registry order (deterministic regardless of flag order),
    each as ``{field}-{v1}-{v2}…`` with values sorted, joined by ``__`` — e.g.
    ``{num_modes: {3, 2}, num_heads: {5, 10}}`` → ``num_modes-2-3__num_heads-5-10``.
    Used to default a filtered run's output under ``<sweep>/subsets/<slug>/`` so it
    never clobbers the full-sweep artefacts.
    """
    parts = []
    for field in FILTERABLE_FIELDS:
        if field.name in filters:
            vals = "-".join(str(v) for v in sorted(filters[field.name]))
            parts.append(f"{field.name}-{vals}")
    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=str, required=True,
        help="sweep directory written by experiments/sweep.py",
    )
    parser.add_argument(
        "--series-by", nargs="+", default=["model", "observables"],
        choices=CONFIG_IDENTITY_FIELDS, metavar="FIELD",
        help="meta fields defining the legend/series of acc_vs_params "
             "(default: model observables — e.g. add scaling_knob to recover "
             "per-knob lines in a multi-knob budget sweep). The acc_vs_<field> "
             "and acc_vs_trunc_lambda figures derive their series automatically "
             "(all-else-equal trend lines) and ignore this. choices: "
             + ", ".join(CONFIG_IDENTITY_FIELDS),
    )
    parser.add_argument(
        "--max-epoch", type=int, default=None, metavar="N",
        help="derive best/final accuracy from each run's first N epochs only "
             "(fair comparison across runs topped up to different lengths); "
             "per-run report_diagnostics figures are unaffected",
    )
    parser.add_argument(
        "--rerolls", choices=["ignore", "replace", "compare"], default="ignore",
        help="how to treat re-roll runs (reroll__ prefix; CONTEXT.md, ADR-0007): "
             "'ignore' (default) drops them so the full-sweep report is unchanged; "
             "'replace' substitutes each re-roll for its original; 'compare' keeps "
             "only the paired re-roll+original configs, capped at their max-common "
             "epoch (paired by the reroll_of reference)",
    )
    parser.add_argument(
        "--skip-per-run-figures", action="store_true",
        help="skip rendering report_diagnostics for each run "
             "(cross-run figures + tables only — the fast JSON-only pass)",
    )
    parser.add_argument(
        "--exclude-file", type=str, action="append", default=[], metavar="PATH",
        help="path to a text file of run names to drop before plotting (one per "
             "line; '#' comments and trailing columns ignored — reads the "
             "low_accuracy_runs.txt format directly). Repeatable",
    )
    parser.add_argument(
        "--exclude-run", type=str, action="append", default=[], metavar="NAME",
        help="run_name to drop before plotting. Repeatable",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None, metavar="DIR",
        help="write tables + cross-run figures here instead of into the sweep "
             "dir (use with --exclude-* / coordinate filters to keep the "
             "full-set artefacts intact). When a coordinate filter is active and "
             "this is omitted, output defaults to <sweep>/subsets/<filter-slug>/",
    )
    add_filter_args(parser)
    args = parser.parse_args()

    # Line-buffer stdout so progress streams live even when piped / redirected /
    # captured (non-TTY block-buffers by default, making the run look hung).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {sweep_dir}")

    rows = load_sweep(sweep_dir, max_epoch=args.max_epoch)
    if not rows:
        print(f"No runs found under {sweep_dir} (need per-run history.json).")
        return

    # Re-roll handling (CONTEXT.md "Re-roll", ADR-0007). Applied first so the
    # default 'ignore' drops re-roll dirs before anything else sees them (the
    # existing full-sweep report is byte-unchanged even when re-rolls coexist in
    # the dir). 'compare' then caps the kept rows at their max-common epoch — by
    # reloading at that horizon when the user hasn't pinned --max-epoch — so an
    # original and its (differently-trained) re-roll are compared fairly.
    rows = select_for_rerolls(rows, args.rerolls)
    if not rows:
        print(f"No runs left after --rerolls {args.rerolls}.")
        return
    if args.rerolls == "compare" and args.max_epoch is None:
        cap = _common_max_epoch(rows)
        if cap is not None:
            rows = select_for_rerolls(
                load_sweep(sweep_dir, max_epoch=cap), "compare"
            )
            print(f"--rerolls compare: capped at max-common epoch {cap}")

    # Coordinate filter (CONTEXT.md): keep only runs whose resolved coordinates
    # match. Applied before the name exclusions so the two compose (AND). Each
    # run's coordinates come from history["meta"] via its row (ADR-0006: no
    # config.json re-read on this side); a run missing a filtered coordinate is
    # excluded with a warning by run_matches.
    filters = parse_filter_args(args)
    if filters:
        before = len(rows)
        rows = [
            r for r in rows
            if run_matches(coords_from_meta(r), filters, run_name=r["run_name"])
        ]
        spec = ", ".join(f"{k}∈{{{', '.join(map(str, sorted(v)))}}}"
                         for k, v in filters.items())
        print(f"Coordinate filter kept {len(rows)}/{before} run(s): {spec}")
        if not rows:
            print("No runs match the coordinate filter — nothing to plot.")
            return

    excluded: set[str] = set(args.exclude_run)
    for fpath in args.exclude_file:
        fpath = Path(fpath)
        if not fpath.is_file():
            parser.error(f"--exclude-file does not exist: {fpath}")
        excluded |= read_exclude_file(fpath)
    if excluded:
        before = len(rows)
        kept = [r for r in rows if r["run_name"] not in excluded]
        dropped = before - len(kept)
        matched = {r["run_name"] for r in rows} & excluded
        unmatched = excluded - matched
        print(f"Excluding {dropped} run(s) from {before} ({len(matched)} names matched)")
        if unmatched:
            warnings.warn(
                f"{len(unmatched)} excluded name(s) matched no run under "
                f"{sweep_dir}: {sorted(unmatched)}",
                RuntimeWarning,
            )
        rows = kept
        if not rows:
            print("All runs excluded — nothing to plot.")
            return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif filters:
        # Default a filtered run under subsets/<slug> so the full-sweep tables +
        # figures at the sweep root are never overwritten (override with --out-dir).
        out_dir = sweep_dir / "subsets" / _filter_slug(filters)
        print(f"Writing filtered artefacts to {out_dir} "
              "(full-sweep artefacts untouched; override with --out-dir)")
    else:
        out_dir = sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {len(rows)} run(s) under {sweep_dir}")
    _check_identity_drift(rows)
    _check_epoch_heterogeneity(rows)
    write_table(rows, out_dir)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plotters = (
        partial(plot_acc_vs_params, series_by=args.series_by),
        plot_acc_by_observable,
        plot_acc_by_scaling_knob,
        plot_acc_vs_trunc_lambda,
        plot_acc_vs_hyperparam,
    )
    for fn in plotters:
        try:
            fn(rows, fig_dir)
        except Exception as e:  # one bad figure must not abort the rest
            name = getattr(fn, "func", fn).__name__
            warnings.warn(f"{name} failed: {type(e).__name__}: {e}", RuntimeWarning)

    if not args.skip_per_run_figures:
        render_per_run_figures(sweep_dir, rows)


if __name__ == "__main__":
    main()
