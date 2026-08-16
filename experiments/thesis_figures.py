"""Thesis-ready renderings of the three headline comparison figures.

`report_sweep.py` / `report_sweep_compare.py` are general-purpose reporting
tools: their labels are the raw code identifiers (`quantum_shared`,
`positional_encoding`, `xpxsps_pnr`), their titles read like variable names, and
they always draw `errorbar` — which is meaningless for the single-seed runs this
thesis reports. This script is a thin *presentation* layer over them. It
reimplements no run parsing and no grouping: it imports `report_sweep`'s loader
and configuration-identity helpers and owns only the styling.

Three figures, matching the three thesis comparisons:

    params  best test accuracy vs parameter count, for the three model variants
            (source: the extended 25-epoch subsets)
    pe      effect of the positional-encoding variant  (3-way ablation, 10 epochs)
    coeff   effect of freezing the combination coefficients (3-way, 10 epochs)

Differences from the generic figures, all deliberate:

  * legends are plain language, and name the variants exactly `quantum`,
    `quantum_shared`, `quantum_stacked` (never a `sweep/model/observables` triple)
  * the ablation figures colour their per-configuration trend lines by model
    variant and carry a three-entry legend — the generic figures suppress the
    legend entirely once there are more than `MAX_LEGEND_CHAINS` lines, which is
    the case for both ablations (16 configurations)
  * no error bars anywhere: every run is single-shot (`seed=42`)
  * categorical axes are ordered baseline-first (`none` before `1d`/`2d`),
    not lexicographically
  * parameter counts are plotted in millions, so no bare `1e6` offset appears

Reads only JSON (no torch / model rebuild). Output goes to a `thesis/` subdir
beside the generic figures, so re-running `report_sweep.py` can never clobber it,
as PNG (for review) and PDF (vector, for LaTeX inclusion).

Usage:
    uv run python experiments/thesis_figures.py                  # all three
    uv run python experiments/thesis_figures.py --figure pe
    uv run python experiments/thesis_figures.py --figure coeff --allow-incomplete
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Shared thesis look + the model-variant identities, so a variant keeps one
# colour across every figure in the chapter (see also thesis_run_figures.py).
from _thesis_style import (
    DPI,  # noqa: F401  (re-exported for callers/tests)
    FIGSIZE,
    MODEL_COLORS,
    MODEL_MARKERS,
    MODEL_ORDER,
    OPACITY_MODES,
    REPO_ROOT,
)
from _thesis_style import CaptionBook, SINGLE_SEED
from _thesis_style import save as _save
from _thesis_style import show_path as _show
from _thesis_style import style_axes as _style_axes
from _thesis_style import top_variant_legend as _variant_legend

# Reuse the single-sweep loader + configuration-identity grouping (JSON only).
# NB: read from the run dirs, never from summary.csv — those go stale whenever a
# run is topped up without the report being re-run.
from report_sweep import (
    _FIELD_INDEX,
    _config_groups,
    _dependent_fields,
    _varying_fields,
    load_sweep,
)

SWEEPS = REPO_ROOT / "results" / "sweeps"

# ---------------------------------------------------------------------------
# Display-name layer: the only place raw config values become figure text.
# ---------------------------------------------------------------------------

# The model-variant names/colours/markers live in _thesis_style (shared with the
# per-run suite). Below: only the axes this script alone owns.

# Categorical ablation axes: display order (baseline first) + tick labels.
PE_ORDER = ["none", "1d", "2d"]
PE_LABELS = {
    "none": "None",
    "1d": "1-D sinusoidal",
    "2d": "2-D sinusoidal",
}
COEFF_ORDER = ["none", "lcu", "lcu_poly"]
COEFF_LABELS = {
    "none": "Learned\n(baseline)",
    "lcu": "Frozen LCU\nweights",
    "lcu_poly": "Frozen LCU\n+ polynomial",
}

# ---------------------------------------------------------------------------
# Completeness guard
# ---------------------------------------------------------------------------


class IncompleteData(RuntimeError):
    """Raised when the runs behind a figure are not comparable as plotted."""


def _report_problems(problems: list[str], allow_incomplete: bool) -> None:
    if not problems:
        return
    detail = "\n".join(f"    - {p}" for p in problems)
    msg = f"figure data is incomplete:\n{detail}"
    if not allow_incomplete:
        raise IncompleteData(
            f"{msg}\n  Top the runs up (experiments/resume_sweep.py) or pass "
            "--allow-incomplete to render anyway."
        )
    print(f"  ! {msg}\n  (rendering anyway: --allow-incomplete)")


def _check_epochs(rows: list[dict], required: int, problems: list[str]) -> None:
    """Collect runs that did not reach the comparison horizon."""
    for r in sorted(rows, key=lambda r: str(r["run_name"])):
        n = r.get("n_epochs") or 0
        if n < required:
            problems.append(f"{r['run_name']}: {n} epoch(s), needs {required}")


# ---------------------------------------------------------------------------
# Figure 1 — accuracy vs parameter count
# ---------------------------------------------------------------------------


def figure_params(
    sweep_dirs: list[Path], out_dir: Path, max_epoch: int | None,
    required_epochs: int, allow_incomplete: bool, book: CaptionBook,
) -> None:
    """Best test accuracy vs trainable parameter count, one point per config.

    One marker per configuration identity (the runs are single-seed, so this is
    one marker per run), coloured and shaped by model variant. Deliberately a
    scatter: the configurations differ along several architecture axes at once,
    so connecting them would imply a 1-D trend that does not exist.
    """
    rows: list[dict] = []
    for d in sweep_dirs:
        rows.extend(load_sweep(d, max_epoch=max_epoch))
    if not rows:
        raise IncompleteData(f"no runs found under {[str(d) for d in sweep_dirs]}")

    problems: list[str] = []
    _check_epochs(rows, required_epochs, problems)
    _report_problems(problems, allow_incomplete)

    groups = {k: g for k, g in _config_groups(rows).items() if g["x"] is not None}
    mi = _FIELD_INDEX["model"]
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key, g in groups.items():
        by_model[str(key[mi])].append((g["x"] / 1e6, g["acc"][0]))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for model in MODEL_ORDER:
        pts = sorted(by_model.get(model, []))
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            linestyle="none", marker=MODEL_MARKERS[model], markersize=7,
            color=MODEL_COLORS[model], markeredgecolor="white",
            markeredgewidth=0.6, label=model,
        )
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Best test accuracy")
    _style_axes(ax)
    _variant_legend(ax, list(by_model))
    _save(fig, out_dir, "acc_vs_params_compare")
    book.add(
        stem="acc_vs_params_compare",
        short="Best test accuracy versus parameter count",
        body=("One marker per configuration, coloured and shaped by model "
              "variant. Deliberately unconnected: the configurations differ "
              "along several architecture axes at once, so a line would imply "
              "a one-dimensional trend that does not exist."),
        facts={
            "Data": (f"FashionMNIST test set, {len(groups)} configurations "
                     f"trained for {required_epochs} epochs"),
            "Caveat": SINGLE_SEED,
        },
    )
    print(f"    {len(groups)} configuration(s) across {len(rows)} run(s)")


# ---------------------------------------------------------------------------
# Figures 2 and 3 — categorical ablation responses
# ---------------------------------------------------------------------------


def figure_ablation(
    sweep_dir: Path, out_dir: Path, field: str, order: list[str],
    labels: dict[str, str], title: str, xlabel: str, stem: str,
    max_epoch: int | None, required_epochs: int, allow_incomplete: bool,
    book: CaptionBook, opacity: str = "translucent",
) -> None:
    """Best test accuracy against a categorical ablation axis.

    Every configuration is drawn as its own *all-else-equal* trend line — the
    same chain construction `report_sweep._plot_acc_vs_field` uses: configs that
    agree on every identity coordinate except `field` (and except the dependent
    fields, which are slaved to the independent axes). Lines are coloured by
    model variant and drawn thin and semi-transparent, so the per-configuration
    spread is visible while the legend stays a readable three entries.
    """
    rows = load_sweep(sweep_dir, max_epoch=max_epoch)
    if not rows:
        raise IncompleteData(f"no runs found under {sweep_dir}")

    problems: list[str] = []
    _check_epochs(rows, required_epochs, problems)

    fi = _FIELD_INDEX[field]
    mi = _FIELD_INDEX["model"]
    groups = {k: g for k, g in _config_groups(rows).items() if k[fi] is not None}

    unknown = {str(k[fi]) for k in groups} - set(order)
    if unknown:
        raise IncompleteData(
            f"{field} value(s) {sorted(unknown)} have no display label — add "
            f"them to the *_ORDER / *_LABELS maps in {Path(__file__).name}"
        )

    # Chains: everything else equal, minus this sweep's derived fields.
    dependent = _dependent_fields(rows) - {field}
    other_varying = [
        f for f in _varying_fields(groups) if f != field and f not in dependent
    ]
    ov_idx = [_FIELD_INDEX[f] for f in other_varying]
    by_chain: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for key, g in groups.items():
        by_chain[tuple(key[i] for i in ov_idx)][key] = g

    for chain_key in sorted(by_chain, key=str):
        chain = by_chain[chain_key]
        present = {str(k[fi]) for k in chain}
        missing = [a for a in order if a not in present]
        if missing:
            model = str(next(iter(chain))[mi])
            problems.append(
                f"{model} configuration {chain_key}: no run for "
                f"{field}={', '.join(missing)}"
            )
    _report_problems(problems, allow_incomplete)

    line_a, marker_a, _suffix = OPACITY_MODES[opacity]
    pos = {v: i for i, v in enumerate(order)}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    models_seen: set[str] = set()
    for chain_key in sorted(by_chain, key=str):
        chain = by_chain[chain_key]
        model = str(next(iter(chain))[mi])
        models_seen.add(model)
        pts = sorted((pos[str(k[fi])], g["acc"][0]) for k, g in chain.items())
        base = MODEL_COLORS.get(model, "#666666")
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            marker=MODEL_MARKERS.get(model, "o"), markersize=4.5, linewidth=1.3,
            color=to_rgba(base, line_a),
            markerfacecolor=to_rgba(base, marker_a),
            markeredgecolor=to_rgba(base, marker_a),
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[v] for v in order])
    # Enough margin to clear the markers and no more: the categorical
    # positions carry the comparison, so wide gutters only shrink the plot.
    ax.set_xlim(-0.18, len(order) - 1 + 0.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Best test accuracy ({required_epochs} epochs)")
    _style_axes(ax)
    _variant_legend(ax, sorted(models_seen))
    _save(fig, out_dir, f"{stem}{_suffix}")
    book.add(
        stem=f"{stem}{_suffix}",
        short=title,
        body=("Each line connects the configurations that differ only in this "
              "setting, holding every other architecture coordinate fixed; "
              "colour gives the model variant."),
        facts={
            "Data": (f"FashionMNIST test set, {len(by_chain)} configurations, "
                     f"best accuracy over {required_epochs} epochs"),
            "Settings": ", ".join(labels[v] for v in order),
            "Caveat": SINGLE_SEED,
        },
    )
    print(f"    {len(by_chain)} configuration(s) across {len(rows)} run(s)")


# ---------------------------------------------------------------------------
# Figure registry + CLI
# ---------------------------------------------------------------------------

FIGURES = {
    "params": {
        "sweep_dirs": [
            SWEEPS / "extended25_subsets" / "quantum",
            SWEEPS / "extended25_subsets" / "shared",
            SWEEPS / "extended25_subsets" / "stacked",
        ],
        "out_dir": SWEEPS / "extended25_subsets" / "compare" / "figures" / "thesis",
        "required_epochs": 25,
        "max_epoch": 25,
    },
    "pe": {
        "sweep_dirs": [SWEEPS / "pe_3way_merged"],
        "out_dir": SWEEPS / "pe_3way_merged" / "figures" / "thesis",
        "required_epochs": 10,
        "max_epoch": 10,
        "field": "positional_encoding",
        "order": PE_ORDER,
        "labels": PE_LABELS,
        "title": "Effect of positional encoding on test accuracy",
        "xlabel": "Positional encoding",
        "stem": "acc_vs_positional_encoding",
    },
    "coeff": {
        "sweep_dirs": [SWEEPS / "coeff_3way_merged"],
        "out_dir": SWEEPS / "coeff_3way_merged" / "figures" / "thesis",
        "required_epochs": 10,
        "max_epoch": 10,
        "field": "coeff_ablation",
        "order": COEFF_ORDER,
        "labels": COEFF_LABELS,
        "title": "Effect of freezing the combination coefficients on test accuracy",
        "xlabel": "Coefficient ablation",
        "stem": "acc_vs_coeff_ablation",
    },
}


def render(name: str, args: argparse.Namespace,
           books: dict[Path, CaptionBook]) -> None:
    spec = dict(FIGURES[name])
    if args.sweep_dir:
        spec["sweep_dirs"] = [Path(d) for d in args.sweep_dir]
    if args.out_dir:
        spec["out_dir"] = Path(args.out_dir)
    if args.max_epoch is not None:
        spec["max_epoch"] = args.max_epoch
        spec["required_epochs"] = args.max_epoch

    print(f"\n[{name}] {_show(spec['out_dir'])}")
    for d in spec["sweep_dirs"]:
        if not Path(d).is_dir():
            raise IncompleteData(f"sweep dir not found: {d}")

    book = books.setdefault(spec["out_dir"], CaptionBook(spec["out_dir"]))
    if name == "params":
        figure_params(
            spec["sweep_dirs"], spec["out_dir"], spec["max_epoch"],
            spec["required_epochs"], args.allow_incomplete, book,
        )
    else:
        modes = (
            list(OPACITY_MODES) if args.opacity == "all" else [args.opacity]
        )
        for mode in modes:
            figure_ablation(
                spec["sweep_dirs"][0], spec["out_dir"], spec["field"],
                spec["order"], spec["labels"], spec["title"], spec["xlabel"],
                spec["stem"], spec["max_epoch"], spec["required_epochs"],
                args.allow_incomplete, book, opacity=mode,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the thesis-ready comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--figure", choices=[*FIGURES, "all"], default="all",
        help="which figure to render",
    )
    parser.add_argument(
        "--sweep-dir", action="append",
        help="override the source sweep dir(s) for the selected figure "
             "(repeatable; requires a single --figure)",
    )
    parser.add_argument("--out-dir", help="override the output directory")
    parser.add_argument(
        "--max-epoch", type=int,
        help="override the comparison horizon (also the completeness threshold)",
    )
    parser.add_argument(
        "--opacity", choices=[*OPACITY_MODES, "all"], default="translucent",
        help="how the ablation figures' overlapping chains are drawn; 'all' "
             "renders every mode side by side for comparison (ignored by "
             "--figure params)",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="downgrade the completeness guard to a warning and render anyway",
    )
    args = parser.parse_args()

    names = list(FIGURES) if args.figure == "all" else [args.figure]
    if (args.sweep_dir or args.out_dir) and len(names) > 1:
        parser.error("--sweep-dir / --out-dir require a single --figure")

    books: dict[Path, CaptionBook] = {}
    failures: list[str] = []
    for name in names:
        try:
            render(name, args, books)
        except IncompleteData as exc:
            failures.append(f"[{name}] {exc}")
            print(f"  ✗ skipped: {exc}")
    for book in books.values():
        book.write()
    if failures:
        print(f"\n{len(failures)} figure(s) not rendered.")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
