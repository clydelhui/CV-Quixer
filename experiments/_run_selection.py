"""Coordinate filter — select a subset of a sweep's runs by configuration value.

A **coordinate filter** (see CONTEXT.md) keeps the runs whose configuration
coordinates match given values: within one coordinate the allowed values are
OR'd, across coordinates they are AND'd. The filterable coordinates are the
configuration-identity set (CONTEXT.md: Configuration identity) — every recorded
sweep coordinate.

This module is the **single shared home** for that mechanism, so the top-up tool
(`resume_sweep.py`) and the reporting tool (`report_sweep.py`) select the *same*
runs for the *same* filter — the guarantee a "top up subset X → plot subset X"
workflow depends on. It owns three things, and deliberately not a fourth:

  * ``FILTERABLE_FIELDS`` — the canonical registry of filterable coordinates
    (name, CLI flag, value type). This is the one true field list; downstream
    constants (`report_sweep.ARCH_META_FIELDS`/`CONFIG_IDENTITY_FIELDS`,
    `sweep.ARCH_AXES`) should derive from it rather than keep parallel copies.
  * ``add_filter_args`` / ``parse_filter_args`` — argparse wiring, so both tools
    expose identical ``--num-modes 2 3`` style flags for free.
  * ``run_matches`` — the matching predicate with the locked semantics
    (OR-within / AND-across / absent-coordinate → exclude-with-warning).

What it does *not* own is **coordinate extraction**: the two tools read a run's
coordinates from different sources (ADR-0006). `resume_sweep` works off
`sweep_manifest.json` and must still match runs that died before epoch 1 (no
``config.json``), so it uses ``coords_from_config_json`` (resolved values,
catches auto-scaled knobs) overlaid on ``coords_from_args`` (the always-present
replayed argv). `report_sweep` already has each run's resolved coordinates from
``history["meta"]`` and uses ``coords_from_meta`` (no ``config.json`` re-read).
The three ``coords_from_*`` helpers live here only so the field registry stays
their single source of truth; *which* source a tool reads is the tool's choice
(ADR-0006), not this module's.

Torch-free on purpose (argparse/warnings/stdlib only) so importing it never
breaks the orchestrators' deferred-torch fast path.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Callable, NamedTuple


class Field(NamedTuple):
    """One filterable configuration coordinate.

    ``config_path`` locates the field inside a resolved ``config.json`` (a tuple
    of nested keys), or is ``None`` when the field is not stored there (the
    observable preset *name* lives only in ``history["meta"]`` / the replayed
    ``--observables`` arg, never in ``config.json``).
    """

    name: str
    py_type: Callable
    config_path: tuple[str, ...] | None


# The canonical filterable-coordinate registry == the configuration-identity set
# (CONTEXT.md). Keep in sync with QuantumConfig (cv_quixer/config/schema.py); the
# drift guard in tests/test_run_selection.py asserts this is a superset of
# sweep.ARCH_AXES. All quantum knobs nest under config.json["quantum"]; seed under
# ["training"]; model is top-level; observables is config-less (see Field.config_path).
FILTERABLE_FIELDS: tuple[Field, ...] = (
    # Integer architecture knobs (also sweep.ARCH_AXES, plus budget fields).
    Field("num_modes", int, ("quantum", "num_modes")),
    Field("num_heads", int, ("quantum", "num_heads")),
    Field("cutoff_dim", int, ("quantum", "cutoff_dim")),
    Field("poly_degree", int, ("quantum", "poly_degree")),
    Field("num_layers", int, ("quantum", "num_layers")),
    Field("cnn_channels_1", int, ("quantum", "cnn_channels_1")),
    Field("cnn_channels_2", int, ("quantum", "cnn_channels_2")),
    Field("cnn_kernel_size", int, ("quantum", "cnn_kernel_size")),
    Field("decoder_hidden_dim", int, ("quantum", "decoder_hidden_dim")),
    Field("cnn_num_conv_layers", int, ("quantum", "cnn_num_conv_layers")),
    Field("hypernet_num_linear_layers", int, ("quantum", "hypernet_num_linear_layers")),
    Field("decoder_num_layers", int, ("quantum", "decoder_num_layers")),
    Field("cvqnn_num_layers", int, ("quantum", "cvqnn_num_layers")),
    Field("num_seq2seq_blocks", int, ("quantum", "num_seq2seq_blocks")),
    Field("target_params", int, ("quantum", "target_params")),
    Field("seed", int, ("training", "seed")),
    # Float knobs.
    Field("trunc_lambda", float, ("quantum", "trunc_lambda")),
    Field("decoder_hidden_mult", float, ("quantum", "decoder_hidden_mult")),
    Field("query_trunc_lambda", float, ("quantum", "query_trunc_lambda")),
    # String knobs. block_residual is stored as a bool in config.json but the
    # flag (and replayed arg) is "on"/"off" — normalised in _normalize().
    Field("model", str, ("model",)),
    Field("scaling_knob", str, ("quantum", "scaling_knob")),
    Field("pooling", str, ("quantum", "pooling")),
    Field("block_residual", str, ("quantum", "block_residual")),
    Field("observables", str, None),  # config-less: name only in meta / --observables
)

_FIELDS_BY_NAME = {f.name: f for f in FILTERABLE_FIELDS}


def flag_for(name: str) -> str:
    """The CLI flag for a field name (``num_modes`` -> ``--num-modes``).

    Matches full_experiment.py / sweep.py spelling exactly, so the flags this
    module adds replay verbatim against those scripts.
    """
    return "--" + name.replace("_", "-")


def _normalize(field: Field, value):
    """Coerce a raw value into the field's comparison space.

    Keeps both extraction sources comparable: ``block_residual`` is a bool in
    config.json but an "on"/"off" string everywhere else, so map it to the
    string form. Everything else passes through its declared type.
    """
    if value is None:
        return None
    if field.name == "block_residual" and isinstance(value, bool):
        return "on" if value else "off"
    try:
        return field.py_type(value)
    except (TypeError, ValueError):
        return value


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add one ``--<field>`` coordinate-filter flag per registry entry.

    Each is ``nargs="+"`` (OR within the flag) and defaults to ``None`` (flag
    absent → that coordinate is unconstrained). Grouped under "coordinate
    filters" in ``--help``. ``block_residual`` takes the "on"/"off" strings the
    matching arg uses rather than a bool.
    """
    group = parser.add_argument_group(
        "coordinate filters",
        "Keep only runs whose configuration matches. Within a flag the values "
        "are OR'd; across flags they are AND'd (and AND with --runs). A run "
        "whose value for a filtered coordinate is absent/unresolved is excluded "
        "with a warning.",
    )
    for field in FILTERABLE_FIELDS:
        arg_type = str if field.name == "block_residual" else field.py_type
        group.add_argument(
            flag_for(field.name), nargs="+", type=arg_type, default=None,
            metavar=field.name.upper(),
            help=f"coordinate filter: keep runs whose {field.name} is one of these",
        )


def parse_filter_args(ns: argparse.Namespace) -> dict[str, set]:
    """Collect the set coordinate-filter flags from a parsed namespace.

    Returns ``{field_name: {allowed values}}`` for every flag the user set,
    typed via the registry. Unset flags are omitted, so an empty dict means
    "no coordinate filter" (``run_matches`` then keeps everything).
    """
    filters: dict[str, set] = {}
    for field in FILTERABLE_FIELDS:
        values = getattr(ns, field.name, None)
        if values is not None:
            filters[field.name] = {_normalize(field, v) for v in values}
    return filters


def coords_from_config_json(cfg: dict) -> dict:
    """Flatten a resolved ``config.json`` dict into the registry's namespace.

    Reads each field from its ``config_path`` (``quantum.*`` knobs, ``training.seed``,
    top-level ``model``). A field is included only when its key is actually
    present — a missing key (e.g. ``num_seq2seq_blocks`` on a pre-ADR-0003 run)
    is left out so a filter on it reports "unresolved" rather than matching None.
    The observable preset is config-less and never appears here.
    """
    coords: dict = {}
    for field in FILTERABLE_FIELDS:
        if field.config_path is None:
            continue
        node = cfg
        for key in field.config_path[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        last = field.config_path[-1]
        if isinstance(node, dict) and last in node:
            coords[field.name] = _normalize(field, node[last])
    return coords


def coords_from_args(args: list[str]) -> dict:
    """Parse a replayed ``full_experiment.py`` argv into the registry namespace.

    Single-valued lookup by flag (each grid point fixes one value per knob).
    This is the fallback for runs that never wrote a ``config.json`` and the only
    source for the config-less ``observables`` coordinate. Auto-scaled knobs
    (e.g. a budget-mode ``num_heads``) are *not* in the argv, so they stay
    unresolved here — config.json is the resolved source when present.
    """
    by_flag = {flag_for(f.name): f for f in FILTERABLE_FIELDS}
    coords: dict = {}
    for i, token in enumerate(args[:-1]):
        field = by_flag.get(token)
        if field is not None:
            coords[field.name] = _normalize(field, args[i + 1])
    return coords


def coords_from_meta(meta: dict) -> dict:
    """Extract registry coordinates from a ``report_sweep`` summary row / history meta.

    `report_sweep` builds one row per run from ``history["meta"]`` with the
    *resolved* (post-auto-scaling) values already keyed by registry field name, so
    this is its extraction source (ADR-0006) — no ``config.json`` re-read. Only
    non-``None`` values are kept: a field absent or ``None`` for a run (e.g. an old
    run that never wrote it) stays out of the dict, so ``run_matches`` reports it
    *unresolved* (warn + exclude) rather than matching it as ``None``.
    ``block_residual`` (a bool in meta) is normalised to the "on"/"off" string the
    filter uses. The observable preset is read from ``observables`` or, in raw
    meta, ``observables_name``.
    """
    coords: dict = {}
    for field in FILTERABLE_FIELDS:
        if field.name == "observables":
            value = meta.get("observables")
            if value is None:
                value = meta.get("observables_name")
        else:
            value = meta.get(field.name)
        if value is not None:
            coords[field.name] = _normalize(field, value)
    return coords


def run_matches(coords: dict, filters: dict, *, run_name: str | None = None) -> bool:
    """True iff ``coords`` satisfies every coordinate filter.

    Semantics (locked; ADR-0006): for each filtered field the run's value must be
    in the allowed set (OR within a field); all filtered fields must pass (AND
    across). A filtered field **absent from ``coords``** (unresolved — e.g. an
    auto-scaled knob on a run that never wrote ``config.json``) emits one warning
    naming the field/run and excludes the run; it is never matched as None. An
    empty ``filters`` keeps every run.
    """
    for field, allowed in filters.items():
        if field not in coords:
            where = f"{run_name}: " if run_name else ""
            warnings.warn(
                f"{where}cannot apply coordinate filter on '{field}' — value "
                "unresolved for this run (not recorded in its metadata/config/"
                "args, e.g. an auto-scaled knob or a field newer than the run); "
                "excluding it.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        if coords[field] not in allowed:
            return False
    return True
