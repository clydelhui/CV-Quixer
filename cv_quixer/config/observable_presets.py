"""Named observable readout presets for CV-Quixer sweeps.

A small registry mapping short preset names to `list[ObservableSpec]`, shared by
`experiments/full_experiment.py` (`--observables NAME`) and
`experiments/sweep.py` so run names and observable configurations stay in sync.

The expanded readout plan — and hence the decoder input width — is derived from
these specs by `QuantumConfig.__post_init__` (see `cv_quixer/config/schema.py`);
this module only chooses *which* specs to hand it.
"""

from __future__ import annotations

from cv_quixer.config.schema import ObservableSpec

# Ordered for stable display (tables, CLI help). `pnr` is last because its
# expanded width depends on cutoff_dim, unlike the fixed-width presets.
PRESET_NAMES: tuple[str, ...] = ("x", "xp", "xpxsps", "n", "xpn", "pnr")


def _presets(cutoff_dim: int) -> dict[str, list[ObservableSpec]]:
    """Build the preset → specs mapping for a given Fock cutoff.

    Only `pnr` depends on `cutoff_dim` (it enumerates photon numbers
    0..cutoff_dim-1), but all presets go through this factory so callers have a
    single uniform interface.
    """
    return {
        # ⟨x̂⟩ per mode — matches the schema default / baseline readout.
        "x": [ObservableSpec(type="x", mode="all")],
        # ⟨x̂⟩, ⟨p̂⟩ per mode.
        "xp": [
            ObservableSpec(type="x", mode="all"),
            ObservableSpec(type="p", mode="all"),
        ],
        # ⟨x̂⟩, ⟨p̂⟩, ⟨x̂²⟩, ⟨p̂²⟩ per mode — the current full_experiment.py default.
        "xpxsps": [
            ObservableSpec(type="x", mode="all"),
            ObservableSpec(type="p", mode="all"),
            ObservableSpec(type="x_squared", mode="all"),
            ObservableSpec(type="p_squared", mode="all"),
        ],
        # ⟨n̂⟩ per mode.
        "n": [ObservableSpec(type="n", mode="all")],
        # ⟨x̂⟩, ⟨p̂⟩, ⟨n̂⟩ per mode.
        "xpn": [
            ObservableSpec(type="x", mode="all"),
            ObservableSpec(type="p", mode="all"),
            ObservableSpec(type="n", mode="all"),
        ],
        # Photon-number-resolving distribution P(n=k), k=0..cutoff_dim-1, per mode.
        "pnr": [
            ObservableSpec(type="prob_n", mode="all", n=list(range(cutoff_dim))),
        ],
    }


def resolve_observables(name: str, cutoff_dim: int) -> list[ObservableSpec]:
    """Return the `list[ObservableSpec]` for a named preset.

    Args:
        name:       Preset name; one of `PRESET_NAMES`.
        cutoff_dim: Fock truncation, needed to enumerate `pnr` photon numbers.

    Returns:
        A fresh `list[ObservableSpec]` (safe to mutate / feed to QuantumConfig).

    Raises:
        ValueError: if `name` is not a known preset.
    """
    presets = _presets(cutoff_dim)
    if name not in presets:
        raise ValueError(
            f"Unknown observable preset {name!r}; valid: {sorted(presets)}"
        )
    return presets[name]
