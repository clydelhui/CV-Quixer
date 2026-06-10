import math
from dataclasses import dataclass, field, fields
from typing import NamedTuple, Optional, Union


def auto_gate_bound(cutoff_dim: int) -> float:
    """Photon-budget soft-clip bound for the magnitude gate params at a given Fock
    cutoff. Keeps squeezed-vacuum mean photon number sinh²(r) at ~cutoff_dim-1
    (the representable budget), which also keeps displacement under budget
    (⟨n⟩ = 2·b² < cutoff_dim-1). ≈1.54 at cutoff 6, 1.82 at cutoff 10.
    """
    return math.asinh(math.sqrt(max(cutoff_dim - 1, 1)))


@dataclass
class DataConfig:
    dataset: str = "fashionmnist"  # "fashionmnist" | "mnist"
    normalize: bool = True         # compute & cache stats on first load; False → ToTensor only
    image_size: int = 28
    patch_size: int = 4            # must divide image_size evenly
    num_classes: int = 10
    batch_size: int = 64
    num_workers: int = 2
    data_root: str = "data/"


_VALID_OBSERVABLE_TYPES = {"x", "p", "x_squared", "p_squared", "n", "prob_n"}
_LEGACY_READOUT_VALUES = {"quadrature_x", "photon_number", "pnr_distribution"}


@dataclass
class ObservableSpec:
    """User-facing observable specification.

    Attributes:
        type: One of "x", "p", "x_squared", "p_squared", "n", "prob_n".
        mode: Mode index (int), list of mode indices, or "all" for every mode.
        n:    Required iff type == "prob_n". Single int or list of ints in
              [0, cutoff_dim). Forbidden for other types.
    """

    type: str
    mode: Union[int, str, list[int]] = "all"
    n: Optional[Union[int, list[int]]] = None


class _EvalSpec(NamedTuple):
    """Normalised single-scalar observable evaluation entry."""

    type: str
    mode: int
    n: Optional[int]


def _expand_observable_specs(
    specs: list[ObservableSpec], num_modes: int, cutoff_dim: int
) -> list[_EvalSpec]:
    """Expand user specs into a flat ordered list of single-scalar evaluations.

    Order: outer = spec list position, middle = mode index, inner = n index
    (for prob_n only). This order is the source of truth for the readout
    vector and for decoder input dimension.
    """
    plan: list[_EvalSpec] = []
    for spec in specs:
        if spec.type not in _VALID_OBSERVABLE_TYPES:
            raise ValueError(
                f"Unknown observable type {spec.type!r}; "
                f"valid: {sorted(_VALID_OBSERVABLE_TYPES)}"
            )

        if isinstance(spec.mode, str):
            if spec.mode != "all":
                raise ValueError(
                    f"observable mode string must be 'all', got {spec.mode!r}"
                )
            mode_list = list(range(num_modes))
        elif isinstance(spec.mode, bool):
            raise ValueError(f"observable mode must not be bool, got {spec.mode!r}")
        elif isinstance(spec.mode, int):
            mode_list = [spec.mode]
        elif isinstance(spec.mode, list):
            mode_list = list(spec.mode)
        else:
            raise ValueError(
                f"observable mode must be int, list[int], or 'all'; got {spec.mode!r}"
            )

        for m in mode_list:
            if isinstance(m, bool) or not isinstance(m, int):
                raise ValueError(f"observable mode entries must be int, got {m!r}")
            if not (0 <= m < num_modes):
                raise ValueError(
                    f"observable mode {m} out of range [0, {num_modes})"
                )

        if spec.type == "prob_n":
            if spec.n is None:
                raise ValueError(
                    "observable type 'prob_n' requires 'n' to be specified"
                )
            if isinstance(spec.n, bool):
                raise ValueError(f"observable n must not be bool, got {spec.n!r}")
            if isinstance(spec.n, int):
                n_list = [spec.n]
            elif isinstance(spec.n, list):
                n_list = list(spec.n)
            else:
                raise ValueError(
                    f"observable n must be int or list[int], got {spec.n!r}"
                )
            for nv in n_list:
                if isinstance(nv, bool) or not isinstance(nv, int):
                    raise ValueError(
                        f"observable n entries must be int, got {nv!r}"
                    )
                if not (0 <= nv < cutoff_dim):
                    raise ValueError(
                        f"observable n={nv} out of range [0, {cutoff_dim})"
                    )
            for m in mode_list:
                for nv in n_list:
                    plan.append(_EvalSpec(type="prob_n", mode=m, n=nv))
        else:
            if spec.n is not None:
                raise ValueError(
                    f"observable type {spec.type!r} must not specify 'n'"
                )
            for m in mode_list:
                plan.append(_EvalSpec(type=spec.type, mode=m, n=None))
    return plan


@dataclass
class QuantumConfig:
    num_modes: int = 4           # number of bosonic modes (qumodes)
    num_layers: int = 1          # per-patch circuit depth: L stacked gate sequences + L-1 BS→Rot interferometers
    cutoff_dim: int = 6          # Fock space truncation (memory ~ cutoff_dim^num_modes)
    grad_mode: str = "backprop"          # "backprop" | "parameter_shift"
    param_shift_shift: float = 1.5708   # shift s for PSR (default π/2)
    bs_topology: str = "linear"         # "linear" | "ring"
    dtype: str = "complex128"           # "complex64" | "complex128"

    # Multi-head attention
    num_heads: int = 4            # parallel CV attention heads
    decoder_hidden_dim: int = 64  # hidden dim of readout→logits MLP
    # Optional: size decoder_hidden_dim relative to the decoder's input width
    # in_dim = num_heads × readout_width, as decoder_hidden_dim = max(1,
    # round(decoder_hidden_mult × in_dim)). Resolved at model build time (after
    # any target_params auto-scaling of num_heads), overriding decoder_hidden_dim.
    # None = off (use the static decoder_hidden_dim). Float ⇒ not a scaling_knob.
    decoder_hidden_mult: Optional[float] = None
    # Total Linear layers in the CVDecoder MLP. 2 = the historic
    # Linear(in→h)→ReLU→Linear(h→classes); >2 inserts (decoder_num_layers-2)
    # extra h→h ReLU hidden blocks. A valid (monotonic) scaling_knob.
    decoder_num_layers: int = 2

    # CNN hypernetwork: Conv(1→C1) → Conv(C1→C2) → [extra C2→C2 convs] → flatten
    # → 2D PE → [extra feature_dim→feature_dim linears] → Linear → gate params.
    # The two channel-transition convs use no padding (spatial output
    # h_out = patch_size - 2*(cnn_kernel_size - 1)); any extra convs use
    # same-padding so they preserve h_out / feature_dim.
    cnn_channels_1: int = 8    # output channels of first conv layer
    cnn_channels_2: int = 16   # output channels of second conv layer (a valid scaling_knob, but num_heads is the default)
    cnn_kernel_size: int = 3   # kernel size for both conv layers
    # Total conv layers in the CNN stack. 2 = the historic conv1→conv2; >2
    # appends (cnn_num_conv_layers-2) same-padding C2→C2 convs. A valid scaling_knob.
    cnn_num_conv_layers: int = 2
    # Total Linear layers in the hypernetwork DNN section (after the 2D-PE add).
    # 1 = the historic single feature_dim→gate_params projection; >1 prepends
    # (hypernet_num_linear_layers-1) feature_dim→feature_dim Tanh blocks before the
    # final projection. A valid scaling_knob.
    hypernet_num_linear_layers: int = 1

    # Matrix polynomial degree for LCU attention (P(M) = Σ c_j M^j, j=0..d)
    poly_degree: int = 2           # keep ≤ 4; d=2 or d=3 recommended

    # CVQNN block W applied to the post-polynomial (post-selected) state before
    # observable readout — a fixed, per-image, trainable Killoran-style circuit
    # with owned nn.Parameters (input-independent), distinct from the
    # hypernetwork-emitted per-patch unitaries U_i. Each W layer is the canonical
    # two-interferometer Killoran form (BS→R)→S→(BS→R)→D→K (the per-patch U_i
    # drops the leading interferometer, trivial on its vacuum input; W acts on a
    # non-vacuum state so it is restored). Zero-init ⇒ W=I at start.
    #   cvqnn_num_layers = 0 disables W entirely → state_dict byte-identical to a
    #     pre-W model (the clean ablation baseline / checkpoint-compat switch).
    #   A valid (monotonic) scaling_knob, but too coarse for budget targeting —
    #     keep num_heads the budget knob; use this as a capacity/ablation axis.
    cvqnn_num_layers: int = 1
    # Weight of the SEPARATE W truncation penalty 1 - ‖W|ψ⟩‖² added to the
    # training loss (independent of trunc_lambda / trunc_penalty — W's leakage is
    # a distinct concern, always computed since it is the norm used for the
    # post-W renormalisation). Defaulted an order below the per-patch
    # trunc_lambda because W's single-block leakage compounds far less than the
    # per-patch penalty does through the polynomial powers. 0 → tracked but not
    # penalised.
    cvqnn_trunc_lambda: float = 0.01

    # Seq-to-seq stacked model (model="quantum_stacked" only — the canonical
    # quantum/quantum_shared models ignore these four fields; see ADR-0002).
    # Number of uniform seq-to-seq blocks. Does NOT count the optional final
    # aggregator block (pooling="quixer"); >= 1 is enforced — a 0-block model
    # with an aggregator on raw patches is exactly the existing CVQuixer, which
    # remains the sole owner of that configuration. A valid (monotonic)
    # scaling_knob, but too coarse for budget targeting.
    num_seq2seq_blocks: int = 1
    # How the final block's N tokens become the decoder input: "mean" pools over
    # positions; "quixer" appends a canonical seq-to-one aggregator block
    # (vacuum input, no query unitaries). Both end at the same H×R decoder width.
    pooling: str = "mean"
    # Identity residual x + block(x) from block 2 onward (block 1's input/output
    # widths differ, so it never has one). False = pure-pipeline ablation.
    block_residual: bool = True
    # Weight of the SEPARATE query truncation penalty mean_i(1 - ‖U_{q,i}|0⟩‖²)
    # added to the training loss. Like cvqnn_trunc_lambda (and unlike the
    # compounding per-patch trunc_lambda), query leakage is a single-application
    # leak — the query unitary fires once per position, before the polynomial —
    # hence the lighter default. 0 → tracked but not penalised.
    query_trunc_lambda: float = 0.01

    # Soft-clip on the magnitude gate params (squeeze r, displacement re/im) via
    # b·tanh(x/b), keeping them in (-b, b). This stops the gates from driving the
    # state far past the Fock cutoff, where the truncated sub-isometries become
    # numerically degenerate and NaN heads (seen at high num_heads). None = off
    # (no clip; exact historic behaviour, checkpoint-compatible); a bounded model
    # is NOT weight-compatible with an unbounded one. Choose b at the representable
    # photon budget: `auto_gate_bound(cutoff_dim)` = asinh(√(cutoff-1)) ≈ 1.54 at
    # cutoff 6 (so squeezed-vacuum ⟨n⟩ ≈ cutoff-1). Avoid large b (~4 → ⟨n⟩≈745 at
    # cutoff 6, the degenerate regime). full_experiment.py's `--gate-param-bound
    # auto` resolves this to a concrete float for the run's cutoff.
    gate_param_bound: Optional[float] = None

    # Auto-scaling: set target_params > 0 to binary-search `scaling_knob` (an
    # integer architecture field) so the built model's trainable-param count hits
    # the budget. scaling_knob defaults to num_heads — the robust knob (sweeps
    # show cnn_channels_2 accuracy degrades with scale while num_heads holds up).
    # Other monotonic knobs (cnn_channels_2, num_modes, cnn_channels_1,
    # decoder_hidden_dim, num_layers) also work — num_layers now deepens the
    # per-patch unitary, so it monotonically increases the param count.
    target_params: int = -1
    scaling_knob: str = "num_heads"

    # Fock truncation penalty added to training loss
    trunc_penalty: str = "none"   # "none" | "norm" | "photon_number"
    trunc_lambda: float = 0.01

    # Legacy single-string readout selector. Kept for backward compatibility
    # (existing YAMLs, saved configs, trained checkpoints). Translated to the
    # canonical list at __post_init__ time.
    #   "quadrature_x"     — 〈x̂〉 per mode                       (num_modes scalars)
    #   "photon_number"    — 〈n̂〉 per mode                       (num_modes scalars)
    #   "pnr_distribution" — P(n_mode=k) for k=0..cutoff_dim-1   (num_modes × cutoff_dim values)
    readout_observable: Optional[str] = None

    # Canonical observable list. Mutually exclusive with readout_observable.
    # If both are None, defaults to a single ⟨x̂⟩-per-mode entry.
    readout_observables: Optional[list[ObservableSpec]] = None

    def __post_init__(self) -> None:
        valid_knobs = {
            f.name
            for f in fields(self)
            if f.type in (int, "int") and f.name != "target_params"
        }
        if self.scaling_knob not in valid_knobs:
            raise ValueError(
                f"scaling_knob={self.scaling_knob!r} must name an integer "
                f"QuantumConfig field; valid choices: {sorted(valid_knobs)}"
            )

        if self.decoder_hidden_mult is not None and self.decoder_hidden_mult <= 0:
            raise ValueError(
                f"decoder_hidden_mult must be > 0, got {self.decoder_hidden_mult}"
            )

        if self.cvqnn_num_layers < 0:
            raise ValueError(
                f"cvqnn_num_layers must be >= 0, got {self.cvqnn_num_layers}"
            )

        if self.num_seq2seq_blocks < 1:
            raise ValueError(
                f"num_seq2seq_blocks must be >= 1, got {self.num_seq2seq_blocks}"
            )
        if self.pooling not in ("mean", "quixer"):
            raise ValueError(
                f"pooling must be 'mean' or 'quixer', got {self.pooling!r}"
            )
        if self.query_trunc_lambda < 0:
            raise ValueError(
                f"query_trunc_lambda must be >= 0, got {self.query_trunc_lambda}"
            )

        if (
            self.readout_observable is not None
            and self.readout_observables is not None
        ):
            raise ValueError(
                "Set either readout_observable (legacy string) or "
                "readout_observables (new list), not both."
            )
        if (
            self.readout_observable is not None
            and self.readout_observable not in _LEGACY_READOUT_VALUES
        ):
            raise ValueError(
                f"readout_observable must be one of "
                f"{sorted(_LEGACY_READOUT_VALUES)}, "
                f"got {self.readout_observable!r}"
            )

        if self.readout_observables is not None:
            specs = self.readout_observables
        elif self.readout_observable == "quadrature_x":
            specs = [ObservableSpec(type="x", mode="all")]
        elif self.readout_observable == "photon_number":
            specs = [ObservableSpec(type="n", mode="all")]
        elif self.readout_observable == "pnr_distribution":
            specs = [
                ObservableSpec(
                    type="prob_n",
                    mode="all",
                    n=list(range(self.cutoff_dim)),
                )
            ]
        else:
            specs = [ObservableSpec(type="x", mode="all")]

        self._observable_plan = _expand_observable_specs(
            specs, self.num_modes, self.cutoff_dim
        )


@dataclass
class ClassicalConfig:
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 30
    optimizer: str = "adam"      # "adam" or "sgd"
    weight_decay: float = 1e-4
    seed: int = 42
    checkpoint_dir: str = "results/checkpoints/"
    log_dir: str = "results/logs/"
    log_interval: int = 10       # log every N batches


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    model: str = "quantum"       # "quantum" or "classical"
    data: DataConfig = field(default_factory=DataConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_wandb: bool = False
