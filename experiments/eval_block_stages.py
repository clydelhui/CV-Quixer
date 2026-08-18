"""Per-block success probabilities + truncation streams for a stacked run.

`StackedCVQuixer.forward` keeps only the decoder-input stage's success
probabilities and discards the earlier blocks' (`cv_seq2seq.py`), and it
reduces all three truncation streams to a flat mean over blocks. So for a
multi-block run the standard artefacts cannot say which block is heralding or
which block is leaking — even though the forward pass computes both per block.

This script re-evaluates a trained run block by block, from its saved
checkpoints, over the *same* test split, and writes one **sidecar** npz per
(epoch, block) beside the existing predictions:

    predictions/epoch_NNNN_block0.npz    success_probs, patch_trunc,
                                         query_trunc, w_trunc

Purely additive — no existing artefact is read-modified-written, and a run that
has never been through this script renders exactly as before. The figures in
`report_diagnostics.py` / `thesis_run_figures.py` pick the sidecars up
automatically and emit one file per stage.

Cost: block *b*'s input is produced by blocks 0..b-1, so evaluating **every**
block costs the same as evaluating the last one — a forward-only pass under
`no_grad`, roughly 3% of a training epoch per checkpoint at the shapes these
runs use.

Two self-checks run automatically and are reported as pass/fail:

  1. The decoder-input stage's recomputed success probabilities must match the
     values already in `predictions/epoch_NNNN.npz`.
  2. The mean over blocks of each truncation stream must reproduce the
     `history["epoch"]["test_*_trunc_loss"]` the training run recorded.

Run:
    uv run python experiments/eval_block_stages.py \\
        --run-dir results/sweeps/<sweep>/<run>/ \\
        [--blocks all | 0 1] \\
        [--epochs all | best | 3 7] \\
        [--batch-size 64] [--device cuda] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.evaluation import artefact_schema as schema
from cv_quixer.models import build_model
from cv_quixer.provenance import invocation_record

#: Sidecar truncation key → the history field carrying the recorded block mean.
_TRUNC_HISTORY_FIELD = {
    schema.PATCH_TRUNC: "test_trunc_loss",
    schema.QUERY_TRUNC: "test_query_trunc_loss",
    schema.W_TRUNC: "test_cvqnn_trunc_loss",
}

#: Tolerance for the two self-checks. The stored artefacts are float32 and the
#: models run complex64/128, so exact equality is not the right bar.
_CHECK_RTOL = 1e-4
_CHECK_ATOL = 1e-6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate a stacked run's per-block success "
                    "probabilities and truncation streams.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="a quantum_stacked run directory")
    p.add_argument("--blocks", nargs="+", default=["all"],
                   help="'all' or 0-indexed block numbers. 'all' costs the "
                        "same as the highest block alone")
    p.add_argument("--epochs", nargs="+", default=["all"],
                   help="'all', 'best', or explicit epoch numbers")
    p.add_argument("--batch-size", type=int, default=None,
                   help="defaults to the run's own batch size")
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu (default: best available)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace sidecars that already exist")
    return p.parse_args(argv)


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_epochs(spec: list[str], history: dict, ckpt_dir: Path) -> list[int]:
    """Map the --epochs spec to epoch numbers that actually have a checkpoint."""
    n_epochs = len(history["epoch"].get("test_acc") or [])
    if spec == ["all"]:
        wanted = list(range(1, n_epochs + 1))
    elif spec == ["best"]:
        best = history.get("meta", {}).get("best_epoch")
        if best is None:
            best = int(np.argmax(history["epoch"]["test_acc"])) + 1
        wanted = [int(best)]
    else:
        wanted = [int(s) for s in spec]

    out = []
    for e in wanted:
        if (ckpt_dir / f"epoch_{e:04d}.pt").is_file():
            out.append(e)
        else:
            warnings.warn(
                f"no checkpoint for epoch {e} "
                f"({ckpt_dir / f'epoch_{e:04d}.pt'}) — skipping.",
                RuntimeWarning, stacklevel=2,
            )
    return out


def resolve_blocks(spec: list[str], n_blocks: int) -> list[int]:
    if spec == ["all"]:
        return list(range(n_blocks))
    blocks = sorted({int(s) for s in spec})
    for b in blocks:
        if not (0 <= b < n_blocks):
            raise SystemExit(
                f"--blocks {b} out of range: this run has {n_blocks} "
                f"seq-to-seq blocks (valid 0..{n_blocks - 1})."
            )
    return blocks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_blocks(model, loader, blocks: list[int], device: torch.device,
                    *, progress: bool = True) -> dict[int, dict]:
    """Walk the stack once per batch, capturing each requested block's outputs.

    Token forwarding mirrors `StackedCVQuixer.forward` exactly — including the
    identity residual from block 2 onward — so every block sees the input it
    would see in a real forward pass.

    Returns ``{block_index: {"success_probs": (N, H, P), "patch_trunc": float,
    "query_trunc": float, "w_trunc": float}}``.
    """
    model.eval()
    highest = max(blocks)
    sp_chunks: dict[int, list[torch.Tensor]] = {b: [] for b in blocks}
    trunc_sums: dict[int, dict[str, float]] = {
        b: {k: 0.0 for k in schema.STAGE_TRUNC_KEYS} for b in blocks
    }
    n_seen = 0

    iterator = tqdm(loader, desc="blocks", unit="batch",
                    leave=False, mininterval=5.0) if progress else loader
    for patches, labels in iterator:
        patches = patches.to(device)
        bsz = labels.size(0)
        tokens: torch.Tensor | None = None
        for i, block in enumerate(model.blocks):
            if i > highest:
                break
            out, _states, sps, pt, qt, wt = block(
                patches if i == 0 else tokens
            )
            if i in sp_chunks:
                # list of H × (B, N) → (B, H, N), the layout evaluate() writes.
                sp_chunks[i].append(
                    torch.stack(sps, dim=1).detach().cpu().float()
                )
                for key, val in zip(schema.STAGE_TRUNC_KEYS, (pt, qt, wt)):
                    trunc_sums[i][key] += float(val) * bsz
            if i > 0 and model.block_residual:
                tokens = tokens + out
            else:
                tokens = out
        n_seen += bsz

    return {
        b: {
            schema.SUCCESS_PROBS: torch.cat(sp_chunks[b]).numpy().astype(
                np.float32
            ),
            **{k: trunc_sums[b][k] / max(n_seen, 1)
               for k in schema.STAGE_TRUNC_KEYS},
        }
        for b in blocks
    }


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def check_decoder_stage(run_dir: Path, epoch: int, n_blocks: int,
                        results: dict[int, dict]) -> str | None:
    """Recomputed decoder-input stage vs the values already on disk."""
    last = n_blocks - 1
    if last not in results:
        return None
    path = run_dir / "predictions" / schema.prediction_filename(epoch)
    if not path.is_file():
        return None
    with np.load(path) as npz:
        if schema.SUCCESS_PROBS not in npz:
            return None
        stored = np.asarray(npz[schema.SUCCESS_PROBS])
    got = results[last][schema.SUCCESS_PROBS]
    if stored.shape != got.shape:
        return f"FAIL shape {got.shape} vs stored {stored.shape}"
    if np.allclose(got, stored, rtol=_CHECK_RTOL, atol=_CHECK_ATOL):
        return "pass"
    worst = float(np.abs(got - stored).max())
    return f"FAIL max|Δ| = {worst:.3g}"


def check_trunc_means(history: dict, epoch: int, n_blocks: int,
                      results: dict[int, dict]) -> dict[str, str]:
    """Mean over blocks of each stream vs the recorded test_*_trunc_loss.

    Only meaningful when every block was evaluated — a partial run cannot
    reproduce the mean.
    """
    out: dict[str, str] = {}
    if len(results) != n_blocks:
        return out
    eh = history["epoch"]
    for key, field in _TRUNC_HISTORY_FIELD.items():
        series = eh.get(field) or []
        if len(series) < epoch:
            continue
        recorded = float(series[epoch - 1])
        derived = float(np.mean([results[b][key] for b in sorted(results)]))
        ok = np.isclose(derived, recorded, rtol=1e-3, atol=1e-8)
        out[field] = (
            "pass" if ok else f"FAIL derived {derived:.6g} vs "
            f"recorded {recorded:.6g}"
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not (run_dir / "config.json").is_file():
        raise SystemExit(f"{run_dir} has no config.json — not a run directory.")

    config = experiment_config_from_dict(
        json.loads((run_dir / "config.json").read_text())
    )
    if config.model != "quantum_stacked":
        raise SystemExit(
            f"--run-dir is a '{config.model}' run; per-block evaluation only "
            "applies to model='quantum_stacked' (a single-stage model's "
            "success probabilities are already complete)."
        )
    history = json.loads((run_dir / "history.json").read_text())
    n_blocks = int(config.quantum.num_seq2seq_blocks)
    blocks = resolve_blocks(args.blocks, n_blocks)
    epochs = resolve_epochs(args.epochs, history, run_dir / "checkpoints")
    if not epochs:
        raise SystemExit("no requested epoch has a checkpoint — nothing to do.")

    device = resolve_device(args.device)
    batch_size = args.batch_size or config.data.batch_size

    subset = np.load(run_dir / "subset_indices.npz")
    test_indices = subset["test_indices"]
    ds = Subset(PatchedDataset(config.data, train=False),
                indices=test_indices.tolist())
    # shuffle=False: row order must match predictions/epoch_NNNN.npz so the
    # stages are comparable per sample.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print(f"Run:          {run_dir}")
    print(f"Model:        {config.model}, {n_blocks} blocks, "
          f"pooling={config.quantum.pooling}")
    print(f"Blocks:       {blocks}")
    print(f"Epochs:       {len(epochs)} ({epochs[0]}..{epochs[-1]})")
    print(f"Test subset:  {len(ds):,} samples (reused from subset_indices.npz)")
    print(f"Device:       {device}  batch={batch_size}\n")

    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config).to(device)

    summary: list[dict] = []
    for epoch in epochs:
        targets = {
            b: preds_dir / schema.stage_prediction_filename(epoch, f"block{b}")
            for b in blocks
        }
        if not args.overwrite and all(p.is_file() for p in targets.values()):
            print(f"epoch {epoch:>4}: sidecars present → skip "
                  "(pass --overwrite to redo)")
            continue

        ckpt = torch.load(run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                          map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=True
        )
        assert not missing and not unexpected

        results = evaluate_blocks(model, loader, blocks, device)
        for b, res in results.items():
            np.savez_compressed(
                targets[b],
                **{schema.SUCCESS_PROBS: res[schema.SUCCESS_PROBS]},
                **{k: np.float32(res[k]) for k in schema.STAGE_TRUNC_KEYS},
            )

        stage_check = check_decoder_stage(run_dir, epoch, n_blocks, results)
        trunc_checks = check_trunc_means(history, epoch, n_blocks, results)
        summary.append({"epoch": epoch, "decoder_stage_check": stage_check,
                        "trunc_checks": trunc_checks})
        checks = [f"decoder-stage {stage_check}"] if stage_check else []
        checks += [f"{k} {v}" for k, v in trunc_checks.items()]
        print(f"epoch {epoch:>4}: wrote {len(results)} sidecar(s)"
              + (f"  |  {'; '.join(checks)}" if checks else ""))

    meta_path = preds_dir / "block_eval_meta.json"
    meta = (json.loads(meta_path.read_text()) if meta_path.is_file()
            else {"invocations": [], "runs": []})
    meta["invocations"].append(invocation_record())
    meta["runs"].append({"blocks": blocks, "epochs": epochs,
                         "n_test": len(ds), "checks": summary})
    meta_path.write_text(json.dumps(meta, indent=2))

    failures = [s for s in summary
                if (s["decoder_stage_check"] or "").startswith("FAIL")
                or any(v.startswith("FAIL") for v in s["trunc_checks"].values())]
    print(f"\nWrote sidecars for {len(summary)} epoch(s) → {preds_dir}")
    print(f"Provenance: {meta_path}")
    if failures:
        print(f"\n!! {len(failures)} epoch(s) failed a self-check — see above.")
        return 1
    print("All self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
