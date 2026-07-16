"""Tests for the qsvt polynomial-mode sweep builder (experiments/build_qsvt_extended.py).

The builder takes the 16 curated epoch-extension configs (run_name + source
sweep + target GPU, from results/extended_runs_25ep.txt) and fans each over the
{qsvt} polynomial mode into one fresh-from-scratch 16-run sweep manifest. The
standard arm is deliberately NOT generated — it is reused from the existing
high_epoch_* runs and compared via report_sweep_compare.py (ADR-0009).

Load-bearing invariants pinned here:

  * each source argv is replayed verbatim except: --poly-mode injected, --run-name
    gets a __<mode> marker, --runs-root repointed at the new sweep dir, --epochs
    normalised to 10 (stacked sources carry --epochs 3), --resume dropped;
  * the heavy h100-96 configs are remapped onto h200-141 by default (MIG-split
    guard) without editing the run-list;
  * runs are ordered so each GPU group is a contiguous index range, dense 0..N-1;
  * the printed sbatch commands carry the right --gres / --time / --array range
    (h200-141's wall is the 3h cluster cap).
"""

import json
import sys
from pathlib import Path

import pytest

# build_qsvt_extended lives in experiments/ (not a package) — import via sys.path
# like the other experiment-script tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import build_qsvt_extended as bqe


def _source_args(run_name, *, sweep_dir, model=None, epochs="10", resume=False):
    """A representative full_experiment.py argv as stored in a source manifest."""
    args = ["--observables", "xpxsps_pnr", "--seed", "42", "--num-layers", "1",
            "--run-name", run_name, "--runs-root", sweep_dir,
            "--subset-seed", "42", "--epochs", epochs,
            "--train-fraction", "1.0", "--test-fraction", "1.0",
            "--gate-param-bound", "auto", "--num-heads", "10", "--num-modes", "2",
            "--poly-degree", "3"]
    if model is not None:
        args += ["--model", model]
    if resume:
        args += ["--resume", f"{sweep_dir}/{run_name}/checkpoints/latest.pt"]
    return args


class TestArmRunName:
    def test_model_prefix_and_mode_marker(self):
        assert bqe.arm_run_name("manual__cw1", "qsvt", "quantum_stacked") == \
            "quantum_stacked__manual__cw1__qsvt"


class TestRewriteRunArgs:
    def _base(self):
        return _source_args("manual__cw1", sweep_dir="results/sweeps/old", epochs="3",
                             model="quantum_stacked", resume=True)

    def test_injects_polymode_marks_name_repoints_root_normalises_epochs_drops_resume(self):
        out = bqe.rewrite_run_args(
            self._base(), "qsvt", model="quantum_stacked",
            runs_root="results/sweeps/qsvt_extended_TS", target_epochs=10)
        assert out[out.index("--poly-mode") + 1] == "qsvt"
        # run-name carries the model prefix (disambiguates cross-model collisions).
        assert out[out.index("--run-name") + 1] == "quantum_stacked__manual__cw1__qsvt"
        assert out[out.index("--runs-root") + 1] == "results/sweeps/qsvt_extended_TS"
        assert out[out.index("--epochs") + 1] == "10"
        assert "--resume" not in out
        # model + arch flags replayed verbatim
        assert out[out.index("--model") + 1] == "quantum_stacked"
        assert out[out.index("--num-modes") + 1] == "2"
        assert out[out.index("--poly-degree") + 1] == "3"


class TestApplyGpuRemap:
    def test_remaps_matching_gpu_leaves_others(self):
        src = [
            {"run_name": "q_a", "gpu": "a100-40", "model": "quantum", "args": []},
            {"run_name": "s_nm3", "gpu": "h100-96", "model": "quantum_stacked", "args": []},
        ]
        out = bqe.apply_gpu_remap(src, {"h100-96": "h200-141"})
        assert out[0]["gpu"] == "a100-40"   # untouched
        assert out[1]["gpu"] == "h200-141"  # remapped
        # originals not mutated (shallow copy per run)
        assert src[1]["gpu"] == "h100-96"

    def test_default_remap_is_h100_to_h200(self):
        assert bqe.DEFAULT_GPU_REMAP == {"h100-96": "h200-141"}


class TestBuildManifest:
    def _sources(self):
        # Two a100 runs + one (already-remapped) h200 run, out of GPU order on input.
        return [
            {"run_name": "q_a", "gpu": "a100-40", "model": "quantum",
             "args": _source_args("q_a", sweep_dir="src", epochs="10")},
            {"run_name": "s_nm3", "gpu": "h200-141", "model": "quantum_stacked",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3",
                                  model="quantum_stacked")},
            {"run_name": "q_b", "gpu": "a100-40", "model": "quantum",
             "args": _source_args("q_b", sweep_dir="src", epochs="10")},
        ]

    def test_fans_over_qsvt_only(self, tmp_path):
        m = bqe.build_manifest(self._sources(), sweep_dir=tmp_path / "qsvt_extended_TS")
        # 3 configs × {qsvt} = 3 runs; no standard arm.
        assert m["n_runs"] == 3
        names = {r["run_name"] for r in m["runs"]}
        assert all(n.endswith("__qsvt") for n in names)
        assert not any(n.endswith("__standard") for n in names)
        assert m["poly_modes"] == ["qsvt"]

    def test_dense_reindex_and_contiguous_gpu_groups(self, tmp_path):
        m = bqe.build_manifest(self._sources(), sweep_dir=tmp_path / "qsvt_extended_TS")
        assert [r["index"] for r in m["runs"]] == list(range(3))
        # a100 group must come first as a contiguous block, then h200.
        gpus = [r["gpu"] for r in m["runs"]]
        assert gpus == ["a100-40"] * 2 + ["h200-141"] * 1
        groups = m["slurm_groups"]
        assert groups["a100-40"] == [0, 1]
        assert groups["h200-141"] == [2, 2]

    def test_epochs_normalised_to_ten_everywhere(self, tmp_path):
        m = bqe.build_manifest(self._sources(), sweep_dir=tmp_path / "qsvt_extended_TS")
        for r in m["runs"]:
            assert r["args"][r["args"].index("--epochs") + 1] == "10"

    def test_polymode_injected_everywhere(self, tmp_path):
        m = bqe.build_manifest(self._sources(), sweep_dir=tmp_path / "qsvt_extended_TS")
        for r in m["runs"]:
            assert r["args"][r["args"].index("--poly-mode") + 1] == "qsvt"

    def test_runs_root_repointed_at_sweep_dir(self, tmp_path):
        sweep_dir = tmp_path / "qsvt_extended_TS"
        m = bqe.build_manifest(self._sources(), sweep_dir=sweep_dir)
        for r in m["runs"]:
            assert r["args"][r["args"].index("--runs-root") + 1] == str(sweep_dir)

    def test_no_resume_anywhere(self, tmp_path):
        srcs = self._sources()
        srcs[0]["args"] = _source_args("q_a", sweep_dir="src", resume=True)
        m = bqe.build_manifest(srcs, sweep_dir=tmp_path / "qsvt_extended_TS")
        for r in m["runs"]:
            assert "--resume" not in r["args"]

    def test_manifest_carries_provenance_and_n_runs(self, tmp_path):
        m = bqe.build_manifest(self._sources(), sweep_dir=tmp_path / "qsvt_extended_TS")
        assert m["n_runs"] == len(m["runs"]) == 3
        assert isinstance(m["invocations"], list) and m["invocations"]
        assert m["target_epochs"] == 10

    def test_model_prefix_disambiguates_same_named_cross_model_configs(self, tmp_path):
        # quantum + shared sweeps share manual run-name strings (model isn't in the
        # name marker). Without the model prefix these would collide into one dir.
        sources = [
            {"run_name": "manual__cw1", "gpu": "a100-40", "model": "quantum",
             "args": _source_args("manual__cw1", sweep_dir="src")},
            {"run_name": "manual__cw1", "gpu": "a100-40", "model": "quantum_shared",
             "args": _source_args("manual__cw1", sweep_dir="src",
                                  model="quantum_shared")},
        ]
        m = bqe.build_manifest(sources, sweep_dir=tmp_path / "qsvt_extended_TS")
        names = {r["run_name"] for r in m["runs"]}
        assert len(names) == m["n_runs"] == 2, "no run-dir collisions"
        assert "quantum__manual__cw1__qsvt" in names
        assert "quantum_shared__manual__cw1__qsvt" in names

    def test_true_duplicate_selection_raises(self, tmp_path):
        dup = {"run_name": "manual__cw1", "gpu": "a100-40", "model": "quantum",
               "args": _source_args("manual__cw1", sweep_dir="src")}
        with pytest.raises(ValueError, match="duplicate run name"):
            bqe.build_manifest([dup, dict(dup)], sweep_dir=tmp_path / "qsvt_extended_TS")

    def test_unknown_gpu_raises(self, tmp_path):
        bad = [{"run_name": "x", "gpu": "v100", "model": "quantum",
                "args": _source_args("x", sweep_dir="src")}]
        with pytest.raises(ValueError, match="unknown / unusable GPU"):
            bqe.build_manifest(bad, sweep_dir=tmp_path / "qsvt_extended_TS")


class TestSbatchCommands:
    """The per-GPU-group sbatch slices are produced by the shared
    build_coeff_ablation.sbatch_commands; pin the qsvt-relevant output (h200 wall)."""

    def _manifest(self, tmp_path):
        sources = [
            {"run_name": "q_a", "gpu": "a100-40", "model": "quantum",
             "args": _source_args("q_a", sweep_dir="src")},
            {"run_name": "s_nm3", "gpu": "h200-141", "model": "quantum_stacked",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3",
                                  model="quantum_stacked")},
        ]
        return bqe.build_manifest(sources, sweep_dir=tmp_path / "qsvt_extended_TS")

    def test_one_command_per_gpu_group_with_ranges_gres_time(self, tmp_path):
        m = self._manifest(tmp_path)
        cmds = bqe.sbatch_commands(
            m, Path("results/sweeps/qsvt_extended_TS/sweep_manifest.json"))
        assert len(cmds) == 2
        a100 = next(c for c in cmds if "a100-40" in c)
        h200 = next(c for c in cmds if "h200-141" in c)
        assert "--array=0-0" in a100
        assert "--time=12:00:00" in a100
        assert "--gres=gpu:a100-40:1" in a100
        assert "--array=1-1" in h200
        assert "--time=03:00:00" in h200           # the 3h h200 cap
        assert "--gres=gpu:h200-141:1" in h200
        assert "scripts/run_sweep.sh" in a100
