"""Tests for the coefficient-ablation sweep builder (build_coeff_ablation.py).

Mirrors test_build_pe_ablation.py: the builder takes the same 16 curated configs
(run_name + source sweep + target GPU, from results/extended_runs_25ep.txt) and
fans each over the {lcu, lcu_poly} coefficient-ablation arms into one
fresh-from-scratch 32-run sweep manifest. The 'none' (full-coefficient) baseline
is deliberately NOT generated — it is reused from the existing high_epoch_* runs
(ADR-0008; compared via report_sweep_compare / notebook).

Load-bearing invariants pinned here:

  * each source argv is replayed verbatim except: --coeff-ablation injected,
    --run-name gets a __ca<v> marker, --runs-root repointed, --epochs normalised
    to 10 (stacked sources carry --epochs 3), --resume dropped;
  * only lcu/lcu_poly arms are emitted (no 'none' arm);
  * runs are ordered so each GPU group is a contiguous index range, dense 0..N-1;
  * the printed sbatch commands carry the right --gres / --time / --array range.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import build_coeff_ablation as bca


def _source_args(run_name, *, sweep_dir, model=None, epochs="10", resume=False):
    args = ["--observables", "xpxsps_pnr", "--seed", "42", "--num-layers", "1",
            "--run-name", run_name, "--runs-root", sweep_dir,
            "--subset-seed", "42", "--epochs", epochs,
            "--train-fraction", "1.0", "--test-fraction", "1.0",
            "--gate-param-bound", "auto", "--num-heads", "10", "--num-modes", "2"]
    if model is not None:
        args += ["--model", model]
    if resume:
        args += ["--resume", f"{sweep_dir}/{run_name}/checkpoints/latest.pt"]
    return args


def _make_source_sweep(tmp_path, name, runs):
    sweep_dir = tmp_path / name
    sweep_dir.mkdir()
    manifest = {
        "sweep_name": name,
        "sweep_dir": str(sweep_dir),
        "n_runs": len(runs),
        "runs": [
            {"index": i, "run_name": r["run_name"], "args": list(r["args"])}
            for i, r in enumerate(runs)
        ],
    }
    with open(sweep_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f)
    return sweep_dir


class TestRewriteRunArgs:
    def _base(self):
        return _source_args("manual__cw1", sweep_dir="results/sweeps/old", epochs="3",
                            model="quantum_stacked", resume=True)

    def test_injects_arm_marks_name_repoints_root_normalises_epochs_drops_resume(self):
        out = bca.rewrite_run_args(
            self._base(), "lcu", model="quantum_stacked",
            runs_root="results/sweeps/coeff_ablation_TS", target_epochs=10)
        assert out[out.index("--coeff-ablation") + 1] == "lcu"
        assert out[out.index("--run-name") + 1] == "quantum_stacked__manual__cw1__calcu"
        assert out[out.index("--runs-root") + 1] == "results/sweeps/coeff_ablation_TS"
        assert out[out.index("--epochs") + 1] == "10"
        assert "--resume" not in out
        assert out[out.index("--model") + 1] == "quantum_stacked"
        assert out[out.index("--num-modes") + 1] == "2"

    def test_lcu_poly_variant_marker(self):
        out = bca.rewrite_run_args(
            self._base(), "lcu_poly", model="quantum_stacked", runs_root="r",
            target_epochs=10)
        assert out[out.index("--run-name") + 1] == "quantum_stacked__manual__cw1__calcu_poly"
        assert out[out.index("--coeff-ablation") + 1] == "lcu_poly"


class TestBuildManifest:
    def _sources(self):
        return [
            {"run_name": "q_a", "gpu": "a100-40",
             "args": _source_args("q_a", sweep_dir="src", epochs="10")},
            {"run_name": "s_nm3", "gpu": "h100-96",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3",
                                  model="quantum_stacked")},
            {"run_name": "q_b", "gpu": "a100-40",
             "args": _source_args("q_b", sweep_dir="src", epochs="10")},
        ]

    def test_cartesian_over_lcu_and_lcu_poly_only(self, tmp_path):
        m = bca.build_manifest(self._sources(), sweep_dir=tmp_path / "coeff_ablation_TS")
        assert m["n_runs"] == 6  # 3 configs × {lcu, lcu_poly}
        names = {r["run_name"] for r in m["runs"]}
        assert all(n.endswith("__calcu") or n.endswith("__calcu_poly") for n in names)
        assert not any(n.endswith("__canone") for n in names)

    def test_dense_reindex_and_contiguous_gpu_groups(self, tmp_path):
        m = bca.build_manifest(self._sources(), sweep_dir=tmp_path / "coeff_ablation_TS")
        assert [r["index"] for r in m["runs"]] == list(range(6))
        gpus = [r["gpu"] for r in m["runs"]]
        assert gpus == ["a100-40"] * 4 + ["h100-96"] * 2
        assert m["slurm_groups"]["a100-40"] == [0, 3]
        assert m["slurm_groups"]["h100-96"] == [4, 5]

    def test_epochs_normalised_to_ten_everywhere(self, tmp_path):
        m = bca.build_manifest(self._sources(), sweep_dir=tmp_path / "coeff_ablation_TS")
        for r in m["runs"]:
            assert r["args"][r["args"].index("--epochs") + 1] == "10"

    def test_no_resume_anywhere(self, tmp_path):
        srcs = self._sources()
        srcs[0]["args"] = _source_args("q_a", sweep_dir="src", resume=True)
        m = bca.build_manifest(srcs, sweep_dir=tmp_path / "coeff_ablation_TS")
        for r in m["runs"]:
            assert "--resume" not in r["args"]

    def test_each_run_records_its_arm(self, tmp_path):
        m = bca.build_manifest(self._sources(), sweep_dir=tmp_path / "coeff_ablation_TS")
        arms = {r["coeff_ablation"] for r in m["runs"]}
        assert arms == {"lcu", "lcu_poly"}

    def test_manifest_carries_provenance_and_n_runs(self, tmp_path):
        m = bca.build_manifest(self._sources(), sweep_dir=tmp_path / "coeff_ablation_TS")
        assert m["n_runs"] == len(m["runs"]) == 6
        assert isinstance(m["invocations"], list) and m["invocations"]

    def test_model_prefix_disambiguates_same_named_cross_model_configs(self, tmp_path):
        sources = [
            {"run_name": "manual__cw1", "gpu": "a100-40",
             "args": _source_args("manual__cw1", sweep_dir="src")},
            {"run_name": "manual__cw1", "gpu": "a100-40",
             "args": _source_args("manual__cw1", sweep_dir="src",
                                  model="quantum_shared")},
        ]
        m = bca.build_manifest(sources, sweep_dir=tmp_path / "coeff_ablation_TS")
        names = {r["run_name"] for r in m["runs"]}
        assert len(names) == m["n_runs"] == 4
        assert "quantum__manual__cw1__calcu" in names
        assert "quantum_shared__manual__cw1__calcu" in names

    def test_true_duplicate_selection_raises(self, tmp_path):
        dup = {"run_name": "manual__cw1", "gpu": "a100-40",
               "args": _source_args("manual__cw1", sweep_dir="src")}
        with pytest.raises(ValueError, match="duplicate run name"):
            bca.build_manifest([dup, dict(dup)], sweep_dir=tmp_path / "coeff_ablation_TS")


class TestSbatchCommands:
    def _manifest(self, tmp_path):
        sources = [
            {"run_name": "q_a", "gpu": "a100-40",
             "args": _source_args("q_a", sweep_dir="src")},
            {"run_name": "s_nm3", "gpu": "h100-96",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3")},
        ]
        return bca.build_manifest(sources, sweep_dir=tmp_path / "coeff_ablation_TS")

    def test_one_command_per_gpu_group_with_ranges_gres_time(self, tmp_path):
        m = self._manifest(tmp_path)
        cmds = bca.sbatch_commands(
            m, Path("results/sweeps/coeff_ablation_TS/sweep_manifest.json"))
        assert len(cmds) == 2
        a100 = next(c for c in cmds if "a100-40" in c)
        h100 = next(c for c in cmds if "h100-96" in c)
        assert "--array=0-1" in a100
        assert "--time=12:00:00" in a100
        assert "--gres=gpu:a100-40:1" in a100
        assert "--array=2-3" in h100
        assert "--gres=gpu:h100-96:1" in h100
        assert "scripts/run_sweep.sh" in a100
