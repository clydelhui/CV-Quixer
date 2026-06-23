"""Tests for the PE-ablation sweep builder (experiments/build_pe_ablation.py).

The builder takes the 16 curated epoch-extension configs (run_name + source
sweep + target GPU, from results/extended_runs_25ep.txt) and fans each over the
{none, 1d} positional-encoding variants into one fresh-from-scratch 32-run sweep
manifest. The 2d arm is deliberately NOT generated — it is reused from the
existing high_epoch_* runs and compared manually (see the Phase B handoff).

Load-bearing invariants pinned here:

  * each source argv is replayed verbatim except: --positional-encoding injected,
    --run-name gets a __pe<v> marker, --runs-root repointed at the new sweep dir,
    --epochs normalised to 10 (stacked sources carry --epochs 3), --resume dropped;
  * only none/1d variants are emitted (no 2d arm);
  * runs are ordered so each GPU group is a contiguous index range, dense 0..N-1;
  * the printed sbatch commands carry the right --gres / --time / --array range.
"""

import json
import sys
from pathlib import Path

import pytest

# build_pe_ablation lives in experiments/ (not a package) — import via sys.path
# like the other experiment-script tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import build_pe_ablation as bpa


def _source_args(run_name, *, sweep_dir, model=None, epochs="10", resume=False):
    """A representative full_experiment.py argv as stored in a source manifest."""
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
    """Synthetic source sweep dir + sweep_manifest.json (cf. test_poly_init_noise)."""
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


class TestParseSourceRunsFile:
    def test_parses_run_sweep_gpu_columns_skipping_comments(self, tmp_path):
        listing = tmp_path / "extended_runs.txt"
        listing.write_text(
            "# header comment\n"
            "\n"
            "# === section ===\n"
            "run_a   results/sweeps/high_epoch_quantum_x   a100-40   19840  9.5  0.885\n"
            "run_b   results/sweeps/high_epoch_stacked_y   h100-96   89163  2.4  0.871\n"
        )
        rows = bpa.parse_source_runs_file(listing)
        assert [r["run_name"] for r in rows] == ["run_a", "run_b"]
        assert rows[0]["sweep_dir"] == "results/sweeps/high_epoch_quantum_x"
        assert rows[0]["gpu"] == "a100-40"
        assert rows[1]["gpu"] == "h100-96"


class TestResolveSourceRuns:
    def test_pulls_verbatim_args_from_source_manifest(self, tmp_path):
        sweep = _make_source_sweep(tmp_path, "high_epoch_quantum_x", [
            {"run_name": "run_a", "args": _source_args("run_a", sweep_dir="src")},
            {"run_name": "run_b", "args": _source_args("run_b", sweep_dir="src")},
        ])
        rows = [{"run_name": "run_b", "sweep_dir": str(sweep), "gpu": "a100-40"}]
        resolved = bpa.resolve_source_runs(rows)
        assert len(resolved) == 1
        assert resolved[0]["run_name"] == "run_b"
        assert resolved[0]["gpu"] == "a100-40"
        assert resolved[0]["model"] == "quantum"  # no --model in argv → default
        assert resolved[0]["args"][resolved[0]["args"].index("--run-name") + 1] == "run_b"

    def test_model_read_from_args(self, tmp_path):
        sweep = _make_source_sweep(tmp_path, "high_epoch_shared_x", [
            {"run_name": "s_a", "args": _source_args("s_a", sweep_dir="src",
                                                     model="quantum_shared")},
        ])
        rows = [{"run_name": "s_a", "sweep_dir": str(sweep), "gpu": "a100-40"}]
        assert bpa.resolve_source_runs(rows)[0]["model"] == "quantum_shared"

    def test_missing_run_in_manifest_raises(self, tmp_path):
        sweep = _make_source_sweep(tmp_path, "high_epoch_quantum_x", [
            {"run_name": "run_a", "args": _source_args("run_a", sweep_dir="src")},
        ])
        rows = [{"run_name": "ghost", "sweep_dir": str(sweep), "gpu": "a100-40"}]
        with pytest.raises(KeyError):
            bpa.resolve_source_runs(rows)


class TestRewriteRunArgs:
    def _base(self):
        return _source_args("manual__cw1", sweep_dir="results/sweeps/old", epochs="3",
                            model="quantum_stacked", resume=True)

    def test_injects_pe_marks_name_repoints_root_normalises_epochs_drops_resume(self):
        out = bpa.rewrite_run_args(
            self._base(), "1d", model="quantum_stacked",
            runs_root="results/sweeps/pe_ablation_TS", target_epochs=10)
        assert out[out.index("--positional-encoding") + 1] == "1d"
        # run-name carries the model prefix (disambiguates cross-model collisions).
        assert out[out.index("--run-name") + 1] == "quantum_stacked__manual__cw1__pe1d"
        assert out[out.index("--runs-root") + 1] == "results/sweeps/pe_ablation_TS"
        assert out[out.index("--epochs") + 1] == "10"
        assert "--resume" not in out
        # model + arch flags replayed verbatim
        assert out[out.index("--model") + 1] == "quantum_stacked"
        assert out[out.index("--num-modes") + 1] == "2"

    def test_none_variant_marker(self):
        out = bpa.rewrite_run_args(
            self._base(), "none", model="quantum_stacked", runs_root="r",
            target_epochs=10)
        assert out[out.index("--run-name") + 1] == "quantum_stacked__manual__cw1__penone"
        assert out[out.index("--positional-encoding") + 1] == "none"


class TestBuildManifest:
    def _sources(self):
        # Two a100 runs + one h100 run, deliberately out of GPU order on input.
        return [
            {"run_name": "q_a", "gpu": "a100-40",
             "args": _source_args("q_a", sweep_dir="src", epochs="10")},
            {"run_name": "s_nm3", "gpu": "h100-96",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3",
                                  model="quantum_stacked")},
            {"run_name": "q_b", "gpu": "a100-40",
             "args": _source_args("q_b", sweep_dir="src", epochs="10")},
        ]

    def test_cartesian_over_none_and_1d_only(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        m = bpa.build_manifest(self._sources(), sweep_dir=sweep_dir)
        # 3 configs × {none, 1d} = 6 runs; no 2d arm.
        assert m["n_runs"] == 6
        names = {r["run_name"] for r in m["runs"]}
        assert all(n.endswith("__penone") or n.endswith("__pe1d") for n in names)
        assert not any(n.endswith("__pe2d") for n in names)

    def test_dense_reindex_and_contiguous_gpu_groups(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        m = bpa.build_manifest(self._sources(), sweep_dir=sweep_dir)
        assert [r["index"] for r in m["runs"]] == list(range(6))
        # a100 group must come first as a contiguous block, then h100.
        gpus = [r["gpu"] for r in m["runs"]]
        assert gpus == ["a100-40"] * 4 + ["h100-96"] * 2
        groups = m["slurm_groups"]
        assert groups["a100-40"] == [0, 3]
        assert groups["h100-96"] == [4, 5]

    def test_epochs_normalised_to_ten_everywhere(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        m = bpa.build_manifest(self._sources(), sweep_dir=sweep_dir)
        for r in m["runs"]:
            assert r["args"][r["args"].index("--epochs") + 1] == "10"

    def test_runs_root_repointed_at_sweep_dir(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        m = bpa.build_manifest(self._sources(), sweep_dir=sweep_dir)
        for r in m["runs"]:
            assert r["args"][r["args"].index("--runs-root") + 1] == str(sweep_dir)

    def test_no_resume_anywhere(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        srcs = self._sources()
        srcs[0]["args"] = _source_args("q_a", sweep_dir="src", resume=True)
        m = bpa.build_manifest(srcs, sweep_dir=sweep_dir)
        for r in m["runs"]:
            assert "--resume" not in r["args"]

    def test_manifest_carries_provenance_and_n_runs(self, tmp_path):
        sweep_dir = tmp_path / "pe_ablation_TS"
        m = bpa.build_manifest(self._sources(), sweep_dir=sweep_dir)
        assert m["n_runs"] == len(m["runs"]) == 6
        assert isinstance(m["invocations"], list) and m["invocations"]

    def test_model_prefix_disambiguates_same_named_cross_model_configs(self, tmp_path):
        # quantum + shared sweeps share manual run-name strings (model isn't in the
        # name marker). Without the model prefix these would collide into one dir.
        sources = [
            {"run_name": "manual__cw1", "gpu": "a100-40",
             "args": _source_args("manual__cw1", sweep_dir="src")},  # quantum
            {"run_name": "manual__cw1", "gpu": "a100-40",
             "args": _source_args("manual__cw1", sweep_dir="src",
                                  model="quantum_shared")},
        ]
        m = bpa.build_manifest(sources, sweep_dir=tmp_path / "pe_ablation_TS")
        names = {r["run_name"] for r in m["runs"]}
        assert len(names) == m["n_runs"] == 4, "no run-dir collisions"
        assert "quantum__manual__cw1__penone" in names
        assert "quantum_shared__manual__cw1__penone" in names

    def test_true_duplicate_selection_raises(self, tmp_path):
        # Same (run_name, model) listed twice → the prefix can't separate them.
        dup = {"run_name": "manual__cw1", "gpu": "a100-40",
               "args": _source_args("manual__cw1", sweep_dir="src")}
        with pytest.raises(ValueError, match="duplicate run name"):
            bpa.build_manifest([dup, dict(dup)], sweep_dir=tmp_path / "pe_ablation_TS")


class TestSelectExtensionRunsSchema:
    """The selection list is shared by select_extension_runs.py (writer) and
    build_pe_ablation.py (reader): column 3 must be the GPU in both."""

    def test_gpu_threshold_matches_hand_map(self):
        import select_extension_runs as ser
        assert ser.gpu_for_peak_mb(31066) == "a100-40"   # heaviest a100 run
        assert ser.gpu_for_peak_mb(86153) == "h100-96"    # lightest h100 run

    def test_writer_line_round_trips_through_reader(self, tmp_path):
        # A line in exactly the form select_extension_runs.py --write emits:
        #   run_name  sweep_dir  gpu  peak_MB  why  best  last3  slope
        import select_extension_runs as ser
        gpu = ser.gpu_for_peak_mb(89163)
        line = (f"s_nm3  results/sweeps/high_epoch_stacked_x  {gpu}  89163  "
                f"best+slope  0.871  0.869  +0.0030\n")
        listing = tmp_path / "extended_runs.txt"
        listing.write_text("# header\n" + line)
        rows = bpa.parse_source_runs_file(listing)
        assert rows[0]["run_name"] == "s_nm3"
        assert rows[0]["sweep_dir"] == "results/sweeps/high_epoch_stacked_x"
        assert rows[0]["gpu"] == "h100-96"  # column 3 read as GPU, not the why-tag


class TestSbatchCommands:
    def _manifest(self, tmp_path):
        sources = [
            {"run_name": "q_a", "gpu": "a100-40",
             "args": _source_args("q_a", sweep_dir="src")},
            {"run_name": "s_nm3", "gpu": "h100-96",
             "args": _source_args("s_nm3", sweep_dir="src", epochs="3")},
        ]
        return bpa.build_manifest(sources, sweep_dir=tmp_path / "pe_ablation_TS")

    def test_one_command_per_gpu_group_with_ranges_gres_time(self, tmp_path):
        m = self._manifest(tmp_path)
        cmds = bpa.sbatch_commands(m, Path("results/sweeps/pe_ablation_TS/sweep_manifest.json"))
        assert len(cmds) == 2
        a100 = next(c for c in cmds if "a100-40" in c)
        h100 = next(c for c in cmds if "h100-96" in c)
        assert "--array=0-1" in a100
        assert "--time=12:00:00" in a100
        assert "--gres=gpu:a100-40:1" in a100
        assert "--array=2-3" in h100
        assert "--gres=gpu:h100-96:1" in h100
        assert "scripts/run_sweep.sh" in a100
