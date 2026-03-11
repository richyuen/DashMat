from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(module_name: str, rel_path: str):
    root = Path(__file__).resolve().parents[2]
    path = root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_seed_dataset_is_deterministic():
    harness = _load_module("result_flow_harness_test", "tools/playwright/result_flow_harness.py")

    df1 = harness.build_seed_dataset()
    df2 = harness.build_seed_dataset()

    assert list(df1.columns) == ["SPX_TRIndex", "R2000_TRIndex", "EAFE_TRIndex", "BCTBill13_TRIndex"]
    assert df1.equals(df2)
    assert len(df1) == 756
    meta = harness.get_seed_dataset_metadata(df1)
    assert meta["rows"] == 756
    assert meta["periodicity"] == "daily"


def test_build_portopt_seed_result_inline_shape():
    harness = _load_module("result_flow_harness_test_po", "tools/playwright/result_flow_harness.py")
    raw_df = harness.build_seed_dataset(periods=20)

    name, entry, selected = harness.build_portopt_seed_result(raw_df, session_id=None, use_artifacts=False)

    assert name == "Seeded Risk Parity"
    assert selected == ["SPX_TRIndex", "R2000_TRIndex", "EAFE_TRIndex"]
    assert "returns_json" in entry
    assert "returns_key" not in entry
    assert entry["config"]["model"] == "risk_parity"
    assert entry["frontier_cache"]["0"]["MV"]["frontier_points"]


def test_build_regression_seed_result_inline_shape():
    harness = _load_module("result_flow_harness_test_reg", "tools/playwright/result_flow_harness.py")
    raw_df = harness.build_seed_dataset(periods=20)

    result_name, entry, dep_var, indep_vars = harness.build_regression_seed_result(raw_df, session_id=None, use_artifacts=False)

    assert result_name == "Seeded OLS"
    assert dep_var == "SPX_TRIndex"
    assert indep_vars == ["R2000_TRIndex", "EAFE_TRIndex"]
    assert "predicted_json" in entry
    assert "residuals_json" in entry
    assert "predicted_key" not in entry
    assert entry["window_results"][0]["anova_table"]["F_stat"] == 33.2


def test_run_mode_storage_seeds_exclude_results_and_set_run_flags():
    harness = _load_module("result_flow_harness_test_run_seed", "tools/playwright/result_flow_harness.py")
    seed = harness.build_seed_payloads(Path("C:/Git/DashMat"), "session-123")

    po_seed = harness.make_portopt_storage_seed(seed, include_results=False)
    reg_seed = harness.make_regression_storage_seed(seed, include_results=False)

    assert po_seed["po-results-store"] == {}
    assert po_seed["po-restore-complete-store"] is True
    assert po_seed["dashmat-pending-new-series-store"] == {}
    assert reg_seed["reg-results-store"] == {}
    assert reg_seed["reg-model-store"] == "ols"
    assert reg_seed["reg-window-type-store"] == "full"


def test_summarize_numeric_runs_and_storage_medians():
    harness = _load_module("result_flow_harness_test_summary", "tools/playwright/result_flow_harness.py")

    summary = harness.summarize_numeric_runs(
        [
            {"pageReadyMs": 100, "returnsOpenMs": 140, "storageBytes": {"po-results-store": 200}},
            {"pageReadyMs": 120, "returnsOpenMs": 160, "storageBytes": {"po-results-store": 300}},
            {"pageReadyMs": 110, "returnsOpenMs": 150, "storageBytes": {"po-results-store": 250}},
        ]
    )

    assert summary["runs"] == 3
    assert summary["pageReadyMedian"] == 110
    assert summary["returnsOpenMedian"] == 150
    assert summary["storageBytesMedian"]["po-results-store"] == 250


def test_collect_artifact_store_metrics_uses_configured_root(monkeypatch, tmp_path):
    harness = _load_module("result_flow_harness_test_artifacts", "tools/playwright/result_flow_harness.py")
    artifact_root = tmp_path / "custom_artifacts"
    artifact_root.mkdir()
    payload = artifact_root / "x.bin"
    payload.write_bytes(b"12345")
    monkeypatch.setenv("DASHMAT_ARTIFACT_ROOT", str(artifact_root))

    metrics = harness.collect_artifact_store_metrics(tmp_path)

    assert metrics["exists"] is True
    assert Path(metrics["path"]) == artifact_root
    assert metrics["fileCount"] == 1
    assert metrics["totalBytes"] == 5


def test_normalize_text_trims_and_defaults_empty():
    harness = _load_module("result_flow_harness_test_text", "tools/playwright/result_flow_harness.py")

    assert harness._normalize_text("  hello  ") == "hello"
    assert harness._normalize_text(None) == ""


def test_result_flow_ab_builds_comparison_and_confounded_flag():
    ab = _load_module("result_flow_ab_test", "tools/playwright/result_flow_ab.py")
    args = SimpleNamespace(label="bench", mode="consume_only", runs=3)
    baseline = {
        "portopt": {
            "pageReadyMedian": 100,
            "initialDefaultContentReadyMedian": 130,
            "returnsOpenMedian": 150,
            "statisticsOpenMedian": 160,
            "rollingOpenMedian": 170,
            "calendarOpenMedian": 180,
            "growthOpenMedian": 190,
            "drawdownOpenMedian": 200,
            "frontierOpenMedian": 210,
            "revisitReadyMedian": 120,
            "defaultContentReadyMedian": 140,
            "storageBytesMedian": {"po-results-store": 1000, "dashmat-saved-series-cache-store": 1000},
        },
        "regression": {
            "pageReadyMedian": 110,
            "initialDefaultContentReadyMedian": 140,
            "returnsOpenMedian": 160,
            "statisticsOpenMedian": 170,
            "rollingReturnsOpenMedian": 180,
            "calendarOpenMedian": 190,
            "growthOpenMedian": 200,
            "drawdownOpenMedian": 210,
            "scatterOpenMedian": 220,
            "anovaOpenMedian": 230,
            "revisitReadyMedian": 130,
            "defaultContentReadyMedian": 150,
            "storageBytesMedian": {"reg-results-store": 900},
        },
    }
    variant = {
        "portopt": {
            **baseline["portopt"],
            "pageReadyMedian": 200,
            "revisitReadyMedian": 210,
            "storageBytesMedian": {"po-results-store": 500, "dashmat-saved-series-cache-store": 80000},
        },
        "regression": {
            **baseline["regression"],
            "pageReadyMedian": 210,
            "revisitReadyMedian": 220,
            "storageBytesMedian": {"reg-results-store": 400},
        },
    }

    payload = ab.build_comparison_payload(baseline, variant, args)

    assert payload["portopt"]["pageReadyMedian"]["absoluteDelta"] == 100.0
    assert payload["storage"]["po-results-store"]["absoluteDelta"] == -500.0
    assert payload["confounded"] is True
