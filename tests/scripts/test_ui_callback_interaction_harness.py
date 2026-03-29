from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import tools.playwright.ui_callback_interaction_harness as harness


def test_filter_targeted_requests_keeps_only_matching_outputs():
    summary = {
        "dashUpdateRequests": [
            {"outputs": ["po-portfolio-name-input.value"], "durationMs": 11, "serverMs": 4, "requestBytes": 10, "responseBytes": 20},
            {"outputs": ["po-date-range-store.data"], "durationMs": 7, "serverMs": 3, "requestBytes": 5, "responseBytes": 9},
            {"outputs": ["other.output"], "durationMs": 2, "serverMs": None, "requestBytes": 3, "responseBytes": 4},
        ]
    }

    result = harness.filter_targeted_requests(summary, ["po-date-range-store.data", "po-portfolio-name-input.value"])

    assert result["targetedDashUpdateRequestCount"] == 2
    assert result["targetedDashUpdateTotalMs"] == 18
    assert result["targetedDashUpdateSummedServerMs"] == 7
    assert result["targetedDashUpdateRequestBytes"] == 15
    assert result["targetedDashUpdateResponseBytes"] == 29
    assert result["targetedDashUpdateCallbacks"] == [
        "po-portfolio-name-input.value",
        "po-date-range-store.data",
    ]


def test_summarize_run_group_uses_medians():
    result = harness.summarize_run_group(
        [
            {
                "flowMs": 100,
                "perfTarget": True,
                "scenarioClass": "ui_only",
                "targetedDashUpdateRequestCount": 2,
                "dashUpdateRequestCount": 4,
                "dashUpdateRequestBytes": 40,
                "dashUpdateResponseBytes": 80,
                "dashUpdateSummedServerMs": 10,
            },
            {
                "flowMs": 140,
                "perfTarget": False,
                "scenarioClass": "ui_only",
                "targetedDashUpdateRequestCount": 0,
                "dashUpdateRequestCount": 2,
                "dashUpdateRequestBytes": 20,
                "dashUpdateResponseBytes": 40,
                "dashUpdateSummedServerMs": 6,
            },
            {
                "flowMs": 120,
                "perfTarget": True,
                "scenarioClass": "ui_only",
                "targetedDashUpdateRequestCount": 1,
                "dashUpdateRequestCount": 3,
                "dashUpdateRequestBytes": 30,
                "dashUpdateResponseBytes": 60,
                "dashUpdateSummedServerMs": 8,
            },
        ]
    )

    assert result == {
        "runs": 3,
        "perfTarget": True,
        "scenarioClass": "ui_only",
        "flowMedian": 120,
        "targetedDashUpdateRequestCountMedian": 1,
        "dashUpdateRequestCountMedian": 3,
        "dashUpdateRequestBytesMedian": 30,
        "dashUpdateResponseBytesMedian": 60,
        "dashUpdateSummedServerMsMedian": 8,
    }


def test_build_seeded_regression_results_returns_two_valid_entries():
    results = harness.build_seeded_regression_results(["SPX_TRIndex", "NDX_TRIndex", "RTY_TRIndex"])

    assert list(results.keys()) == ["Harness Result 1", "Harness Result 2"]
    first = results["Harness Result 1"]
    assert first["dependent_var"]
    assert first["independent_vars"]
    assert first["window_results"]
    assert "predicted_json" in first
    assert "residuals_json" in first
    assert "display_json" in first
    assert first["display_columns"] == ["Predicted", "Actual (Y)", "NDX_TRIndex", "RTY_TRIndex", "Residual"]


def test_wait_for_quiet_window_waits_for_requests_to_settle():
    class _FakePage:
        def __init__(self):
            self._released = False

        def wait_for_timeout(self, ms):
            if not self._released:
                tracker.active_requests = {}
                tracker.records.append({"id": len(tracker.records) + 1})
                self._released = True

    class _FakeTracker:
        def __init__(self):
            self.active_requests = {"req": object()}
            self.records = []

    tracker = _FakeTracker()
    harness.wait_for_quiet_window(_FakePage(), tracker, quiet_ms=0, timeout_ms=200)
    assert len(tracker.records) == 1


def test_apply_network_profile_uses_cdp_emulation():
    class _FakeSession:
        def __init__(self):
            self.calls = []

        def send(self, method, params=None):
            self.calls.append((method, params))

    class _FakeContext:
        def __init__(self, session):
            self._session = session

        def new_cdp_session(self, page):
            assert page is fake_page
            return self._session

    class _FakePage:
        pass

    session = _FakeSession()
    fake_page = _FakePage()
    fake_page.context = _FakeContext(session)

    applied = harness._apply_network_profile(fake_page, "office-wan")

    assert applied["name"] == "office-wan"
    assert ("Network.enable", None) in session.calls
    emulate_call = next(call for call in session.calls if call[0] == "Network.emulateNetworkConditions")
    assert emulate_call[1]["latency"] == 40
    assert emulate_call[1]["downloadThroughput"] > 0
    assert emulate_call[1]["uploadThroughput"] > 0


def _tmp_repo_root() -> Path:
    root = Path("tests/.tmp") / f"ui_callback_harness_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_main_writes_result_file(monkeypatch, capsys):
    repo_root = _tmp_repo_root()
    monkeypatch.setattr(harness, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        harness,
        "run_page_suite",
        lambda page_name, base_url, db_series, headless, runs, network_profile: {
            "page": page_name,
            "runs": runs,
            "networkProfile": {"name": network_profile},
            "scenarios": {"scenario": {"summary": {"runs": 1, "scenarioClass": "ui_only"}, "runs": []}},
        },
    )

    exit_code = harness.main(["--pages", "portopt", "--runs", "1", "--label", "unit-test", "--network-profile", "office-wan"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "RESULT_PATH=" in out
    result_path = Path(out.strip().split("RESULT_PATH=", 1)[1])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["label"] == "unit-test"
    assert payload["networkProfile"] == "office-wan"
    assert payload["pages"]["portopt"]["page"] == "portopt"
    assert payload["pages"]["portopt"]["networkProfile"]["name"] == "office-wan"


def test_main_returns_nonzero_on_failure(monkeypatch, capsys):
    repo_root = _tmp_repo_root()
    monkeypatch.setattr(harness, "REPO_ROOT", repo_root)
    monkeypatch.setattr(harness, "run_page_suite", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    exit_code = harness.main(["--pages", "regression"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "FAIL: boom" in out
    assert "TRACEBACK_PATH=" in out


def test_run_regression_scenarios_includes_visible_result_select_guards(monkeypatch):
    captured = []

    def _fake_measure_scenario(**kwargs):
        captured.append((kwargs["scenario_name"], kwargs["scenario_class"], kwargs["targeted_outputs"]))
        return {"scenario": kwargs["scenario_name"]}

    monkeypatch.setattr(harness, "measure_scenario", _fake_measure_scenario)

    scenarios = harness.run_regression_scenarios(page=object(), tracker=object(), db_series=["SPX_TRIndex", "NDX_TRIndex", "RTY_TRIndex"])

    assert len(scenarios) == len(captured)
    assert (
        "regression_result_select_visible_anova",
        "visible_result_tab",
        ["reg-anova-content.children"],
    ) in captured
    assert (
        "regression_result_select_visible_scatter",
        "visible_result_tab",
        ["reg-scatter-content.children"],
    ) in captured


def test_run_portopt_scenarios_includes_visible_benchmark_baselines(monkeypatch):
    captured = []

    def _fake_measure_scenario(**kwargs):
        captured.append((kwargs["scenario_name"], kwargs.get("scenario_class", "ui_only"), kwargs["targeted_outputs"]))
        return {"scenario": kwargs["scenario_name"]}

    monkeypatch.setattr(harness, "measure_scenario", _fake_measure_scenario)

    scenarios = harness.run_portopt_scenarios(page=object(), tracker=object())

    assert len(scenarios) == len(captured)
    assert (
        "portopt_statistics_portfolio_switch_visible",
        "visible_result_tab",
        ["po-statistics-grid-content.children"],
    ) in captured
    assert (
        "portopt_weight_portfolio_switch_visible",
        "visible_result_tab",
        ["po-weight-chart-graph.figure", "po-weight-chart-content.children"],
    ) in captured
    assert (
        "portopt_weight_table_portfolio_switch_visible",
        "visible_result_tab",
        ["po-weight-grid-content.children"],
    ) in captured
    assert (
        "portopt_growth_portfolio_switch_visible",
        "visible_result_tab",
        ["po-growth-chart-container.children"],
    ) in captured
    assert (
        "portopt_turnover_portfolio_switch_visible",
        "visible_result_tab",
        ["po-turnover-chart-container.children"],
    ) in captured
    assert (
        "portopt_turnover_table_portfolio_switch_visible",
        "visible_result_tab",
        ["po-turnover-grid-container.children"],
    ) in captured
    assert (
        "portopt_frontier_portfolio_switch_visible",
        "visible_result_tab",
        ["po-frontier-chart-graph.figure", "po-frontier-chart-container.children"],
    ) in captured
    assert (
        "portopt_frontier_table_portfolio_switch_visible",
        "visible_result_tab",
        ["po-frontier-grid-container.children"],
    ) in captured
    assert (
        "portopt_risk_portfolio_switch_visible",
        "visible_result_tab",
        ["po-risk-chart-container.children"],
    ) in captured
    assert (
        "portopt_risk_table_portfolio_switch_visible",
        "visible_result_tab",
        ["po-risk-grid-container.children"],
    ) in captured
    assert (
        "portopt_attribution_portfolio_switch_visible",
        "visible_result_tab",
        ["po-attribution-chart-container.children"],
    ) in captured
    assert (
        "portopt_attribution_table_portfolio_switch_visible",
        "visible_result_tab",
        ["po-attribution-grid-container.children"],
    ) in captured
    assert (
        "portopt_statistics_visible",
        "visible_result_tab",
        ["po-statistics-grid-content.children"],
    ) in captured
    assert (
        "portopt_rolling_metric_visible",
        "visible_result_tab",
        ["po-rolling-content.children"],
    ) in captured
    assert (
        "portopt_rolling_window_visible",
        "visible_result_tab",
        ["po-rolling-content.children"],
    ) in captured
    assert (
        "portopt_rolling_return_type_visible",
        "visible_result_tab",
        ["po-rolling-content.children"],
    ) in captured
    assert (
        "portopt_returns_visible",
        "visible_result_tab",
        ["po-returns-grid-content.children"],
    ) in captured
    assert (
        "portopt_returns_portfolio_switch_visible",
        "visible_result_tab",
        ["po-returns-grid-content.children"],
    ) in captured
    assert (
        "portopt_calendar_visible",
        "visible_result_tab",
        ["po-calendar-content.children"],
    ) in captured
    assert (
        "portopt_calendar_portfolio_switch_visible",
        "visible_result_tab",
        ["po-calendar-content.children"],
    ) in captured
    assert (
        "portopt_calendar_monthly_portfolio_switch_visible",
        "visible_result_tab",
        ["po-calendar-content.children"],
    ) in captured
    assert (
        "portopt_drawdown_visible",
        "visible_result_tab",
        ["po-drawdown-content.children"],
    ) in captured
    assert (
        "portopt_drawdown_portfolio_switch_visible",
        "visible_result_tab",
        ["po-drawdown-content.children"],
    ) in captured
