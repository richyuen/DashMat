(function () {
  window.dash_clientside = window.dash_clientside || {};
  function noUpdate() {
    return window.dash_clientside.no_update;
  }
  const flexStyle = { display: "flex", flexDirection: "column", flex: "1", overflow: "hidden" };
  const flexScrollStyle = { display: "flex", flexDirection: "column", flex: "1", overflow: "auto" };
  const hiddenStyle = { display: "none" };

  function triggeredId() {
    const ctx = window.dash_clientside.callback_context;
    const triggered = (ctx && ctx.triggered) ? ctx.triggered : [];
    if (!triggered.length || !triggered[0] || !triggered[0].prop_id) {
      return null;
    }
    return triggered[0].prop_id.split(".")[0];
  }

  function clickUploadInput(rootId) {
    setTimeout(function () {
      const uploadDiv = document.getElementById(rootId);
      if (!uploadDiv) {
        return;
      }
      const input = uploadDiv.querySelector('input[type="file"]');
      if (input) {
        input.click();
      }
    }, 100);
  }

  const workspacePrefixes = ["dashmat-", "at-", "po-", "reg-"];
  const workspaceExtraKeys = ["dashmat-bctbill13-cache-store"];
  const workspaceExcludedKeys = ["userinfo"];

  function isWorkspaceSessionKey(key) {
    if (!key || workspaceExcludedKeys.indexOf(key) !== -1) {
      return false;
    }
    if (workspaceExtraKeys.indexOf(key) !== -1) {
      return true;
    }
    for (let i = 0; i < workspacePrefixes.length; i += 1) {
      if (key.indexOf(workspacePrefixes[i]) === 0) {
        return true;
      }
    }
    return false;
  }

  function collectWorkspaceSessionData() {
    const data = {};
    for (let i = 0; i < sessionStorage.length; i += 1) {
      const key = sessionStorage.key(i);
      if (!isWorkspaceSessionKey(key)) {
        continue;
      }
      data[key] = sessionStorage.getItem(key);
    }
    return data;
  }

  function clearWorkspaceSessionKeys() {
    const keysToRemove = [];
    for (let i = 0; i < sessionStorage.length; i += 1) {
      const key = sessionStorage.key(i);
      if (isWorkspaceSessionKey(key)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(function (key) {
      sessionStorage.removeItem(key);
    });
  }

  function triggerUploadWithCancel(rootId, blockerStoreId) {
    setTimeout(function () {
      const uploadDiv = document.getElementById(rootId);
      if (!uploadDiv) {
        return;
      }
      const input = uploadDiv.querySelector('input[type="file"]');
      if (!input) {
        return;
      }
      const onFocus = function () {
        window.removeEventListener("focus", onFocus);
        setTimeout(function () {
          if (!input.files || input.files.length === 0) {
            window.dash_clientside.set_props(blockerStoreId, { data: false });
          }
        }, 500);
      };
      window.addEventListener("focus", onFocus);
      input.click();
    }, 100);
    return true;
  }

  function uiBlockerEnable() {
    const trigger = triggeredId();
    if (!trigger) {
      return noUpdate();
    }
    for (let i = 0; i < arguments.length; i += 1) {
      if (arguments[i]) {
        return true;
      }
    }
    return noUpdate();
  }

  function triggerAnalyticsUpload(menuAddClicks, welcomeAddClicks) {
    if (menuAddClicks || welcomeAddClicks) {
      return triggerUploadWithCancel("at-upload-data", "at-ui-blocker-store");
    }
    return noUpdate();
  }

  function triggerPortoptUpload(menuAddClicks, welcomeAddClicks) {
    if (menuAddClicks || welcomeAddClicks) {
      return triggerUploadWithCancel("po-upload-data", "po-ui-blocker-store");
    }
    return noUpdate();
  }

  function triggerRegressionUpload(menuAddClicks, welcomeAddClicks) {
    if (menuAddClicks || welcomeAddClicks) {
      return triggerUploadWithCancel("reg-upload-data", "reg-ui-blocker-store");
    }
    return noUpdate();
  }

  function uiBlockerRelease(dbOpened, dbErrorHidden, rawOpened, rawErrorHidden, portfolioOpened, portfolioErrorHidden, underlyingOpened, underlyingErrorHidden, seriesSelectionOpened) {
    const trigger = triggeredId() || "";
    if (trigger.indexOf("series-selection-modal") !== -1) {
      return false;
    }
    if (trigger.indexOf("raw-db-add-") !== -1) {
      return (rawOpened === false || rawErrorHidden === false) ? false : noUpdate();
    }
    if (trigger.indexOf("portfolio-add-") !== -1) {
      return (portfolioOpened === false || portfolioErrorHidden === false) ? false : noUpdate();
    }
    if (trigger.indexOf("underlying-add-") !== -1) {
      return (underlyingOpened === false || underlyingErrorHidden === false) ? false : noUpdate();
    }
    if (trigger.indexOf("db-add-") !== -1) {
      return (dbOpened === false || dbErrorHidden === false) ? false : noUpdate();
    }
    return noUpdate();
  }

  function clearWorkspaceSession(n_clicks) {
    if (!n_clicks) {
      return noUpdate();
    }
    clearWorkspaceSessionKeys();
    window.location.reload();
    return noUpdate();
  }

  function saveWorkspaceSession(n_clicks) {
    if (!n_clicks) {
      return noUpdate();
    }
    const data = collectWorkspaceSessionData();
    const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "dashmat_session.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return noUpdate();
  }

  function loadWorkspaceSessionDialog(rootId, n_clicks) {
    if (!n_clicks) {
      return noUpdate();
    }
    clickUploadInput(rootId);
    return noUpdate();
  }

  function loadWorkspaceSession(contents) {
    if (!contents) {
      return noUpdate();
    }
    const raw = atob(contents.split(",")[1]);
    const data = JSON.parse(raw);
    clearWorkspaceSessionKeys();
    Object.keys(data || {}).forEach(function (key) {
      if (!isWorkspaceSessionKey(key)) {
        return;
      }
      sessionStorage.setItem(key, data[key]);
    });
    window.location.reload();
    return noUpdate();
  }

  function navigateAnalytics(menuExit, menuPortfolio, menuRegression, welcomePortfolio, welcomeRegression) {
    const trigger = triggeredId();
    if (!trigger) {
      return noUpdate();
    }
    if (trigger === "at-menu-exit") {
      window.location.href = "/";
      return noUpdate();
    }
    if (trigger === "at-menu-view-portfolio" || trigger === "at-welcome-view-portfolio") {
      window.location.pathname = "/portopt";
      return noUpdate();
    }
    if (trigger === "at-menu-view-regression" || trigger === "at-welcome-view-regression") {
      window.location.pathname = "/regression";
      return noUpdate();
    }
    return noUpdate();
  }

  function navigatePortopt(menuExit, menuAnalytics, menuRegression, welcomeAnalytics, welcomeRegression) {
    const trigger = triggeredId();
    if (!trigger) {
      return noUpdate();
    }
    if (trigger === "po-menu-exit") {
      window.location.href = "/";
      return noUpdate();
    }
    if (trigger === "po-menu-view-analytics" || trigger === "po-welcome-view-analytics") {
      window.location.pathname = "/analyticstool";
      return noUpdate();
    }
    if (trigger === "po-menu-view-regression" || trigger === "po-welcome-view-regression") {
      window.location.pathname = "/regression";
      return noUpdate();
    }
    return noUpdate();
  }

  function navigateRegression(menuExit, menuAnalytics, menuPortfolio, welcomeAnalytics, welcomePortfolio) {
    const trigger = triggeredId();
    if (!trigger) {
      return noUpdate();
    }
    if (trigger === "reg-menu-exit") {
      window.location.href = "/";
      return noUpdate();
    }
    if (trigger === "reg-menu-view-analytics" || trigger === "reg-welcome-view-analytics") {
      window.location.pathname = "/analyticstool";
      return noUpdate();
    }
    if (trigger === "reg-menu-view-portfolio" || trigger === "reg-welcome-view-portfolio") {
      window.location.pathname = "/portopt";
      return noUpdate();
    }
    return noUpdate();
  }

  function syncAnalyticsPeriodicity(rawData, periodicityValue) {
    const trigger = triggeredId();
    if (trigger !== "dashmat-raw-data-store" || !rawData || !periodicityValue) {
      return noUpdate();
    }
    sessionStorage.setItem("po-periodicity-value-store", JSON.stringify(periodicityValue));
    return periodicityValue;
  }

  function syncPortoptPeriodicity(rawData, periodicityValue) {
    const trigger = triggeredId();
    if (trigger !== "dashmat-raw-data-store" || !rawData || !periodicityValue) {
      return noUpdate();
    }
    sessionStorage.setItem("at-periodicity-value-store", JSON.stringify(periodicityValue));
    return periodicityValue;
  }

  function analyticsControlSync(periodicity, returnsType, volScaler, seriesSelect, activeTab, rollingWindow, rollingMetric, rollingReturnType, monthlyView, monthlySeries) {
    return [
      periodicity,
      returnsType,
      volScaler,
      seriesSelect || [],
      activeTab || "statistics",
      rollingWindow || "1y",
      rollingMetric || "total_return",
      rollingReturnType || "annualized",
      monthlyView !== null && monthlyView !== undefined ? monthlyView : "annual",
      monthlySeries
    ];
  }

  function analyticsViewSync(rollingView, drawdownView, growthView) {
    const rolling = rollingView !== null && rollingView !== undefined ? rollingView : "chart";
    const drawdown = drawdownView !== null && drawdownView !== undefined ? drawdownView : "chart";
    const growth = growthView !== null && growthView !== undefined ? growthView : "chart";
    return [
      rolling,
      rolling === "chart" ? hiddenStyle : flexStyle,
      rolling === "chart" ? flexStyle : hiddenStyle,
      drawdown,
      drawdown === "chart" ? hiddenStyle : flexStyle,
      drawdown === "chart" ? flexScrollStyle : hiddenStyle,
      growth,
      growth === "chart" ? hiddenStyle : flexStyle,
      growth === "chart" ? flexScrollStyle : hiddenStyle
    ];
  }

  function analyticsFactorRegimeSync(factorMode, factorQuantiles, factorTransform, factorSeries, regimeDefinition, regimeMethodType) {
    let quantiles = 5;
    if (factorQuantiles !== null && factorQuantiles !== undefined) {
      const parsed = parseInt(factorQuantiles, 10);
      if (Number.isFinite(parsed)) {
        quantiles = Math.min(20, Math.max(2, parsed));
      }
    }
    const mode = factorMode || "box";
    const method = String(regimeMethodType || "1");
    return [
      mode,
      quantiles,
      factorTransform === "zscore" ? "zscore" : "raw",
      factorSeries,
      mode === "box" ? { display: "block" } : { display: "none" },
      regimeDefinition,
      method === "3" ? { display: "none" } : { display: "block" },
      method === "3" ? { display: "block" } : { display: "none" }
    ];
  }

  function regressionControlSync(periodicity, volScaler, model, name, forceZero, robustSe, expWt, halflife, windowType, windowSize, optStep, optStepUnit, fillInSample, missingData, alpha, l1Ratio, activeTab) {
    return [
      periodicity,
      volScaler !== null && volScaler !== undefined ? volScaler : 0,
      model || "ols",
      name,
      forceZero,
      robustSe,
      expWt,
      halflife !== null && halflife !== undefined ? halflife : 63,
      windowType,
      windowSize !== null && windowSize !== undefined ? windowSize : 36,
      optStep !== null && optStep !== undefined ? optStep : 1,
      optStepUnit,
      fillInSample,
      missingData,
      alpha !== null && alpha !== undefined ? alpha : 1.0,
      l1Ratio !== null && l1Ratio !== undefined ? l1Ratio : 0.5,
      activeTab
    ];
  }

  function portoptControlSync(periodicity, volScaler, activeTab, seriesSelect, fillInSample, optStepUnit, optWindow, windowSize, optStep, optModel, portfolioName, expWtCov, halflife, covShrinkage, covShrinkageTarget, missingData, objective, blTau, exAnteMode) {
    return [
      periodicity,
      volScaler,
      activeTab || "weight",
      seriesSelect || [],
      fillInSample,
      optStepUnit,
      optWindow,
      windowSize,
      optStep,
      optModel || "risk_parity",
      portfolioName,
      !!expWtCov,
      halflife,
      covShrinkage || "none",
      covShrinkageTarget || "scaled_identity",
      missingData,
      objective,
      blTau,
      exAnteMode || "ret_cov"
    ];
  }

  function portoptViewSync(weightView, attributionView, riskView, turnoverView, frontierView) {
    function renderState(view) {
      const normalized = view !== null && view !== undefined ? view : "chart";
      return [
        normalized,
        normalized === "chart" ? hiddenStyle : flexStyle,
        normalized === "chart" ? flexScrollStyle : hiddenStyle
      ];
    }
    return []
      .concat(renderState(weightView))
      .concat(renderState(attributionView))
      .concat(renderState(riskView))
      .concat(renderState(turnoverView))
      .concat(renderState(frontierView));
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    dashmat_callbacks: {
      analyticsControlSync: analyticsControlSync,
      analyticsFactorRegimeSync: analyticsFactorRegimeSync,
      analyticsViewSync: analyticsViewSync,
      clearWorkspaceSession: clearWorkspaceSession,
      loadWorkspaceSession: loadWorkspaceSession,
      loadWorkspaceSessionDialog: loadWorkspaceSessionDialog,
      navigateAnalytics: navigateAnalytics,
      navigatePortopt: navigatePortopt,
      navigateRegression: navigateRegression,
      portoptControlSync: portoptControlSync,
      portoptViewSync: portoptViewSync,
      regressionControlSync: regressionControlSync,
      saveWorkspaceSession: saveWorkspaceSession,
      syncAnalyticsPeriodicity: syncAnalyticsPeriodicity,
      syncPortoptPeriodicity: syncPortoptPeriodicity,
      triggerAnalyticsUpload: triggerAnalyticsUpload,
      triggerPortoptUpload: triggerPortoptUpload,
      triggerRegressionUpload: triggerRegressionUpload,
      uiBlockerEnable: uiBlockerEnable,
      uiBlockerRelease: uiBlockerRelease
    }
  });
})();
