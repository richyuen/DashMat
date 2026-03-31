(function () {
  window.dash_clientside = window.dash_clientside || {};
  function noUpdate() {
    return window.dash_clientside.no_update;
  }
  function sleepMs(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, ms);
    });
  }
  const flexStyle = { display: "flex", flexDirection: "column", flex: "1", overflow: "hidden" };
  const flexScrollStyle = { display: "flex", flexDirection: "column", flex: "1", overflow: "auto" };
  const hiddenStyle = { display: "none" };
  const portoptModelDefaultNames = {
    risk_parity: "RP",
    factor_risk_parity: "FRP",
    hierarchical_risk_parity: "HRP",
    hrp: "HRP",
    maximize_sharpe: "MSR",
    minimize_variance: "MinVar",
    minimize_cvar: "MinCVaR",
    equal_weight: "EW",
    ex_ante_mv: "ExAnteMV",
    black_litterman: "BL"
  };
  const regressionModelDefaultNames = {
    ols: "OLS",
    constrained_ols: "Constrained OLS",
    style_analysis: "Style Analysis",
    ridge: "Ridge",
    lasso: "Lasso",
    elastic_net: "Elastic Net"
  };
  const deferredModalOpenHandles = {};

  function cancelDeferredModalOpen(modalId) {
    const pending = deferredModalOpenHandles[modalId];
    if (!pending) {
      return;
    }
    if (pending.type === "idle" && typeof cancelIdleCallback === "function") {
      cancelIdleCallback(pending.handle);
    } else {
      clearTimeout(pending.handle);
    }
    delete deferredModalOpenHandles[modalId];
  }

  function deferModalOpen(modalId) {
    var fn = function () {
      delete deferredModalOpenHandles[modalId];
      window.dash_clientside.set_props(modalId, { opened: true });
    };
    cancelDeferredModalOpen(modalId);
    if (typeof requestIdleCallback === "function") {
      deferredModalOpenHandles[modalId] = {
        type: "idle",
        handle: requestIdleCallback(fn, { timeout: 500 })
      };
    } else {
      deferredModalOpenHandles[modalId] = {
        type: "timeout",
        handle: setTimeout(fn, 300)
      };
    }
  }

  function triggeredId() {
    const ctx = window.dash_clientside.callback_context;
    const triggered = (ctx && ctx.triggered) ? ctx.triggered : [];
    if (!triggered.length || !triggered[0] || !triggered[0].prop_id) {
      return null;
    }
    return triggered[0].prop_id.split(".")[0];
  }

  function normalizePath(pathname) {
    return String(pathname || "").split("?")[0].replace(/\/$/, "") || "/";
  }

  function stableValue(value) {
    if (Array.isArray(value)) {
      return value.map(stableValue);
    }
    if (value && typeof value === "object") {
      const out = {};
      Object.keys(value).sort().forEach(function (key) {
        out[key] = stableValue(value[key]);
      });
      return out;
    }
    return value;
  }

  function sameValue(left, right) {
    if (left === right) {
      return true;
    }
    const leftIsObject = left !== null && typeof left === "object";
    const rightIsObject = right !== null && typeof right === "object";
    if (leftIsObject || rightIsObject) {
      try {
        return JSON.stringify(stableValue(left)) === JSON.stringify(stableValue(right));
      } catch (_err) {
        return false;
      }
    }
    return false;
  }

  function pythonTruthy(value) {
    if (Array.isArray(value)) {
      return value.length > 0;
    }
    if (value && typeof value === "object") {
      return Object.keys(value).length > 0;
    }
    return !!value;
  }

  function periodicityDefaults(periodicity) {
    if (periodicity && periodicity.indexOf("weekly") === 0) {
      return [52, 4, 1, 13];
    }
    if (periodicity === "monthly") {
      return [12, 1, 1, 6];
    }
    return [252, 21, 1, 63];
  }

  function portoptModelDefaultName(model) {
    return portoptModelDefaultNames[model] || "Port";
  }

  function regressionModelDefaultName(model) {
    return regressionModelDefaultNames[model] || "Regression";
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

  function readSessionStoreValue(storeId) {
    if (!storeId) {
      return null;
    }
    try {
      const raw = sessionStorage.getItem(storeId);
      if (raw === null || raw === undefined) {
        return null;
      }
      return JSON.parse(raw);
    } catch (_err) {
      return null;
    }
  }

  function resolveStoredList(currentValue, storeId) {
    if (Array.isArray(currentValue) && currentValue.length) {
      return currentValue.slice();
    }
    const stored = readSessionStoreValue(storeId);
    return Array.isArray(stored) ? stored.slice() : [];
  }

  function resolveStoredString(currentValue, storeId) {
    if (typeof currentValue === "string" && currentValue.trim()) {
      return currentValue;
    }
    const stored = readSessionStoreValue(storeId);
    return typeof stored === "string" ? stored : currentValue;
  }

  function resolveStoredBool(currentValue, storeId) {
    if (currentValue === true) {
      return true;
    }
    return readSessionStoreValue(storeId) === true;
  }

  function resolveStoredNames(currentValue, storeId) {
    return storeNames(currentValue).length ? storeNames(currentValue) : storeNames(readSessionStoreValue(storeId));
  }

  function triggerUploadWithCancel(rootId, blockerStoreId) {
    const uploadDiv = document.getElementById(rootId);
    if (!uploadDiv) {
      return noUpdate();
    }
    const input = uploadDiv.querySelector('input[type="file"]');
    if (!input) {
      return noUpdate();
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
    if (typeof input.showPicker === "function") {
      input.showPicker();
    } else {
      input.click();
    }
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

  function uiBlockerRelease(dbErrorHidden, rawErrorHidden, portfolioErrorHidden, underlyingErrorHidden, seriesSelectionOpened) {
    const trigger = triggeredId() || "";
    if (trigger.indexOf("series-selection-modal") !== -1) {
      if (seriesSelectionOpened === false) {
        cancelDeferredModalOpen(trigger);
      }
      return seriesSelectionOpened === false ? false : noUpdate();
    }
    if (trigger.indexOf("raw-db-add-") !== -1) {
      return rawErrorHidden === false ? false : noUpdate();
    }
    if (trigger.indexOf("portfolio-add-") !== -1) {
      return portfolioErrorHidden === false ? false : noUpdate();
    }
    if (trigger.indexOf("underlying-add-") !== -1) {
      return underlyingErrorHidden === false ? false : noUpdate();
    }
    if (trigger.indexOf("db-add-") !== -1) {
      return dbErrorHidden === false ? false : noUpdate();
    }
    return noUpdate();
  }

  function releaseBlockerOnSeriesGridReady(virtualRows, modalOpened) {
    if (modalOpened === false) {
      return false;
    }
    if (modalOpened !== true) {
      return noUpdate();
    }
    if (Array.isArray(virtualRows)) {
      return false;
    }
    return noUpdate();
  }

  function commonDailyButtonDisabled(candidates, commonDailyCandidates, periodicityOptions) {
    const hasSeries = !!(candidates && candidates.available_series && candidates.available_series.length);
    if (!hasSeries) {
      return true;
    }
    const hasCommonDaily = !!(
      commonDailyCandidates &&
      commonDailyCandidates.common_daily_start &&
      commonDailyCandidates.common_daily_end
    );
    if (!hasCommonDaily) {
      return true;
    }
    const options = Array.isArray(periodicityOptions) ? periodicityOptions : [];
    const hasDailyTrading = options.some(function (opt) {
      return opt && opt.value === "daily_trading";
    });
    return !hasDailyTrading;
  }

  const accountListMaxEndSentinel = "3999-12-31";

  function sameClientsideValue(left, right) {
    if (left === right) {
      return true;
    }
    if (left == null || right == null) {
      return left == null && right == null;
    }
    if (typeof left !== typeof right) {
      return false;
    }
    if (typeof left !== "object") {
      return left === right;
    }
    try {
      return JSON.stringify(left) === JSON.stringify(right);
    } catch (err) {
      return false;
    }
  }

  function accountListDedupeStrings(values) {
    const out = [];
    const seen = new Set();
    (Array.isArray(values) ? values : []).forEach(function (value) {
      const textValue = String(value || "").trim();
      if (!textValue) {
        return;
      }
      const key = textValue.toLowerCase();
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      out.push(textValue);
    });
    return out;
  }

  function accountListNormalizeProvenance(provenanceStore) {
    if (!provenanceStore || typeof provenanceStore !== "object" || Array.isArray(provenanceStore)) {
      return {};
    }
    const normalized = {};
    Object.keys(provenanceStore).forEach(function (rawKey) {
      const rawValue = provenanceStore[rawKey];
      if (!rawValue || typeof rawValue !== "object" || Array.isArray(rawValue)) {
        return;
      }
      const loaderType = String(rawValue.loader_type || "").trim().toLowerCase();
      if (!loaderType) {
        return;
      }
      const emittedSeries = accountListDedupeStrings(rawValue.emitted_series);
      if (!emittedSeries.length) {
        return;
      }
      const entryId = String(rawValue.entry_id || rawKey || "").trim();
      if (!entryId) {
        return;
      }
      const loaderArgs = rawValue.loader_args && typeof rawValue.loader_args === "object" && !Array.isArray(rawValue.loader_args)
        ? rawValue.loader_args
        : {};
      const primarySeries = String(rawValue.primary_series || "").trim() || emittedSeries[0];
      normalized[entryId] = {
        entry_id: entryId,
        loader_type: loaderType,
        loader_args: loaderArgs,
        emitted_series: emittedSeries,
        primary_series: primarySeries
      };
    });
    return normalized;
  }

  function accountListPruneDbImportProvenance(rawMeta, provenanceStore) {
    const normalized = accountListNormalizeProvenance(provenanceStore);
    const allowed = new Set(
      (((rawMeta && rawMeta.columns) || []).map(function (name) {
        return String(name || "").trim();
      })).filter(function (name) {
        return !!name;
      })
    );
    const pruned = {};
    Object.keys(normalized).forEach(function (entryId) {
      const entry = normalized[entryId];
      const remaining = (entry.emitted_series || []).filter(function (series) {
        return allowed.has(series);
      });
      if (!remaining.length) {
        return;
      }
      const primarySeries = String(entry.primary_series || "").trim();
      pruned[entryId] = {
        entry_id: entry.entry_id,
        loader_type: entry.loader_type,
        loader_args: entry.loader_args && typeof entry.loader_args === "object" && !Array.isArray(entry.loader_args)
          ? entry.loader_args
          : {},
        emitted_series: remaining,
        primary_series: remaining.indexOf(primarySeries) !== -1 ? primarySeries : remaining[0]
      };
    });
    return sameValue(pruned, provenanceStore) ? noUpdate() : pruned;
  }

  function accountListRenderNotice(notice) {
    const message = String((notice && notice.message) || "").trim();
    if (!message) {
      return [];
    }
    const color = String((notice && notice.color) || "blue");
    return {
      props: {
        children: {
          props: {
            children: [
              {
                props: {
                  children: message,
                  style: { flex: "1 1 auto" }
                },
                type: "Text",
                namespace: "dash_mantine_components"
              },
              {
                props: {
                  children: "x",
                  id: "dashmat-account-list-notice-close-button",
                  color: color,
                  size: "sm",
                  variant: "subtle",
                  "aria-label": "Dismiss account list notice"
                },
                type: "ActionIcon",
                namespace: "dash_mantine_components"
              }
            ],
            align: "flex-start",
            justify: "space-between",
            wrap: "nowrap"
          },
          type: "Group",
          namespace: "dash_mantine_components"
        },
        color: color,
        mb: "sm",
        title: "Account Lists",
        variant: "light",
        withCloseButton: false
      },
      type: "Alert",
      namespace: "dash_mantine_components"
    };
  }

  function accountListDismissNotice(nClicks) {
    if (!nClicks) {
      return noUpdate();
    }
    return null;
  }

  function accountListSaveState(mode, nameValue, rows, provenanceStore) {
    if (String(mode || "load") !== "save") {
      return ["", true];
    }
    const cleanName = String(nameValue || "").trim();
    const normalizedRows = Array.isArray(rows) ? rows : [];
    const duplicateCount = cleanName
      ? normalizedRows.filter(function (row) {
          return String((row && row.ListName) || "").trim().toLowerCase() === cleanName.toLowerCase();
        }).length
      : 0;
    const helper = duplicateCount
      ? duplicateCount + " existing list(s) already use this name."
      : "Duplicate names are allowed.";
    const disabled = !cleanName || !Object.keys(accountListNormalizeProvenance(provenanceStore)).length;
    return [helper, disabled];
  }

  function accountListRowsRefreshTrigger(opened, mode, refreshCount, loadState, currentTrigger) {
    if (!opened || String(mode || "load") !== "load") {
      return noUpdate();
    }
    const status = (loadState && loadState.status) ? String(loadState.status).toLowerCase() : "idle";
    if (status !== "idle") {
      return noUpdate();
    }
    const nextTrigger = { refresh: Number(refreshCount || 0) };
    const trigger = triggeredId() || "";
    const forceUpdate = trigger === "dashmat-account-list-modal"
      || (trigger === "dashmat-account-list-modal-mode-store" && String(mode || "load") === "load");
    if (sameClientsideValue(currentTrigger, nextTrigger) && !forceUpdate) {
      return noUpdate();
    }
    return nextTrigger;
  }

  function analyticsResolveInitialRange(candidates, storedRange) {
    const source = candidates && typeof candidates === "object" ? candidates : {};
    const maxStart = source.max_start;
    const maxEnd = source.max_end;
    if (!maxStart || !maxEnd) {
      return [null, null];
    }
    if (storedRange && storedRange.start && storedRange.end) {
      const storedStart = storedRange.start;
      const storedEnd = storedRange.end === accountListMaxEndSentinel ? maxEnd : storedRange.end;
      if (storedStart >= maxStart && storedEnd <= maxEnd) {
        return [storedStart, storedEnd];
      }
    }
    return [maxStart, maxEnd];
  }

  function analyticsInitDateRange(
    candidates,
    storedRange,
    currentStartDate,
    currentEndDate,
    currentWrapperStyle,
    currentCommonDisabled,
    currentMaxDisabled,
    currentStateReady
  ) {
    const nu = noUpdate();
    const disabledStyle = { display: "flex", opacity: 0.5, pointerEvents: "none", alignItems: "flex-start" };
    const enabledStyle = { display: "flex", alignItems: "flex-start" };
    const hasSeries = !!(candidates && Array.isArray(candidates.available_series) && candidates.available_series.length);

    if (!hasSeries) {
      return [
        currentStartDate == null ? nu : null,
        currentEndDate == null ? nu : null,
        sameClientsideValue(currentWrapperStyle || {}, disabledStyle) ? nu : disabledStyle,
        currentCommonDisabled === true ? nu : true,
        currentMaxDisabled === true ? nu : true,
        storedRange == null ? nu : null,
        currentStateReady === false ? nu : false
      ];
    }

    const resolved = analyticsResolveInitialRange(candidates, storedRange);
    const startDate = resolved[0];
    const endDate = resolved[1];
    if (!startDate || !endDate) {
      return [
        currentStartDate == null ? nu : null,
        currentEndDate == null ? nu : null,
        sameClientsideValue(currentWrapperStyle || {}, disabledStyle) ? nu : disabledStyle,
        currentCommonDisabled === true ? nu : true,
        currentMaxDisabled === true ? nu : true,
        storedRange == null ? nu : null,
        currentStateReady === false ? nu : false
      ];
    }

    const nextRange = { start: startDate, end: endDate };
    return [
      currentStartDate === startDate ? nu : startDate,
      currentEndDate === endDate ? nu : endDate,
      sameClientsideValue(currentWrapperStyle || {}, enabledStyle) ? nu : enabledStyle,
      currentCommonDisabled === false ? nu : false,
      currentMaxDisabled === false ? nu : false,
      storedRange && storedRange.start === startDate && storedRange.end === endDate ? nu : nextRange,
      currentStateReady === true ? nu : true
    ];
  }

  function analyticsResolveButtonRange(candidates, buttonId, commonDailyCandidates) {
    const source = candidates && typeof candidates === "object" ? candidates : {};
    const normalizedId = String(buttonId || "");
    if (normalizedId.endsWith("common-range-button")) {
      return [source.common_start || null, source.common_end || null, false];
    }
    if (normalizedId.endsWith("common-daily-button")) {
      const dailySource = commonDailyCandidates && typeof commonDailyCandidates === "object"
        ? commonDailyCandidates
        : source;
      return [dailySource.common_daily_start || null, dailySource.common_daily_end || null, true];
    }
    if (normalizedId.endsWith("maximum-range-button")) {
      return [source.max_start || null, source.max_end || null, false];
    }
    return [null, null, false];
  }

  function analyticsDateRangeButtons(commonClicks, commonDailyClicks, maxClicks, candidates, commonDailyCandidates) {
    const nu = noUpdate();
    if (!(candidates && Array.isArray(candidates.available_series) && candidates.available_series.length)) {
      return [nu, nu, nu, nu, nu];
    }
    const buttonId = triggeredId();
    if (!buttonId) {
      return [nu, nu, nu, nu, nu];
    }
    const resolved = analyticsResolveButtonRange(candidates, buttonId, commonDailyCandidates);
    const startDate = resolved[0];
    const endDate = resolved[1];
    const forceDaily = !!resolved[2];
    if (!startDate || !endDate) {
      return [nu, nu, nu, nu, nu];
    }
    const periodicityValue = forceDaily ? "daily_trading" : nu;
    return [
      startDate,
      endDate,
      { start: startDate, end: endDate },
      periodicityValue,
      periodicityValue
    ];
  }

  function analyticsDateRangeStoreUpdate(startDate, endDate, existingRange) {
    const nu = noUpdate();
    if (!startDate || !endDate) {
      return nu;
    }
    if (
      existingRange &&
      existingRange.start === startDate &&
      existingRange.end === endDate
    ) {
      return nu;
    }
    return { start: startDate, end: endDate };
  }

  const portoptSplitReportingModels = [
    "risk_parity",
    "factor_risk_parity",
    "hierarchical_risk_parity",
    "minimize_variance",
    "minimize_cvar"
  ];

  function portoptSupportsSplitReporting(optModel, selectedSeries, longShortAssignments) {
    if (portoptSplitReportingModels.indexOf(optModel || "risk_parity") === -1) {
      return false;
    }
    const seriesList = Array.isArray(selectedSeries) ? selectedSeries : [];
    const assignmentMap = longShortAssignments && typeof longShortAssignments === "object" ? longShortAssignments : {};
    return seriesList.some(function (series) {
      return !!assignmentMap[series];
    });
  }

  function portoptReportingBasisControl(optModel, selectedSeries, longShortAssignments, currentValue) {
    const eligible = portoptSupportsSplitReporting(optModel || "risk_parity", selectedSeries || [], longShortAssignments || {});
    if (eligible) {
      return [
        false,
        noUpdate(),
        "Uses long-only returns for portfolio performance while keeping optimization on the selected basis."
      ];
    }
    return [
      true,
      currentValue === "match" ? noUpdate() : "match",
      "Available only for supported risk-based models when at least one selected series is marked Long/Short."
    ];
  }

  const portoptValidModels = [
    "risk_parity",
    "factor_risk_parity",
    "equal_weight",
    "hrp",
    "maximize_sharpe",
    "minimize_cvar",
    "minimize_variance",
    "ex_ante_mv",
    "black_litterman"
  ];
  const portoptValidCovShrinkage = ["none", "ledoit_wolf", "oas"];
  const portoptValidCovShrinkageTarget = ["scaled_identity", "constant_correlation"];

  function portoptBootstrapReady(bootstrapState) {
    return !!(bootstrapState && bootstrapState.phase === "ready");
  }

  function portoptCoerceFloat(value) {
    const fval = Number(value);
    return Number.isFinite(fval) ? fval : null;
  }

  function portoptValidateLinearConstraints(linearConstraints, selectedSeries) {
    const assets = Array.isArray(selectedSeries) ? selectedSeries.map(function (series) { return String(series); }) : [];
    const rows = Array.isArray(linearConstraints) ? linearConstraints : [];
    for (let idx = 0; idx < rows.length; idx += 1) {
      const row = rows[idx];
      if (!row || typeof row !== "object" || Array.isArray(row)) {
        return "Linear constraint row #" + (idx + 1) + " is invalid.";
      }

      let coeffCount = 0;
      for (let assetIdx = 0; assetIdx < assets.length; assetIdx += 1) {
        const asset = assets[assetIdx];
        const value = row[asset];
        if (value === null || value === undefined || value === "") {
          continue;
        }
        const fval = portoptCoerceFloat(value);
        if (fval === null) {
          return "Linear constraint row #" + (idx + 1) + " has invalid coefficient for " + asset + ".";
        }
        if (Math.abs(fval) > 1e-12) {
          coeffCount += 1;
        }
      }

      const minRaw = row.Min;
      const maxRaw = row.Max;
      const minVal = (minRaw === null || minRaw === undefined || minRaw === "") ? null : portoptCoerceFloat(minRaw);
      const maxVal = (maxRaw === null || maxRaw === undefined || maxRaw === "") ? null : portoptCoerceFloat(maxRaw);
      if (minRaw !== null && minRaw !== undefined && minRaw !== "" && minVal === null) {
        return "Linear constraint row #" + (idx + 1) + " has invalid Min value.";
      }
      if (maxRaw !== null && maxRaw !== undefined && maxRaw !== "" && maxVal === null) {
        return "Linear constraint row #" + (idx + 1) + " has invalid Max value.";
      }
      if (minVal !== null && maxVal !== null && minVal > maxVal) {
        return "Linear constraint row #" + (idx + 1) + " has Min greater than Max.";
      }
      if (coeffCount === 0 && (minVal !== null || maxVal !== null)) {
        return "Linear constraint row #" + (idx + 1) + " needs at least one non-zero coefficient.";
      }
    }
    return null;
  }

  function portoptValidateExAnteInputs(selectedSeries, exAnteMode, exAnteReturns, exAnteCov, exAnteVol, exAnteCorr) {
    const assets = Array.isArray(selectedSeries) ? selectedSeries.map(function (series) { return String(series); }) : [];
    if (!assets.length) {
      return "Select at least one series.";
    }

    const mode = exAnteMode || "ret_cov";
    const returnsMap = exAnteReturns && typeof exAnteReturns === "object" ? exAnteReturns : {};
    const covMap = exAnteCov && typeof exAnteCov === "object" ? exAnteCov : {};
    const volMap = exAnteVol && typeof exAnteVol === "object" ? exAnteVol : {};
    const corrMap = exAnteCorr && typeof exAnteCorr === "object" ? exAnteCorr : {};

    const missingReturns = assets.filter(function (asset) {
      return portoptCoerceFloat(returnsMap[asset]) === null;
    });
    if (missingReturns.length) {
      return "Missing expected return for: " + missingReturns.join(", ") + ".";
    }

    if (mode === "ret_vol_corr") {
      const missingVols = assets.filter(function (asset) {
        return portoptCoerceFloat(volMap[asset]) === null;
      });
      if (missingVols.length) {
        return "Missing expected volatility for: " + missingVols.join(", ") + ".";
      }

      for (let rIdx = 0; rIdx < assets.length; rIdx += 1) {
        const rowAsset = assets[rIdx];
        const row = corrMap && typeof corrMap === "object" ? corrMap[rowAsset] : {};
        if (!row || typeof row !== "object" || Array.isArray(row)) {
          return "Correlation row for '" + rowAsset + "' is invalid.";
        }
        for (let cIdx = 0; cIdx < assets.length; cIdx += 1) {
          const colAsset = assets[cIdx];
          const corrVal = portoptCoerceFloat(row[colAsset]);
          if (corrVal === null) {
            return "Missing correlation value for (" + rowAsset + ", " + colAsset + ").";
          }
          if (corrVal < -1 || corrVal > 1) {
            return "Correlation (" + rowAsset + ", " + colAsset + ") must be between -1 and 1.";
          }
        }
      }
      return null;
    }

    for (let rIdx = 0; rIdx < assets.length; rIdx += 1) {
      const rowAsset = assets[rIdx];
      const row = covMap && typeof covMap === "object" ? covMap[rowAsset] : {};
      if (!row || typeof row !== "object" || Array.isArray(row)) {
        return "Covariance row for '" + rowAsset + "' is invalid.";
      }
      for (let cIdx = 0; cIdx < assets.length; cIdx += 1) {
        const colAsset = assets[cIdx];
        if (portoptCoerceFloat(row[colAsset]) === null) {
          return "Missing covariance value for (" + rowAsset + ", " + colAsset + ").";
        }
      }
    }
    return null;
  }

  function portoptValidateBlackLittermanInputs(selectedSeries, blViews, blTau) {
    const tau = portoptCoerceFloat(blTau);
    if (tau === null || tau <= 0) {
      return "BL tau must be greater than 0.";
    }

    const assets = new Set(Array.isArray(selectedSeries) ? selectedSeries.map(function (series) { return String(series); }) : []);
    const views = Array.isArray(blViews) ? blViews : [];
    for (let idx = 0; idx < views.length; idx += 1) {
      const view = views[idx];
      const viewNumber = idx + 1;
      if (!view || typeof view !== "object" || Array.isArray(view)) {
        return "BL view #" + viewNumber + " is invalid.";
      }
      const viewType = String(view.type || "absolute").trim().toLowerCase();
      if (viewType !== "absolute" && viewType !== "relative") {
        return "BL view #" + viewNumber + " type must be absolute or relative.";
      }
      if (portoptCoerceFloat(view["return"]) === null) {
        return "BL view #" + viewNumber + " return is invalid.";
      }
      const confidence = portoptCoerceFloat(view.confidence === undefined ? 1.0 : view.confidence);
      if (confidence === null || confidence <= 0) {
        return "BL view #" + viewNumber + " confidence must be greater than 0.";
      }

      const asset = String(view.asset || "").trim();
      if (viewType === "absolute") {
        if (!asset || !assets.has(asset)) {
          return "BL view #" + viewNumber + " asset must be one of the selected series.";
        }
        continue;
      }

      const assetTo = String(view.asset_to || "").trim();
      if (!asset || !assetTo) {
        return "BL view #" + viewNumber + " relative pair is incomplete.";
      }
      if (!assets.has(asset) || !assets.has(assetTo)) {
        return "BL view #" + viewNumber + " relative assets must be selected series.";
      }
      if (asset === assetTo) {
        return "BL view #" + viewNumber + " relative assets must be different.";
      }
    }

    return null;
  }

  function portoptValidateOptimizationInputs(
    portfolioName,
    selectedSeries,
    optModel,
    optWindow,
    windowSize,
    optStep,
    optStepUnit,
    expWtCov,
    halflife,
    covShrinkage,
    covShrinkageTarget,
    minWt,
    maxWt,
    forceMax,
    linearConstraints,
    exAnteMode,
    exAnteReturns,
    exAnteCov,
    exAnteVol,
    exAnteCorr,
    blViews,
    blTau
  ) {
    if (!portfolioName || !String(portfolioName).trim()) {
      return "Enter a portfolio name.";
    }

    const seriesList = Array.isArray(selectedSeries) ? selectedSeries : [];
    if (!seriesList.length) {
      return "Select at least one series.";
    }
    if (seriesList.length < 2) {
      return "Select at least two series.";
    }

    if (portoptValidModels.indexOf(optModel) === -1) {
      return "Select a valid optimization model.";
    }

    if (optModel !== "ex_ante_mv" && optModel !== "black_litterman") {
      if (["full", "rolling", "expanding"].indexOf(optWindow) === -1) {
        return "Select a valid optimization window.";
      }
      if (optWindow !== "full") {
        const ws = portoptCoerceFloat(windowSize);
        if (ws === null || ws < 2 || Math.trunc(ws) !== ws) {
          return "Window size must be an integer >= 2.";
        }
        const step = portoptCoerceFloat(optStep);
        if (step === null || step < 1 || Math.trunc(step) !== step) {
          return "Optimization step must be an integer >= 1.";
        }
        if (optStepUnit !== "periods" && optStepUnit !== "months") {
          return "Optimization step unit must be periods or months.";
        }
      }
    }

    if (!!expWtCov) {
      const hl = portoptCoerceFloat(halflife);
      if (hl === null || hl <= 0) {
        return "Decay input must be greater than 0 when exponential weighting is enabled.";
      }
    }

    const rawShrinkage = (covShrinkage === null || covShrinkage === undefined || covShrinkage === "")
      ? "none"
      : String(covShrinkage).trim().toLowerCase();
    if (portoptValidCovShrinkage.indexOf(rawShrinkage) === -1) {
      return "Select a valid covariance shrinkage option.";
    }
    const rawTarget = (covShrinkageTarget === null || covShrinkageTarget === undefined || covShrinkageTarget === "")
      ? "scaled_identity"
      : String(covShrinkageTarget).trim().toLowerCase();
    if (portoptValidCovShrinkageTarget.indexOf(rawTarget) === -1) {
      return "Select a valid covariance shrinkage target.";
    }

    const minMap = minWt && typeof minWt === "object" ? minWt : {};
    const maxMap = maxWt && typeof maxWt === "object" ? maxWt : {};
    const forceMap = forceMax && typeof forceMax === "object" ? forceMax : {};
    for (let idx = 0; idx < seriesList.length; idx += 1) {
      const asset = seriesList[idx];
      const mn = portoptCoerceFloat(Object.prototype.hasOwnProperty.call(minMap, asset) ? minMap[asset] : 0);
      const mx = portoptCoerceFloat(Object.prototype.hasOwnProperty.call(maxMap, asset) ? maxMap[asset] : 100);
      if (mn === null || mx === null) {
        return "Invalid min/max bound for " + asset + ".";
      }
      if (mn < 0 || mx > 100) {
        return "Bounds for " + asset + " must stay within 0-100%.";
      }
      if (mn > mx) {
        return "Min bound cannot exceed max bound for " + asset + ".";
      }
      if (!!forceMap[asset] && mx <= 0) {
        return "Force Max requires a positive max bound for " + asset + ".";
      }
    }

    const linearError = portoptValidateLinearConstraints(linearConstraints, seriesList);
    if (linearError) {
      return linearError;
    }

    if (optModel === "ex_ante_mv") {
      const exAnteError = portoptValidateExAnteInputs(
        seriesList,
        exAnteMode,
        exAnteReturns,
        exAnteCov,
        exAnteVol,
        exAnteCorr
      );
      if (exAnteError) {
        return exAnteError;
      }
    }

    if (optModel === "black_litterman") {
      const blError = portoptValidateBlackLittermanInputs(seriesList, blViews, blTau);
      if (blError) {
        return blError;
      }
    }

    return null;
  }

  function portoptToggleUiElements(
    bootstrapState,
    portfolioName,
    selectedSeries,
    optModel,
    optWindow,
    windowSize,
    optStep,
    optStepUnit,
    expWtCov,
    halflife,
    covShrinkage,
    covShrinkageTarget,
    minWt,
    maxWt,
    forceMax,
    linearConstraints,
    exAnteMode,
    exAnteReturns,
    exAnteCov,
    exAnteVol,
    exAnteCorr,
    blViews,
    blTau,
    welcomeStyle,
    resultsMeta
  ) {
    const saveDisabled = !(welcomeStyle && welcomeStyle.display === "none");
    const downloadDisabled = !((resultsMeta && resultsMeta.has_results) === true);
    if (!portoptBootstrapReady(bootstrapState)) {
      return [true, "Loading controls...", false, saveDisabled, downloadDisabled];
    }

    const validationError = portoptValidateOptimizationInputs(
      portfolioName,
      selectedSeries,
      optModel,
      optWindow,
      windowSize,
      optStep,
      optStepUnit,
      expWtCov,
      halflife,
      covShrinkage,
      covShrinkageTarget,
      minWt,
      maxWt,
      forceMax,
      linearConstraints,
      exAnteMode,
      exAnteReturns,
      exAnteCov,
      exAnteVol,
      exAnteCorr,
      blViews,
      blTau
    );
    const runDisabled = validationError !== null;
    return [runDisabled, validationError || "Run optimization.", false, saveDisabled, downloadDisabled];
  }

  function portoptLinearConstraintColumnDefs(selectedSeries) {
    if (!Array.isArray(selectedSeries) || !selectedSeries.length) {
      return [];
    }

    const numericColumn = {
      editable: true,
      width: 90,
      type: "numericColumn",
      valueFormatter: { function: "d3.format('.4f')(params.value)" },
      headerClass: "dashmat-center-header"
    };
    const columns = [
      { field: "Constraint", editable: true, width: 120, headerClass: "dashmat-center-header" },
      Object.assign({ field: "Min" }, numericColumn),
      Object.assign({ field: "Max" }, numericColumn)
    ];

    selectedSeries.forEach(function (series) {
      columns.push({
        field: series,
        editable: true,
        width: 100,
        type: "numericColumn",
        valueFormatter: { function: "d3.format('.4f')(params.value)" },
        headerClass: "dashmat-center-header"
      });
    });

    return columns;
  }

  function portoptReturnsGridData(selectedSeries, mode, existingReturns, existingVol) {
    if (!Array.isArray(selectedSeries) || !selectedSeries.length) {
      return [[], []];
    }

    const returnsMap = existingReturns && typeof existingReturns === "object" ? existingReturns : {};
    const volMap = existingVol && typeof existingVol === "object" ? existingVol : {};
    const resolvedMode = mode || "ret_cov";
    const hideVol = resolvedMode !== "ret_vol_corr";
    const columnDefs = [
      { field: "Asset", editable: false, width: 140, headerClass: "dashmat-center-header" },
      {
        field: "Return",
        editable: true,
        width: 110,
        type: "numericColumn",
        valueFormatter: { function: "d3.format('.2%')(params.value)" },
        valueParser: { function: "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;" },
        headerClass: "dashmat-center-header"
      },
      {
        field: "Volatility",
        editable: true,
        width: 110,
        type: "numericColumn",
        valueFormatter: { function: "d3.format('.2%')(params.value)" },
        valueParser: { function: "var v=params.newValue; if (v===null || v===undefined || v==='') return null; var n=Number(v); if (!isFinite(n)) return null; return Math.abs(n) > 1 ? n/100 : n;" },
        hide: hideVol,
        headerClass: "dashmat-center-header"
      }
    ];
    const rows = selectedSeries.map(function (series) {
      return {
        Asset: series,
        Return: Object.prototype.hasOwnProperty.call(returnsMap, series) ? returnsMap[series] : 0.0,
        Volatility: Object.prototype.hasOwnProperty.call(volMap, series) ? volMap[series] : 0.0
      };
    });
    return [rows, columnDefs];
  }

  function portoptMatrixGridData(selectedSeries, mode, covStore, corrStore) {
    if (!Array.isArray(selectedSeries) || !selectedSeries.length) {
      return [[], []];
    }

    const resolvedMode = mode || "ret_cov";
    const isCorr = resolvedMode === "ret_vol_corr";
    const matrixStore = (isCorr ? corrStore : covStore) && typeof (isCorr ? corrStore : covStore) === "object"
      ? (isCorr ? corrStore : covStore)
      : {};
    const columnDefs = [
      {
        field: "Asset",
        editable: false,
        width: 140,
        pinned: "left",
        valueFormatter: { function: "params.value" },
        headerClass: "dashmat-center-header"
      }
    ];
    selectedSeries.forEach(function (series) {
      columnDefs.push({
        field: series,
        editable: true,
        width: 110,
        type: "numericColumn",
        valueFormatter: { function: "params.value !== null && params.value !== undefined && params.value !== '' && isFinite(Number(params.value)) ? d3.format(',.4f')(Number(params.value)) : ''" },
        headerClass: "dashmat-center-header"
      });
    });

    const rows = selectedSeries.map(function (rowName) {
      const rowKey = String(rowName);
      const rowVals = matrixStore[rowKey] || matrixStore[rowName] || {};
      const row = { Asset: rowKey };
      selectedSeries.forEach(function (columnName) {
        let value = rowVals[columnName];
        if (value === undefined || value === null) {
          value = NaN;
        }
        row[columnName] = value;
      });
      return row;
    });
    return [rows, columnDefs];
  }

  function portoptActiveVisTrigger(activeTab, selectedPortfolio, periodicity, returnsBasis, rollingWindow, rollingReturnType, rollingMetric, rollingView, savedSeriesStore, useRiskFree, calendarView, calendarSeries, partialMode, drawdownView) {
    const gatedTabs = ["returns", "rolling", "statistics", "calendar", "drawdown"];
    if (gatedTabs.indexOf(activeTab) === -1) {
      return noUpdate();
    }
    return {
      tab: activeTab,
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function startInitialSeriesModalBlocker(pathname, pageLoadReady, modalOpened, modalStillNeeded, virtualRows, targetPath) {
    const pagePath = normalizePath(pathname);
    if (pagePath !== targetPath) {
      return noUpdate();
    }
    if (Array.isArray(virtualRows)) {
      return false;
    }
    if (modalOpened === true) {
      return true;
    }
    if (modalStillNeeded) {
      return true;
    }
    return pageLoadReady ? false : noUpdate();
  }

  function buildLocalBlockerState(pathname, pageLoadReady, modalOpened, modalStillNeeded, virtualRows, targetPath) {
    return startInitialSeriesModalBlocker(
      pathname,
      pageLoadReady,
      modalOpened,
      modalStillNeeded,
      virtualRows,
      targetPath
    );
  }

  function analyticsInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited) {
    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      return false;
    }
    const columnSet = new Set(columns);
    const selected = Array.isArray(currentSelect) ? currentSelect : [];
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set((Array.isArray(currentOrder) ? currentOrder : []).filter(function (series) {
      return columnSet.has(series);
    }));
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const poOriginSet = new Set(storeNames(poOriginSeries).filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });
    return (!pageVisited && !selectedValid.length) || genericNew.length > 0;
  }

  function analyticsInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      analyticsInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/analyticstool"
    );
  }

  function analyticsSeriesSelectionRenderTrigger(
    opened,
    rawMeta,
    selectedSeries,
    seriesOrder,
    deletedSeries,
    benchmarkAssignments,
    longShortAssignments,
    volScalingAssignments,
    currentTrigger
  ) {
    if (!opened) {
      return noUpdate();
    }
    const nextTrigger = {
      dataset_key: rawMeta && rawMeta.dataset_key ? rawMeta.dataset_key : null,
      has_data: !!(rawMeta && rawMeta.has_data),
      columns_sig: rawMetaColumns(rawMeta).join("|"),
      selected_sig: JSON.stringify(Array.isArray(selectedSeries) ? selectedSeries : []),
      order_sig: JSON.stringify(Array.isArray(seriesOrder) ? seriesOrder : []),
      deleted_sig: JSON.stringify(Array.isArray(deletedSeries) ? deletedSeries : []),
      benchmark_sig: JSON.stringify(benchmarkAssignments && typeof benchmarkAssignments === "object" ? benchmarkAssignments : {}),
      long_short_sig: JSON.stringify(longShortAssignments && typeof longShortAssignments === "object" ? longShortAssignments : {}),
      vol_scaling_sig: JSON.stringify(volScalingAssignments && typeof volScalingAssignments === "object" ? volScalingAssignments : {})
    };
    if (sameValue(currentTrigger, nextTrigger) && triggeredId() !== "at-series-selection-modal") {
      return noUpdate();
    }
    return nextTrigger;
  }

  function portoptInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited) {
    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      return false;
    }
    const columnSet = new Set(columns);
    const selected = resolveStoredList(currentSelect, "po-series-select");
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set(
      (selectedValid.length ? resolveStoredList(currentOrder, "po-series-order-store") : []).filter(function (series) {
        return columnSet.has(series);
      })
    );
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const poOriginSet = new Set(resolveStoredNames(poOriginSeries, "dashmat-pending-new-series-store").filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });
    if (!resolveStoredBool(pageVisited, "po-page-visited-store") && !selectedValid.length) {
      return columns.some(function (series) {
        return !poOriginSet.has(series);
      });
    }
    return genericNew.length > 0;
  }

  function portoptInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      portoptInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/portopt"
    );
  }

  function rawMetaColumns(rawMeta) {
    if (Array.isArray(rawMeta)) {
      return rawMeta.slice();
    }
    if (rawMeta && Array.isArray(rawMeta.columns)) {
      return rawMeta.columns.slice();
    }
    return [];
  }

  function validateAnalyticsDbAddSelection(selectedBenches, opened, rawMeta) {
    if (!opened) {
      return [noUpdate(), noUpdate(), noUpdate()];
    }

    const selected = Array.isArray(selectedBenches)
      ? selectedBenches.filter(function (series) {
          return typeof series === "string" && series.length > 0;
        })
      : (typeof selectedBenches === "string" && selectedBenches.length > 0 ? [selectedBenches] : []);
    if (!selected.length) {
      return [noUpdate(), true, true];
    }

    const existing = new Set(rawMetaColumns(rawMeta));
    const duplicates = selected.filter(function (series) {
      return existing.has(series);
    });
    if (duplicates.length) {
      return ["Cannot add duplicate series: " + duplicates.join(", "), false, true];
    }
    return [noUpdate(), true, false];
  }

  function latestGridEvent(payload) {
    let evt = payload;
    if (Array.isArray(evt)) {
      for (let i = evt.length - 1; i >= 0; i -= 1) {
        if (evt[i] && typeof evt[i] === "object") {
          evt = evt[i];
          break;
        }
      }
    }
    return evt && typeof evt === "object" ? evt : null;
  }

  async function captureGridSnapshot(gridId, modalOpened) {
    if (modalOpened === false) {
      return noUpdate();
    }
    if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
      return noUpdate();
    }
    for (let attempt = 0; attempt < 8; attempt += 1) {
      try {
        const api = await window.dash_ag_grid.getApiAsync(gridId);
        if (!api) {
          if (attempt < 7) {
            await sleepMs(75);
          }
          continue;
        }
        try {
          api.stopEditing();
        } catch (_err) {
        }
        const rows = [];
        api.forEachNodeAfterFilterAndSort(function (node) {
          if (node && node.data) {
            rows.push(Object.assign({}, node.data));
          }
        });
        return {
          rows: rows,
          capturedAt: Date.now()
        };
      } catch (_err) {
        if (attempt < 7) {
          await sleepMs(75);
        }
      }
    }
    return noUpdate();
  }

  async function captureAnalyticsSeriesSnapshot(nClicks, modalOpened) {
    if (!nClicks) {
      return noUpdate();
    }
    return captureGridSnapshot("at-series-selection-grid", modalOpened);
  }

  async function capturePortoptSeriesSnapshot(nClicks, modalOpened) {
    if (!nClicks) {
      return noUpdate();
    }
    return captureGridSnapshot("po-series-selection-grid", modalOpened);
  }

  async function captureRegressionSeriesSnapshot(nClicks, modalOpened) {
    if (!nClicks) {
      return noUpdate();
    }
    return captureGridSnapshot("reg-series-selection-grid", modalOpened);
  }

  async function bulkUpdateSeriesSelection(selectAllClicks, unselectAllClicks, modalOpened) {
    const trigger = triggeredId() || "";
    if (!trigger || modalOpened === false) {
      return noUpdate();
    }
    if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
      return noUpdate();
    }

    const isSelectAll = trigger.indexOf("-select-all-button") !== -1;
    const isUnselectAll = trigger.indexOf("-unselect-all-button") !== -1;
    if ((!isSelectAll && !isUnselectAll) || (!selectAllClicks && !unselectAllClicks)) {
      return noUpdate();
    }

    let gridId = null;
    let targetField = null;
    if (trigger.indexOf("at-") === 0) {
      gridId = "at-series-selection-grid";
      targetField = "Selected";
    } else if (trigger.indexOf("po-") === 0) {
      gridId = "po-series-selection-grid";
      targetField = "Selected";
    } else if (trigger.indexOf("reg-") === 0) {
      gridId = "reg-series-selection-grid";
      targetField = "X";
    } else {
      return noUpdate();
    }

    try {
      const api = await window.dash_ag_grid.getApiAsync(gridId);
      if (!api) {
        return noUpdate();
      }
      try {
        api.stopEditing();
      } catch (_err) {
      }

      const nextValue = isSelectAll;
      let changed = false;
      api.forEachNode(function (node) {
        if (!node || !node.data || node.data.Delete) {
          return;
        }
        if (!!node.data[targetField] === nextValue) {
          return;
        }
        changed = true;
        node.setData(
          Object.assign({}, node.data, {
            [targetField]: nextValue
          })
        );
      });

      if (!changed) {
        return noUpdate();
      }
      try {
        api.refreshCells({ columns: [targetField], force: true });
      } catch (_err) {
      }
      return Date.now();
    } catch (_err) {
      return noUpdate();
    }
  }

  async function enforceRegressionSingleY(cellClick, modalOpened) {
    const evt = latestGridEvent(cellClick);
    if (!evt || modalOpened === false) {
      return noUpdate();
    }
    const colId = evt.colId || (evt.column && evt.column.colId);
    if (colId !== "YDisplay" && colId !== "Y") {
      return noUpdate();
    }
    if (!window.dash_ag_grid || !window.dash_ag_grid.getApiAsync) {
      return noUpdate();
    }
    try {
      const api = await window.dash_ag_grid.getApiAsync("reg-series-selection-grid");
      if (!api) {
        return noUpdate();
      }
      try {
        api.stopEditing();
      } catch (_err) {
      }
      const targetKey = evt.rowId || (evt.data && (evt.data.__row_key || evt.data.Series));
      if (!targetKey) {
        return noUpdate();
      }

      let targetNode = null;
      if (api.getRowNode) {
        targetNode = api.getRowNode(String(targetKey));
      }
      if (!targetNode) {
        api.forEachNode(function (node) {
          if (node && node.data) {
            const rowKey = node.data.__row_key || node.data.Series;
            if (String(rowKey) === String(targetKey)) {
              targetNode = node;
            }
          }
        });
      }

      if (!targetNode || !targetNode.data) {
        return noUpdate();
      }

      api.forEachNode(function (node) {
        if (!node || !node.data) {
          return;
        }
        const rowKey = node.data.__row_key || node.data.Series;
        const shouldBeSelected = String(rowKey) === String(targetKey);
        const nextDisplay = shouldBeSelected ? "●" : "○";
        const nextDelete = shouldBeSelected ? false : !!node.data.Delete;
        if (
          !!node.data.Y !== shouldBeSelected ||
          (node.data.YDisplay || "") !== nextDisplay ||
          !!node.data.Delete !== nextDelete
        ) {
          node.setData(
            Object.assign({}, node.data, {
              Y: shouldBeSelected,
              YDisplay: nextDisplay,
              Delete: nextDelete
            })
          );
        }
      });
      try {
        api.refreshCells({ columns: ["Y", "YDisplay", "Delete"], force: true });
      } catch (_err) {
      }
      return Date.now();
    } catch (_err) {
      return noUpdate();
    }
  }

  function storeNames(store) {
    if (!store) {
      return [];
    }
    if (Array.isArray(store)) {
      return store.slice();
    }
    if (typeof store === "object") {
      return Object.keys(store);
    }
    return [];
  }

  function nonEmptyStringMap(store) {
    const source = (store && typeof store === "object") ? store : {};
    const cleaned = {};
    Object.keys(source).forEach(function (key) {
      const value = source[key];
      if (typeof value === "string" && value.trim()) {
        cleaned[key] = value.trim();
      }
    });
    return cleaned;
  }

  function normalizePortoptSeriesOrder(allSeries, seriesOrder) {
    const known = new Set(allSeries || []);
    const normalized = Array.isArray(seriesOrder) && seriesOrder.length ? seriesOrder.slice() : (allSeries || []).slice();
    (allSeries || []).forEach(function (series) {
      if (known.has(series) && normalized.indexOf(series) === -1) {
        normalized.push(series);
      }
    });
    return normalized.filter(function (series) {
      return known.has(series);
    });
  }

  function portoptSeriesSelectionColumnDefs(benchmarkValues, cmabenchEditorValues) {
    return [
      {
        headerName: "",
        rowDrag: true,
        editable: false,
        sortable: false,
        filter: false,
        resizable: false,
        width: 36,
        pinned: "left",
        valueGetter: { function: "''" },
        cellClass: "dashmat-series-center-cell"
      },
      {
        field: "Selected",
        headerName: "Use",
        editable: true,
        cellRenderer: "agCheckboxCellRenderer",
        cellEditor: "agCheckboxCellEditor",
        width: 72,
        pinned: "left",
        cellClass: "dashmat-series-center-cell"
      },
      {
        field: "Series",
        editable: true,
        minWidth: 150,
        cellStyle: { textAlign: "left", fontFamily: "monospace" },
        headerClass: "dashmat-left-header"
      },
      {
        field: "Benchmark",
        editable: true,
        cellEditor: "agSelectCellEditor",
        cellEditorParams: { values: benchmarkValues },
        minWidth: 150,
        cellStyle: { textAlign: "left" },
        headerClass: "dashmat-left-header"
      },
      {
        field: "CMABench",
        editable: true,
        cellEditor: "agSelectCellEditor",
        cellEditorParams: { values: cmabenchEditorValues },
        minWidth: 130,
        cellStyle: { textAlign: "left" },
        headerClass: "dashmat-left-header"
      },
      {
        field: "LongShort",
        headerName: "L/S",
        editable: true,
        cellRenderer: "agCheckboxCellRenderer",
        cellEditor: "agCheckboxCellEditor",
        width: 72,
        cellClass: "dashmat-series-center-cell"
      },
      {
        field: "ScaleVol",
        headerName: "Scale Vol",
        editable: true,
        cellRenderer: "agCheckboxCellRenderer",
        cellEditor: "agCheckboxCellEditor",
        width: 112,
        cellClass: "dashmat-series-center-cell"
      },
      {
        field: "MinWt",
        headerName: "Min Wt",
        editable: { function: "!params.data.ForceMax" },
        width: 98,
        valueParser: {
          function: "var n=Number(params.newValue); if(!isFinite(n)) return 0; return Math.max(0, Math.min(100, n));"
        },
        cellClass: "dashmat-series-center-cell",
        headerClass: "dashmat-center-header"
      },
      {
        field: "MaxWt",
        headerName: "Max Wt",
        editable: true,
        width: 98,
        valueParser: {
          function: "var n=Number(params.newValue); if(!isFinite(n)) return 100; return Math.max(0, Math.min(100, n));"
        },
        cellClass: "dashmat-series-center-cell",
        headerClass: "dashmat-center-header"
      },
      {
        field: "ForceMax",
        headerName: "Force",
        editable: true,
        cellRenderer: "agCheckboxCellRenderer",
        cellEditor: "agCheckboxCellEditor",
        width: 70,
        cellClass: "dashmat-series-center-cell"
      },
      {
        field: "Delete",
        editable: true,
        cellRenderer: "agCheckboxCellRenderer",
        cellEditor: "agCheckboxCellEditor",
        width: 74,
        cellClass: "dashmat-series-center-cell"
      }
    ];
  }

  function syncPortoptSeriesModalGrid(
    rawMeta,
    selectedSeries,
    seriesOrder,
    deletedSeries,
    currentAssignments,
    currentCmabenchAssignments,
    currentCmabenchDefaults,
    longShortAssignments,
    volScalingAssignments,
    minWt,
    maxWt,
    forceMax,
    cmabenchOptionValues,
    currentColumnDefs,
    modalOpened
  ) {
    const allSeries = rawMetaColumns(rawMeta);
    const emptyColumnDefs = portoptSeriesSelectionColumnDefs(["None"], [""]);
    const currentOrder = Array.isArray(seriesOrder) ? seriesOrder.slice() : [];
    const emptyOrderUpdate = currentOrder.length ? [] : noUpdate();
    const currentColumnDefsJson = JSON.stringify(Array.isArray(currentColumnDefs) ? currentColumnDefs : []);
    const emptyColumnDefsJson = JSON.stringify(emptyColumnDefs);
    const emptyColumnUpdate = currentColumnDefsJson === emptyColumnDefsJson ? noUpdate() : emptyColumnDefs;
    if (!allSeries.length) {
      return [[], emptyColumnUpdate, emptyOrderUpdate, modalOpened === true ? false : noUpdate()];
    }

    const normalizedOrder = normalizePortoptSeriesOrder(allSeries, currentOrder);
    const selectedSet = new Set(Array.isArray(selectedSeries) ? selectedSeries : []);
    const deletedSet = new Set(Array.isArray(deletedSeries) ? deletedSeries : []);
    const benchmarkAssignments = (currentAssignments && typeof currentAssignments === "object") ? currentAssignments : {};
    const explicitCmabenchAssignments = nonEmptyStringMap(currentCmabenchAssignments);
    const importedCmabenchDefaults = nonEmptyStringMap(currentCmabenchDefaults);
    const longShortMap = (longShortAssignments && typeof longShortAssignments === "object") ? longShortAssignments : {};
    const volScalingMap = (volScalingAssignments && typeof volScalingAssignments === "object") ? volScalingAssignments : {};
    const minWtMap = (minWt && typeof minWt === "object") ? minWt : {};
    const maxWtMap = (maxWt && typeof maxWt === "object") ? maxWt : {};
    const forceMaxMap = (forceMax && typeof forceMax === "object") ? forceMax : {};

    const benchmarkValues = ["None"].concat(allSeries);
    const cmabenchOptionSet = new Set([""]);
    (Array.isArray(cmabenchOptionValues) ? cmabenchOptionValues : []).forEach(function (value) {
      if (typeof value === "string" && value.trim()) {
        cmabenchOptionSet.add(value.trim());
      }
    });
    Object.keys(importedCmabenchDefaults).forEach(function (key) {
      cmabenchOptionSet.add(importedCmabenchDefaults[key]);
    });
    Object.keys(explicitCmabenchAssignments).forEach(function (key) {
      cmabenchOptionSet.add(explicitCmabenchAssignments[key]);
    });
    const cmabenchEditorValues = Array.from(cmabenchOptionSet).sort(function (a, b) {
      if (a === "") {
        return -1;
      }
      if (b === "") {
        return 1;
      }
      return a.localeCompare(b);
    });

    const rowData = normalizedOrder.map(function (series) {
      let benchmarkValue = benchmarkAssignments[series];
      benchmarkValue = typeof benchmarkValue === "string" && benchmarkValue.trim() ? benchmarkValue.trim() : "None";
      if (benchmarkValue !== "None" && allSeries.indexOf(benchmarkValue) === -1) {
        benchmarkValue = "None";
      }
      let minWtValue = Number(minWtMap[series]);
      if (!isFinite(minWtValue)) {
        minWtValue = 0;
      }
      let maxWtValue = Number(maxWtMap[series]);
      if (!isFinite(maxWtValue)) {
        maxWtValue = 100;
      }
      const explicitCmabench = explicitCmabenchAssignments[series] || "";
      const defaultCmabench = importedCmabenchDefaults[series] || "";
      return {
        __row_key: series,
        Selected: selectedSet.has(series) && !deletedSet.has(series),
        Series: series,
        Benchmark: benchmarkValue,
        CMABench: explicitCmabench || defaultCmabench || "",
        LongShort: !!longShortMap[series],
        ScaleVol: Object.prototype.hasOwnProperty.call(volScalingMap, series) ? !!volScalingMap[series] : true,
        MinWt: minWtValue,
        MaxWt: maxWtValue,
        ForceMax: !!forceMaxMap[series],
        Delete: deletedSet.has(series)
      };
    });

    const columnDefs = portoptSeriesSelectionColumnDefs(benchmarkValues, cmabenchEditorValues);
    const columnDefsUpdate = currentColumnDefsJson === JSON.stringify(columnDefs) ? noUpdate() : columnDefs;
    const orderUpdate = JSON.stringify(normalizedOrder) === JSON.stringify(currentOrder) ? noUpdate() : normalizedOrder;
    return [rowData, columnDefsUpdate, orderUpdate, noUpdate()];
  }

  function portoptCollectModalSnapshotRows(rows, validSeriesSet) {
    const activeRows = [];
    const finalNames = [];
    const renameMap = {};
    const deletedOriginals = [];
    const deletedFinalNames = [];

    for (let idx = 0; idx < rows.length; idx += 1) {
      const row = rows[idx];
      const original = String((row && row.__row_key) || "").trim();
      if (!original || !validSeriesSet.has(original)) {
        continue;
      }
      const finalName = String((row && row.Series) || "").trim();
      if (!finalName || finalNames.indexOf(finalName) !== -1) {
        return { invalid: true };
      }
      finalNames.push(finalName);
      if (finalName !== original) {
        renameMap[original] = finalName;
      }
      activeRows.push([original, finalName, row]);
      if (!!row.Delete) {
        deletedOriginals.push(original);
        deletedFinalNames.push(finalName);
      }
    }

    return {
      invalid: false,
      activeRows: activeRows,
      renameMap: renameMap,
      deletedOriginals: deletedOriginals,
      deletedFinalNames: deletedFinalNames
    };
  }

  function applyPortoptSeriesSnapshot(
    snapshotData,
    rawMeta,
    currentSelect,
    currentBench,
    currentCmabench,
    currentLs,
    currentOrder,
    currentVolScaling,
    currentMinWt,
    currentMaxWt,
    currentForceMax
  ) {
    const nu = noUpdate();
    const noopResult = [nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu];
    const rows = (snapshotData && Array.isArray(snapshotData.rows))
      ? snapshotData.rows.filter(function (row) { return row && typeof row === "object"; }).map(function (row) {
        return Object.assign({}, row);
      })
      : [];
    const existingCols = rawMetaColumns(rawMeta);
    if (!rows.length || !existingCols.length) {
      return noopResult;
    }

    const existingSet = new Set(existingCols);
    const collected = portoptCollectModalSnapshotRows(rows, existingSet);
    if (!collected || collected.invalid) {
      return noopResult;
    }

    const activeRows = collected.activeRows;
    const renameMap = collected.renameMap;
    const deletedOriginals = collected.deletedOriginals;
    const deletedFinalNames = collected.deletedFinalNames;
    const structuralChange = Object.keys(renameMap).length > 0 || deletedFinalNames.length > 0;

    let remainingCols = existingCols.slice();
    if (structuralChange) {
      remainingCols = remainingCols.map(function (series) {
        return Object.prototype.hasOwnProperty.call(renameMap, series) ? renameMap[series] : series;
      }).filter(function (series) {
        return deletedFinalNames.indexOf(series) === -1;
      });
    }
    const remainingSet = new Set(remainingCols);

    const nextSelect = [];
    const nextBench = {};
    const nextCmabench = {};
    const nextLs = {};
    const nextOrder = [];
    const nextVolScaling = {};
    const nextMinWt = {};
    const nextMaxWt = {};
    const nextForceMax = {};

    for (let idx = 0; idx < activeRows.length; idx += 1) {
      const entry = activeRows[idx];
      const finalName = entry[1];
      const row = entry[2];
      if (!remainingSet.has(finalName)) {
        continue;
      }
      nextOrder.push(finalName);
      if (!!row.Selected) {
        nextSelect.push(finalName);
      }

      let benchmarkValue = String(row.Benchmark || "None").trim() || "None";
      benchmarkValue = Object.prototype.hasOwnProperty.call(renameMap, benchmarkValue) ? renameMap[benchmarkValue] : benchmarkValue;
      if (benchmarkValue !== "None" && !remainingSet.has(benchmarkValue)) {
        benchmarkValue = "None";
      }
      nextBench[finalName] = benchmarkValue;
      nextCmabench[finalName] = String(row.CMABench || "").trim();
      nextLs[finalName] = !!row.LongShort;
      nextVolScaling[finalName] = row.ScaleVol === undefined ? true : !!row.ScaleVol;

      let minWtValue = portoptCoerceFloat(row.MinWt);
      let maxWtValue = portoptCoerceFloat(row.MaxWt);
      minWtValue = minWtValue === null ? 0 : Math.max(0, Math.min(100, minWtValue));
      maxWtValue = maxWtValue === null ? 100 : Math.max(0, Math.min(100, maxWtValue));
      const forceMaxValue = !!row.ForceMax;
      if (forceMaxValue) {
        minWtValue = 0;
      }
      nextMinWt[finalName] = minWtValue;
      nextMaxWt[finalName] = maxWtValue;
      nextForceMax[finalName] = forceMaxValue;
    }

    const currentSelectList = Array.isArray(currentSelect) ? currentSelect.slice() : [];
    const currentBenchMap = (currentBench && typeof currentBench === "object") ? currentBench : {};
    const currentCmabenchMap = (currentCmabench && typeof currentCmabench === "object") ? currentCmabench : {};
    const currentLsMap = (currentLs && typeof currentLs === "object") ? currentLs : {};
    const currentOrderList = Array.isArray(currentOrder) ? currentOrder.slice() : [];
    const currentVolScalingMap = (currentVolScaling && typeof currentVolScaling === "object") ? currentVolScaling : {};
    const currentMinWtMap = (currentMinWt && typeof currentMinWt === "object") ? currentMinWt : {};
    const currentMaxWtMap = (currentMaxWt && typeof currentMaxWt === "object") ? currentMaxWt : {};
    const currentForceMaxMap = (currentForceMax && typeof currentForceMax === "object") ? currentForceMax : {};

    if (!structuralChange) {
      const nextSelectSet = new Set(nextSelect);
      const currentSelectSet = new Set(currentSelectList);
      if (nextSelect.length === currentSelectList.length) {
        let sameMembers = true;
        currentSelectSet.forEach(function (series) {
          if (!nextSelectSet.has(series)) {
            sameMembers = false;
          }
        });
        if (sameMembers) {
          nextSelect.splice(0, nextSelect.length, ...currentSelectList);
        }
      }
    }

    function stableValue(value) {
      if (Array.isArray(value)) {
        return value.map(stableValue);
      }
      if (value && typeof value === "object") {
        const out = {};
        Object.keys(value).sort().forEach(function (key) {
          out[key] = stableValue(value[key]);
        });
        return out;
      }
      return value;
    }

    function sameValue(left, right) {
      if (left === right) {
        return true;
      }
      const leftIsObject = left !== null && typeof left === "object";
      const rightIsObject = right !== null && typeof right === "object";
      if (leftIsObject || rightIsObject) {
        try {
          return JSON.stringify(stableValue(left)) === JSON.stringify(stableValue(right));
        } catch (_err) {
          return false;
        }
      }
      return false;
    }

    function resolvedOutput(nextValue, currentValue) {
      return sameValue(currentValue, nextValue) ? nu : nextValue;
    }

    if (structuralChange) {
      return [
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        nu,
        {
          kind: "structural",
          stamp: Date.now(),
          rows: rows,
          renameMap: renameMap,
          deletedOriginals: deletedOriginals,
          deletedFinalNames: deletedFinalNames
        }
      ];
    }

    const selectOutput = resolvedOutput(nextSelect, currentSelectList);
    const benchOutput = resolvedOutput(nextBench, currentBenchMap);
    const cmabenchOutput = resolvedOutput(nextCmabench, currentCmabenchMap);
    const lsOutput = resolvedOutput(nextLs, currentLsMap);
    const orderOutput = resolvedOutput(nextOrder, currentOrderList);
    const selectValueOutput = resolvedOutput(nextSelect, currentSelectList);
    const volScalingOutput = resolvedOutput(nextVolScaling, currentVolScalingMap);
    const minWtOutput = resolvedOutput(nextMinWt, currentMinWtMap);
    const maxWtOutput = resolvedOutput(nextMaxWt, currentMaxWtMap);
    const forceMaxOutput = resolvedOutput(nextForceMax, currentForceMaxMap);
    const changed = [
      selectOutput,
      benchOutput,
      cmabenchOutput,
      lsOutput,
      orderOutput,
      selectValueOutput,
      volScalingOutput,
      minWtOutput,
      maxWtOutput,
      forceMaxOutput
    ].some(function (value) { return value !== nu; });

    return [
      selectOutput,
      benchOutput,
      cmabenchOutput,
      lsOutput,
      orderOutput,
      changed ? false : false,
      selectValueOutput,
      volScalingOutput,
      minWtOutput,
      maxWtOutput,
      forceMaxOutput,
      nu
    ];
  }

  function openPortoptSeriesModal(
    nClicks,
    pathname,
    pageLoadIntervals,
    rawMeta,
    currentSelect,
    currentBench,
    currentCmabench,
    currentLs,
    currentOrder,
    currentVolScaling,
    currentMinWt,
    currentMaxWt,
    currentForceMax,
    poOriginSeries,
    pageVisited
  ) {
    const trigger = triggeredId();
    if (trigger !== "po-open-modal-button" && trigger !== "po-url-location") {
      if (trigger !== "dashmat-raw-data-meta-store" && trigger !== "po-page-load-trigger") {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
    }

    if (trigger === "po-page-load-trigger" && (pageLoadIntervals === null || pageLoadIntervals === undefined)) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    if (trigger === "po-open-modal-button") {
      if (!nClicks) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      return [
        true,
        currentSelect,
        currentBench,
        currentCmabench,
        currentLs,
        currentOrder,
        [],
        currentVolScaling,
        currentMinWt,
        currentMaxWt,
        currentForceMax,
        noUpdate(),
        true
      ];
    }

    const pagePath = String(pathname || "").split("?")[0].replace(/\/$/, "") || "/";
    if (pagePath !== "/portopt") {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      if (trigger === "po-url-location") {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
      ];
    }

    const columnSet = new Set(columns);
    const selected = resolveStoredList(currentSelect, "po-series-select");
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set(
      (selectedValid.length ? resolveStoredList(currentOrder, "po-series-order-store") : []).filter(function (series) {
        return columnSet.has(series);
      })
    );
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const poOriginSet = new Set(resolveStoredNames(poOriginSeries, "dashmat-pending-new-series-store").filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });

    let shouldOpen = false;
    let tempSelect = noUpdate();
    if (trigger === "dashmat-raw-data-meta-store") {
      if (!resolveStoredBool(pageVisited, "po-page-visited-store")) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      shouldOpen = genericNew.length > 0;
      if (shouldOpen) {
        const selectedSet = new Set(selectedValid);
        genericNew.forEach(function (series) {
          selectedSet.add(series);
        });
        tempSelect = columns.filter(function (series) {
          return selectedSet.has(series);
        });
      }
    } else {
      if (!resolveStoredBool(pageVisited, "po-page-visited-store") && !selectedValid.length) {
        tempSelect = columns.filter(function (series) {
          return !poOriginSet.has(series);
        });
        shouldOpen = tempSelect.length > 0;
      } else if (genericNew.length) {
        shouldOpen = true;
        const selectedSet = new Set(selectedValid);
        genericNew.forEach(function (series) {
          selectedSet.add(series);
        });
        tempSelect = columns.filter(function (series) {
          return selectedSet.has(series);
        });
      }
    }

    if (!shouldOpen) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
      ];
    }

    if (trigger === "po-page-load-trigger") {
      deferModalOpen("po-series-selection-modal");
      return [
        noUpdate(),
        tempSelect,
        currentBench,
        currentCmabench,
        currentLs,
        currentOrder,
        [],
        currentVolScaling,
        currentMinWt,
        currentMaxWt,
        currentForceMax,
        true,
        true
      ];
    }

    return [
      true,
      tempSelect,
      currentBench,
      currentCmabench,
      currentLs,
      currentOrder,
      [],
      currentVolScaling,
      currentMinWt,
      currentMaxWt,
      currentForceMax,
      true,
      true
    ];
  }

  function openAnalyticsSeriesModal(
    nClicks,
    pageLoadIntervals,
    rawMeta,
    pathname,
    currentSelect,
    currentBench,
    currentLs,
    currentOrder,
    currentVolScaling,
    poOriginSeries,
    pageVisited,
    modalOpened
  ) {
    const trigger = triggeredId();
    if (
      trigger !== "at-open-series-modal-button" &&
      trigger !== "at-page-load-trigger" &&
      trigger !== "dashmat-raw-data-meta-store"
    ) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    if (trigger === "at-open-series-modal-button") {
      if (!nClicks) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      return [
        true,
        currentSelect,
        currentBench,
        currentLs,
        currentOrder,
        [],
        currentVolScaling,
        noUpdate(),
        true
      ];
    }

    if (
      trigger === "at-page-load-trigger" &&
      (pageLoadIntervals === null || pageLoadIntervals === undefined)
    ) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const pagePath = String(pathname || "").split("?")[0].replace(/\/$/, "") || "/";
    if (pagePath !== "/analyticstool") {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      if (trigger === "dashmat-raw-data-meta-store") {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), false
      ];
    }

    const columnSet = new Set(columns);
    const selected = Array.isArray(currentSelect) ? currentSelect : [];
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set((Array.isArray(currentOrder) ? currentOrder : []).filter(function (series) {
      return columnSet.has(series);
    }));
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const poOriginSet = new Set(storeNames(poOriginSeries).filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });
    const poNew = columns.filter(function (series) {
      return !knownColumns.has(series) && poOriginSet.has(series);
    });

    let shouldOpen = false;
    let tempSelect = noUpdate();
    if (trigger === "dashmat-raw-data-meta-store") {
      if (modalOpened === true) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      if (!pageVisited && !selectedValid.length) {
        shouldOpen = true;
        tempSelect = columns.slice();
      } else if (pageVisited && genericNew.length > 0) {
        shouldOpen = true;
        const selectedSet = new Set(selectedValid);
        genericNew.forEach(function (series) {
          selectedSet.add(series);
        });
        poNew.forEach(function (series) {
          selectedSet.add(series);
        });
        tempSelect = columns.filter(function (series) {
          return selectedSet.has(series);
        });
      }
    } else if (!pageVisited && !selectedValid.length) {
      shouldOpen = true;
      tempSelect = columns.slice();
    } else if (genericNew.length) {
      shouldOpen = true;
      const selectedSet = new Set(selectedValid);
      genericNew.forEach(function (series) {
        selectedSet.add(series);
      });
      poNew.forEach(function (series) {
        selectedSet.add(series);
      });
      tempSelect = columns.filter(function (series) {
        return selectedSet.has(series);
      });
    }

    if (!shouldOpen) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
      ];
    }

    return [
      true,
      tempSelect,
      currentBench,
      currentLs,
      currentOrder,
      [],
      currentVolScaling,
      true,
      true
    ];
  }

  function openRegressionSeriesModal(
    nClicks,
    rawMeta,
    pageLoadIntervals,
    pathname,
    currentSelect,
    currentOrder,
    currentBench,
    currentLs,
    currentVolScaling,
    currentDepVar,
    currentLag,
    currentMinBeta,
    currentMaxBeta,
    currentEnable,
    poOriginSeries,
    pageVisited
  ) {
    const trigger = triggeredId();
    if (
      trigger !== "reg-open-modal-button" &&
      trigger !== "dashmat-raw-data-meta-store" &&
      trigger !== "reg-page-load-trigger"
    ) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    if (trigger === "reg-open-modal-button") {
      if (!nClicks) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      return [
        true,
        currentSelect,
        currentOrder,
        [],
        currentBench,
        currentLs,
        currentVolScaling,
        currentDepVar,
        currentLag,
        currentMinBeta,
        currentMaxBeta,
        currentEnable,
        noUpdate(),
        true
      ];
    }

    if (trigger === "reg-page-load-trigger" && (pageLoadIntervals === null || pageLoadIntervals === undefined)) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const pagePath = String(pathname || "").split("?")[0].replace(/\/$/, "") || "/";
    if (pagePath !== "/regression") {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      if (trigger === "reg-page-load-trigger") {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
        ];
      }
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    const columnSet = new Set(columns);
    const selected = resolveStoredList(currentSelect, "reg-series-select");
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set(resolveStoredList(currentOrder, "reg-series-order-store").filter(function (series) {
      return columnSet.has(series);
    }));
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const effectiveDepVar = resolveStoredString(currentDepVar, "reg-dependent-var-store");
    if (typeof effectiveDepVar === "string" && columnSet.has(effectiveDepVar)) {
      knownColumns.add(effectiveDepVar);
    }
    const poOriginSet = new Set(resolveStoredNames(poOriginSeries, "dashmat-pending-new-series-store").filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });

    let shouldOpen = false;
    let tempSelect = selected.slice();
    if (trigger === "dashmat-raw-data-meta-store") {
      if (!resolveStoredBool(pageVisited, "reg-page-visited-store")) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
        ];
      }
      shouldOpen = genericNew.length > 0;
      if (shouldOpen) {
        const selectedSet = new Set(selectedValid);
        genericNew.forEach(function (series) {
          selectedSet.add(series);
        });
        tempSelect = columns.filter(function (series) {
          return selectedSet.has(series);
        });
      }
    } else {
      if (!resolveStoredBool(pageVisited, "reg-page-visited-store") && !selectedValid.length) {
        tempSelect = columns.filter(function (series) {
          return !poOriginSet.has(series);
        });
        shouldOpen = tempSelect.length > 0;
      } else if (genericNew.length) {
        shouldOpen = true;
        const selectedSet = new Set(selectedValid);
        genericNew.forEach(function (series) {
          selectedSet.add(series);
        });
        tempSelect = columns.filter(function (series) {
          return selectedSet.has(series);
        });
      }

      if (!shouldOpen) {
        return [
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
          noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
        ];
      }
    }

    if (!shouldOpen) {
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(),
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate()
      ];
    }

    return [
      true,
      tempSelect,
      currentOrder || [],
      [],
      currentBench || {},
      currentLs || {},
      currentVolScaling || {},
      effectiveDepVar,
      currentLag || {},
      currentMinBeta || {},
      currentMaxBeta || {},
      currentEnable || {},
      trigger === "reg-page-load-trigger" ? true : noUpdate(),
      true
    ];
  }

  function regressionInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, currentDepVar, poOriginSeries, pageVisited) {
    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      return false;
    }
    const columnSet = new Set(columns);
    const selected = resolveStoredList(currentSelect, "reg-series-select");
    const selectedValid = selected.filter(function (series) {
      return columnSet.has(series);
    });
    const knownColumns = new Set(resolveStoredList(currentOrder, "reg-series-order-store").filter(function (series) {
      return columnSet.has(series);
    }));
    selectedValid.forEach(function (series) {
      knownColumns.add(series);
    });
    const effectiveDepVar = resolveStoredString(currentDepVar, "reg-dependent-var-store");
    if (typeof effectiveDepVar === "string" && columnSet.has(effectiveDepVar)) {
      knownColumns.add(effectiveDepVar);
    }
    const poOriginSet = new Set(resolveStoredNames(poOriginSeries, "dashmat-pending-new-series-store").filter(function (series) {
      return columnSet.has(series);
    }));
    const genericNew = columns.filter(function (series) {
      return !knownColumns.has(series) && !poOriginSet.has(series);
    });
    if (!resolveStoredBool(pageVisited, "reg-page-visited-store") && !selectedValid.length) {
      return columns.some(function (series) {
        return !poOriginSet.has(series);
      });
    }
    return genericNew.length > 0;
  }

  function regressionInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, pageVisited, currentOrder, currentDepVar, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      regressionInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, currentDepVar, poOriginSeries, pageVisited),
      virtualRows,
      "/regression"
    );
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

  function analyticsControlSync(periodicity, returnsType, volScaler, seriesSelect, activeTab, rollingWindow, rollingMetric, rollingReturnType, monthlyView, monthlySeries, useRiskFree) {
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
      monthlySeries,
      useRiskFree === "zero" ? false : true
    ];
  }

  function analyticsSyncReturnsTypeMirrors(
    currentValue,
    returnsValue,
    calendarValue,
    drawdownValue,
    correlogramValue,
    factorValue,
    conditionalValue,
    regimeValue
  ) {
    const normalized = currentValue === "excess" ? "excess" : "total";
    function sync(value) {
      return value === normalized ? noUpdate() : normalized;
    }
    return [
      sync(returnsValue),
      sync(calendarValue),
      sync(drawdownValue),
      sync(correlogramValue),
      sync(factorValue),
      sync(conditionalValue),
      sync(regimeValue)
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

  function analyticsTabTrigger(targetTab, activeTab, initialTabReady, stateReady) {
    if (String(activeTab || "statistics") !== String(targetTab || "")) {
      return noUpdate();
    }
    if (initialTabReady === false || !stateReady) {
      return noUpdate();
    }
    return {
      tab: String(targetTab || ""),
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsBootstrapCandidateTrigger(datasetKey, periodicity, selectedSeries, candidates, stateReady) {
    const hasDataset = typeof datasetKey === "string" && datasetKey.length > 0;
    const hasPeriodicity = typeof periodicity === "string" && periodicity.length > 0;
    const seriesList = Array.isArray(selectedSeries) ? selectedSeries.filter(Boolean) : [];
    const candidateSeries = candidates && Array.isArray(candidates.available_series)
      ? candidates.available_series.filter(Boolean)
      : [];
    if (stateReady || !hasDataset || !hasPeriodicity || !seriesList.length || candidateSeries.length) {
      return noUpdate();
    }

    return {
      phase: "bootstrap",
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsCandidateRefreshTrigger(activeTab, stateReady, datasetKey, selectedSeries) {
    const hasDataset = typeof datasetKey === "string" && datasetKey.length > 0;
    const seriesList = Array.isArray(selectedSeries) ? selectedSeries.filter(Boolean) : [];
    if (!hasDataset || !seriesList.length) {
      return noUpdate();
    }

    if (!stateReady || String(activeTab || "statistics") !== "correlogram") {
      return noUpdate();
    }

    return {
      tab: "correlogram",
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsModalPreviewTrigger(opened) {
    if (!opened) {
      return noUpdate();
    }
    return {
      opened: true,
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsDownloadExcelDisabled(rawData, selectedSeries, dateRange, stateReady) {
    if (!rawData) {
      return true;
    }
    if (!selectedSeries || !selectedSeries.length) {
      return true;
    }
    if (!stateReady) {
      return true;
    }
    return !(dateRange && dateRange.start && dateRange.end);
  }

  function analyticsFactorRegimeSync(factorMode, factorQuantiles, factorTransform, factorSeries, factorQqReference, regimeDefinition, regimeMethodType) {
    let quantiles = 5;
    if (factorQuantiles !== null && factorQuantiles !== undefined) {
      const parsed = parseInt(factorQuantiles, 10);
      if (Number.isFinite(parsed)) {
        quantiles = Math.min(20, Math.max(2, parsed));
      }
    }
    const mode = factorMode || "box";
    const qqReference = factorQqReference === "reference" ? "reference" : "normal";
    const method = String(regimeMethodType || "1");
    const showFactorControls = mode === "box" || mode === "scatter" || mode === "detail" || (mode === "qq" && qqReference === "reference");
    const quantileStyle = mode === "box" || mode === "detail" ? { display: "block" } : { display: "none" };
    const transformStyle = mode === "box" || mode === "scatter" || mode === "detail" ? { display: "block" } : { display: "none" };
    const qqReferenceStyle = mode === "qq" ? { display: "block" } : { display: "none" };
    const factorStyle = showFactorControls ? { display: "block" } : { display: "none" };
    return [
      mode,
      quantiles,
      factorTransform === "zscore" ? "zscore" : "raw",
      factorSeries,
      qqReference,
      factorStyle,
      factorStyle,
      quantileStyle,
      transformStyle,
      qqReferenceStyle,
      mode === "qq" ? "Reference" : "Factor",
      regimeDefinition,
      method === "3" ? { display: "none" } : { display: "block" },
      method === "3" ? { display: "block" } : { display: "none" }
    ];
  }

  function regressionControlSync(periodicity, volScaler, model, name, forceZero, robustSe, expWt, halflife, windowType, windowSize, optStep, optStepUnit, fillInSample, missingData, alpha, l1Ratio, activeTab, useRiskFree) {
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
      activeTab,
      useRiskFree === "zero" ? false : true
    ];
  }

  function defaultPortoptLoadedTabs() {
    return {
      weight: false,
      attribution: false,
      risk: false,
      frontier: false
    };
  }

  function portoptBootstrapRestore(
    pageLoadIntervals,
    rawMeta,
    storedPeriodicity,
    storedVolScaler,
    storedSeries,
    storedActiveTab,
    storedWeightView,
    storedAttributionView,
    storedRiskView,
    storedTurnoverView,
    storedFrontierView,
    storedOptWindow,
    storedWindowSize,
    storedOptStep,
    storedOptStepUnit,
    storedOptModel,
    storedPortfolioName,
    storedExpWtCov,
    storedHalflife,
    storedCovShrinkage,
    storedCovShrinkageTarget,
    storedMissingData,
    storedFillInSample,
    storedExAnteMode,
    storedObjective,
    storedUseRiskFree,
    storedReturnsBasis,
    storedReportingBasis,
    currentPeriodicityOptions,
    currentPeriodicity,
    currentVolScaler,
    currentSeries,
    currentActiveTab,
    currentWeightView,
    currentAttributionView,
    currentRiskView,
    currentTurnoverView,
    currentFrontierView,
    currentOptWindow,
    currentWindowSize,
    currentOptStep,
    currentOptStepUnit,
    currentOptModel,
    currentReturnsBasis,
    currentReportingBasis
  ) {
    const nu = noUpdate();
    const idleState = { phase: "idle", loadedTabs: defaultPortoptLoadedTabs() };
    if (!pageLoadIntervals || !rawMeta || rawMeta.has_data !== true) {
      return [
        nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu,
        nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, idleState
      ];
    }

    const columns = rawMetaColumns(rawMeta);
    if (!columns.length) {
      return [
        nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu,
        nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, nu, idleState
      ];
    }

    const periodicityOptions = Array.isArray(rawMeta.periodicity_options) && rawMeta.periodicity_options.length
      ? rawMeta.periodicity_options.slice()
      : [{ value: "daily_trading", label: "Daily (Trading)" }];
    const periodicityValues = periodicityOptions
      .map(function (option) { return option && option.value; })
      .filter(function (value) { return typeof value === "string" && value.length; });
    const fallbackPeriodicity = rawMeta.original_periodicity === "daily"
      ? "daily_trading"
      : (rawMeta.original_periodicity || periodicityValues[0] || "daily_trading");
    const resolvedPeriodicity = periodicityValues.indexOf(storedPeriodicity) !== -1
      ? storedPeriodicity
      : (periodicityValues.indexOf(fallbackPeriodicity) !== -1 ? fallbackPeriodicity : (periodicityValues[0] || "daily_trading"));

    const columnSet = new Set(columns);
    const selectedSeries = Array.isArray(storedSeries)
      ? storedSeries.filter(function (series) { return columnSet.has(series); })
      : [];
    const resolvedVolScaler = storedVolScaler !== null && storedVolScaler !== undefined && Number.isFinite(Number(storedVolScaler))
      ? Number(storedVolScaler)
      : 0;

    const allowedTabs = ["weight", "attribution", "risk", "turnover", "frontier", "statistics", "returns", "rolling", "calendar", "growth", "drawdown"];
    const resolvedActiveTab = allowedTabs.indexOf(storedActiveTab) !== -1 ? storedActiveTab : "weight";

    function normalizeView(viewMode) {
      return viewMode === "table" ? "table" : "chart";
    }

    function sameValue(left, right) {
      if (left === right) {
        return true;
      }
      const leftIsObject = left !== null && typeof left === "object";
      const rightIsObject = right !== null && typeof right === "object";
      if (leftIsObject || rightIsObject) {
        try {
          return JSON.stringify(left) === JSON.stringify(right);
        } catch (_err) {
          return false;
        }
      }
      return false;
    }

    function resolvedOutput(nextValue, currentValue) {
      return sameValue(currentValue, nextValue) ? nu : nextValue;
    }

    const validWindowTypes = ["rolling", "expanding", "full"];
    const resolvedOptWindow = validWindowTypes.indexOf(storedOptWindow) !== -1 ? storedOptWindow : "rolling";
    const resolvedWindowSize = storedWindowSize !== null && storedWindowSize !== undefined && Number.isFinite(Number(storedWindowSize)) && Number(storedWindowSize) >= 2
      ? Number(storedWindowSize)
      : 252;
    const resolvedOptStep = storedOptStep !== null && storedOptStep !== undefined && Number.isFinite(Number(storedOptStep)) && Number(storedOptStep) >= 1
      ? Number(storedOptStep)
      : 1;

    const validStepUnits = ["months", "periods"];
    const resolvedOptStepUnit = validStepUnits.indexOf(storedOptStepUnit) !== -1 ? storedOptStepUnit : "months";

    const validModels = [
      "risk_parity",
      "factor_risk_parity",
      "hierarchical_risk_parity",
      "hrp",
      "maximize_sharpe",
      "minimize_variance",
      "minimize_cvar",
      "equal_weight",
      "ex_ante_mv",
      "black_litterman"
    ];
    const resolvedOptModel = validModels.indexOf(storedOptModel) !== -1 ? storedOptModel : "risk_parity";
    const trimmedName = typeof storedPortfolioName === "string" ? storedPortfolioName.trim() : "";
    const resolvedPortfolioName = trimmedName || portoptModelDefaultName(resolvedOptModel);

    const expWeighted = !!storedExpWtCov;
    const resolvedHalflife = storedHalflife !== null && storedHalflife !== undefined && Number.isFinite(Number(storedHalflife)) && Number(storedHalflife) > 0
      ? Number(storedHalflife)
      : 63;

    const validShrinkage = ["none", "ledoit_wolf", "oas"];
    const resolvedCovShrinkage = validShrinkage.indexOf(storedCovShrinkage) !== -1 ? storedCovShrinkage : "none";
    const validShrinkageTargets = ["scaled_identity", "constant_correlation"];
    const resolvedCovShrinkageTarget = validShrinkageTargets.indexOf(storedCovShrinkageTarget) !== -1
      ? storedCovShrinkageTarget
      : "scaled_identity";
    const halflifeDisabled = !expWeighted;
    const covShrinkageTargetDisabled = expWeighted || resolvedCovShrinkage !== "ledoit_wolf";

    const validMissingData = ["fill_na", "fill_0"];
    const resolvedMissingData = validMissingData.indexOf(storedMissingData) !== -1 ? storedMissingData : "fill_na";
    const validFillInSample = ["off", "on"];
    const resolvedFillInSample = validFillInSample.indexOf(storedFillInSample) !== -1 ? storedFillInSample : "off";

    const resolvedExAnteMode = storedExAnteMode === "ret_cov" ? "ret_cov" : "ret_vol_corr";
    const validObjectives = ["maximize_sharpe", "minimize_variance", "minimize_cvar"];
    const resolvedObjective = validObjectives.indexOf(storedObjective) !== -1 ? storedObjective : "maximize_sharpe";
    const resolvedUseRiskFree = storedUseRiskFree === false ? "zero" : "tbill";
    const resolvedReturnsBasis = storedReturnsBasis === "excess" ? "excess" : "total";
    const resolvedReportingBasis = storedReportingBasis ? "split" : "match";

    const loadedTabs = defaultPortoptLoadedTabs();
    if (Object.prototype.hasOwnProperty.call(loadedTabs, resolvedActiveTab)) {
      loadedTabs[resolvedActiveTab] = true;
    }

    const resolvedWeightView = normalizeView(storedWeightView);
    const resolvedAttributionView = normalizeView(storedAttributionView);
    const resolvedRiskView = normalizeView(storedRiskView);
    const resolvedTurnoverView = normalizeView(storedTurnoverView);
    const resolvedFrontierView = normalizeView(storedFrontierView);

    return [
      resolvedOutput(periodicityOptions, currentPeriodicityOptions),
      resolvedOutput(resolvedPeriodicity, currentPeriodicity),
      resolvedOutput(resolvedVolScaler, currentVolScaler),
      resolvedOutput(selectedSeries, currentSeries),
      resolvedOutput(resolvedActiveTab, currentActiveTab),
      resolvedOutput(resolvedWeightView, currentWeightView),
      resolvedOutput(resolvedAttributionView, currentAttributionView),
      resolvedOutput(resolvedRiskView, currentRiskView),
      resolvedOutput(resolvedTurnoverView, currentTurnoverView),
      resolvedOutput(resolvedFrontierView, currentFrontierView),
      resolvedOutput(resolvedOptWindow, currentOptWindow),
      resolvedOutput(resolvedWindowSize, currentWindowSize),
      resolvedOutput(resolvedOptStep, currentOptStep),
      resolvedOutput(resolvedOptStepUnit, currentOptStepUnit),
      resolvedOutput(resolvedOptModel, currentOptModel),
      resolvedPortfolioName,
      expWeighted,
      resolvedHalflife,
      resolvedCovShrinkage,
      resolvedCovShrinkageTarget,
      halflifeDisabled,
      covShrinkageTargetDisabled,
      resolvedMissingData,
      resolvedFillInSample,
      resolvedExAnteMode,
      resolvedObjective,
      resolvedUseRiskFree,
      resolvedOutput(resolvedReturnsBasis, currentReturnsBasis),
      resolvedOutput(resolvedReportingBasis, currentReportingBasis),
      { phase: "ready", loadedTabs: loadedTabs }
    ];
  }

  function portoptMarkVisitedTabLoaded(activeTab, bootstrapState) {
    if (!bootstrapState || bootstrapState.phase !== "ready" || !bootstrapState.loadedTabs) {
      return noUpdate();
    }
    if (!Object.prototype.hasOwnProperty.call(bootstrapState.loadedTabs, activeTab)) {
      return noUpdate();
    }
    if (bootstrapState.loadedTabs[activeTab]) {
      return noUpdate();
    }
    return {
      phase: "ready",
      loadedTabs: Object.assign({}, bootstrapState.loadedTabs, { [activeTab]: true })
    };
  }

  function portoptControlSync(
    periodicity,
    volScaler,
    activeTab,
    seriesSelect,
    fillInSample,
    optStepUnit,
    optWindow,
    windowSize,
    optStep,
    optModel,
    portfolioName,
    expWtCov,
    halflife,
    covShrinkage,
    covShrinkageTarget,
    missingData,
    objective,
    blTau,
    exAnteMode,
    useRiskFree,
    returnsBasis,
    reportingBasis,
    currentSeriesSelectStore,
    currentBlTauStore,
    currentExAnteModeStore,
    currentUseRiskFreeStore,
    currentReturnsBasisStore,
    currentReportingBasisStore
  ) {
    const nu = noUpdate();

    function sameValue(left, right) {
      if (left === right) {
        return true;
      }
      const leftIsObject = left !== null && typeof left === "object";
      const rightIsObject = right !== null && typeof right === "object";
      if (leftIsObject || rightIsObject) {
        try {
          return JSON.stringify(left) === JSON.stringify(right);
        } catch (_err) {
          return false;
        }
      }
      return false;
    }

    function resolvedOutput(nextValue, currentValue) {
      return sameValue(currentValue, nextValue) ? nu : nextValue;
    }

    return [
      periodicity,
      volScaler,
      activeTab || "weight",
      resolvedOutput(seriesSelect || [], currentSeriesSelectStore),
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
      resolvedOutput(blTau, currentBlTauStore),
      resolvedOutput(exAnteMode || "ret_cov", currentExAnteModeStore),
      resolvedOutput(useRiskFree === "zero" ? false : true, currentUseRiskFreeStore),
      resolvedOutput(returnsBasis === "excess" ? "excess" : "total", currentReturnsBasisStore),
      resolvedOutput(reportingBasis === "split", currentReportingBasisStore)
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

  function portoptToggleRawDivideBy(mode, convertToReturns, opened) {
    if (!opened) {
      return noUpdate();
    }
    const modeKey = String(mode || "").trim().toLowerCase();
    return !(modeKey === "factor" && !Boolean(convertToReturns));
  }

  function portoptSyncReturnsBasisFromMirrors(returnsValue, calendarValue, drawdownValue, currentValue) {
    const valueByTrigger = {
      "po-returns-basis-control-returns": returnsValue,
      "po-returns-basis-control-calendar": calendarValue,
      "po-returns-basis-control-drawdown": drawdownValue
    };
    const nextValue = valueByTrigger[triggeredId()];
    if (nextValue === null || nextValue === undefined) {
      return noUpdate();
    }
    const normalized = nextValue === "excess" ? "excess" : "total";
    const currentNormalized = currentValue === "excess" ? "excess" : "total";
    return normalized === currentNormalized ? noUpdate() : normalized;
  }

  function portoptSyncReturnsBasisMirrors(currentValue, returnsValue, calendarValue, drawdownValue) {
    const normalized = currentValue === "excess" ? "excess" : "total";
    function syncValue(value) {
      return value === normalized ? noUpdate() : normalized;
    }
    return [syncValue(returnsValue), syncValue(calendarValue), syncValue(drawdownValue)];
  }

  function portoptSyncNameWithModel(model) {
    return portoptModelDefaultName(model);
  }

  function portoptClearReturns(nClicks, selectedSeries) {
    const nu = noUpdate();
    if (!nClicks) {
      return [nu, nu, nu];
    }
    const rows = (selectedSeries || []).map(function (series) {
      return { Asset: series, Return: 0.0, Volatility: 0.0 };
    });
    return [rows, {}, {}];
  }

  function portoptUpdateMatrixUi(mode) {
    const resolvedMode = mode || "ret_cov";
    if (resolvedMode === "ret_vol_corr") {
      return ["Correlation Matrix", "Upload Corr CSV", "Expected Returns and Volatility"];
    }
    return ["Covariance Matrix", "Upload Cov CSV", "Expected Returns"];
  }

  function portoptSyncTau(value) {
    return value || 0.05;
  }

  function portoptInitTau(storeValue) {
    if (storeValue === null || storeValue === undefined) {
      return noUpdate();
    }
    return storeValue;
  }

  function portoptInitLinearConstraintsGrid(storeData, currentRows) {
    if (storeData === null || storeData === undefined) {
      return noUpdate();
    }
    return sameValue(currentRows, storeData) ? noUpdate() : storeData;
  }

  function portoptUpdateOptStepOnUnitChange(unit, periodicity, storedStep) {
    const defaults = periodicityDefaults(periodicity);
    const stepDefault = unit === "months" ? defaults[2] : defaults[1];
    if (storedStep !== null && storedStep !== undefined && storedStep !== stepDefault) {
      return storedStep;
    }
    return stepDefault;
  }

  function portoptUpdateDateRangeStore(start, end) {
    if (start && end) {
      return { start: start, end: end };
    }
    return noUpdate();
  }

  function portoptToggleRollingReturnType(metric) {
    const disabled = ["total_return", "excess_return"].indexOf(metric || "total_return") === -1;
    return [disabled, disabled ? { opacity: 0.5, pointerEvents: "none" } : {}];
  }

  function portoptClearRawDbRows(nClear) {
    const nu = noUpdate();
    if (!nClear) {
      return [nu, nu, nu, nu];
    }
    return [[], [], nu, true];
  }

  function regressionModelSelectSync(model, current) {
    const show = { display: "block" };
    const hide = hiddenStyle;
    const resolvedModel = model || "ols";
    const checked = !triggeredId()
      ? noUpdate()
      : (resolvedModel === "style_analysis" ? true : current);
    return [
      (resolvedModel === "ols" || resolvedModel === "constrained_ols") ? show : hide,
      (resolvedModel === "ridge" || resolvedModel === "lasso" || resolvedModel === "elastic_net") ? show : hide,
      resolvedModel === "elastic_net" ? show : hide,
      resolvedModel === "style_analysis",
      checked,
      regressionModelDefaultName(resolvedModel)
    ];
  }

  function regressionToggleWindowControls(windowType) {
    const isFull = windowType === "full";
    return [isFull, isFull, isFull];
  }

  function regressionToggleRollingReturnType(metric) {
    const disabled = (metric || "total_return") !== "total_return";
    return [disabled, disabled ? { opacity: 0.5, pointerEvents: "none" } : {}];
  }

  function regressionSyncCalendarControls(
    triggerPayload,
    selected,
    calendarView,
    partialMode,
    activeEntry,
    currentDisabled,
    currentOptions,
    currentValue,
    currentSignature
  ) {
    const nu = noUpdate();
    if (!triggerPayload || typeof triggerPayload !== "object" || String(triggerPayload.tab || "") !== "calendar") {
      return [nu, nu, nu, nu];
    }

    const orderedCols = activeEntry && typeof activeEntry === "object" && Array.isArray(activeEntry.display_columns)
      ? activeEntry.display_columns.slice()
      : [];
    const nextOptions = orderedCols.map(function (name) {
      return { value: name, label: name };
    });

    let nextDisabled = true;
    let nextValue = null;
    if (String(calendarView || "annual") === "monthly" && orderedCols.length) {
      nextDisabled = false;
      nextValue = orderedCols.indexOf(currentValue) !== -1 ? currentValue : orderedCols[0];
    }

    const nextSignature = {
      tab: "calendar",
      selected: selected || null,
      view: calendarView || "annual",
      series: nextValue || null,
      partialMode: partialMode || "partial"
    };

    return [
      currentDisabled === nextDisabled ? nu : nextDisabled,
      sameValue(currentOptions, nextOptions) ? nu : nextOptions,
      currentValue === nextValue ? nu : nextValue,
      sameValue(currentSignature, nextSignature) ? nu : nextSignature
    ];
  }

  function regressionClearRawDbRows(nClear) {
    const nu = noUpdate();
    if (!nClear) {
      return [nu, nu, nu, nu];
    }
    return [[], [], nu, true];
  }

  function regressionDeleteRawDbRow(nDelete, stagedRows, selectedRows) {
    const nu = noUpdate();
    if (!nDelete) {
      return [nu, nu, nu, nu];
    }
    const rows = (stagedRows || []).filter(function (row) {
      return row && typeof row === "object";
    }).map(function (row) {
      return Object.assign({}, row);
    });
    if (!rows.length) {
      return [rows, rows, "No staged rows to delete.", false];
    }
    const selected = selectedRows || [];
    if (!selected.length) {
      return [rows, rows, "Select one staged row to delete.", false];
    }
    const selectedId = String(((selected[0] || {}).row_id) || "").trim();
    if (!selectedId) {
      return [rows, rows, "Select one staged row to delete.", false];
    }
    const kept = rows.filter(function (row) {
      return String((row && row.row_id) || "").trim() !== selectedId;
    });
    return [kept, kept, nu, true];
  }

  function regressionToggleSheetSelectDisabled(selectedSheets) {
    return !pythonTruthy(selectedSheets);
  }

  function regressionToggleFileMenuActions(rawData, results) {
    return [!pythonTruthy(rawData), !pythonTruthy(results)];
  }

  function regressionSyncSaveSeriesUi(selected, results, currentDisabled, currentStatus) {
    const entry = (selected && results && results[selected]) ? results[selected] : null;
    let nextDisabled = true;
    let nextStatus = "";
    if (entry) {
      const savedName = entry.saved_series_name;
      nextDisabled = false;
      nextStatus = savedName ? "Saved as " + savedName + "." : "";
    }
    return [
      currentDisabled === nextDisabled ? noUpdate() : nextDisabled,
      currentStatus === nextStatus ? noUpdate() : nextStatus
    ];
  }

  function portoptSyncSaveSeriesUi(selected, results, currentDisabled, currentStatus) {
    return regressionSyncSaveSeriesUi(selected, results, currentDisabled, currentStatus);
  }

  function portoptFrontierRiskMeasureOptions(selectedPortfolio, results, currentRm) {
    const allOptions = [
      { value: "MV", label: "Volatility" },
      { value: "CVaR", label: "CVaR" }
    ];
    const current = currentRm === "CVaR" ? "CVaR" : "MV";
    const entry = (selectedPortfolio && results && results[selectedPortfolio]) ? results[selectedPortfolio] : null;
    const model = String((((entry || {}).config) || {}).model || "");
    if (model === "ex_ante_mv" || model === "black_litterman") {
      return [[{ value: "MV", label: "Volatility" }], "MV"];
    }
    return [allOptions, current];
  }

  function portoptFrontierWindowOptions(selectedPortfolio, results, activeTab) {
    if (String(activeTab || "") !== "frontier" || !selectedPortfolio || !results) {
      return [[], null, false];
    }
    const entry = results[selectedPortfolio];
    if (!entry || typeof entry !== "object") {
      return [[], null, false];
    }
    const windowWeights = Array.isArray(entry.window_weights) ? entry.window_weights : [];
    const model = String((((entry || {}).config) || {}).model || "");
    if (!windowWeights.length) {
      return [[], null, false];
    }
    const options = windowWeights.map(function (ww, idx) {
      const estStart = String((((ww || {}).est_start) || ((ww || {}).apply_start) || "")).slice(0, 10);
      const estEnd = String((((ww || {}).est_end) || ((ww || {}).apply_end) || "")).slice(0, 10);
      return {
        value: String(idx),
        label: estStart + " - " + estEnd
      };
    });
    const disabled = model === "ex_ante_mv" || model === "black_litterman";
    return [options, String(windowWeights.length - 1), disabled];
  }

  function portoptSyncFrontierControls(
    triggerPayload,
    activeTab,
    selectedPortfolio,
    currentRmOptions,
    currentRm,
    currentWindowOptions,
    currentWindow,
    currentWindowDisabled,
    chartSwitch,
    results,
    resultsMeta,
    currentSignature
  ) {
    const nu = noUpdate();
    if (!triggerPayload || typeof triggerPayload !== "object" || String(triggerPayload.tab || "") !== "frontier") {
      return [nu, nu, nu, nu, nu, nu];
    }

    const frontierView = chartSwitch === "table" ? "table" : "chart";
    const pair = portoptFrontierRiskMeasureOptions(selectedPortfolio, results, currentRm);
    const nextRmOptions = pair[0];
    const nextRm = pair[1];
    const windowState = portoptFrontierWindowOptions(selectedPortfolio, results, activeTab);
    const nextWindowOptions = windowState[0];
    const nextWindow = windowState[1];
    const nextWindowDisabled = windowState[2];

    let signatureMeta = null;
    if (resultsMeta && typeof resultsMeta === "object") {
      if (selectedPortfolio && Object.prototype.hasOwnProperty.call(resultsMeta, selectedPortfolio)) {
        signatureMeta = resultsMeta[selectedPortfolio];
      } else {
        signatureMeta = resultsMeta;
      }
    }
    const nextSignature = {
      tab: "frontier",
      portfolio: selectedPortfolio || null,
      rm: nextRm || null,
      window: nextWindow || null,
      view: frontierView,
      resultsMeta: signatureMeta
    };

    return [
      sameValue(currentRmOptions, nextRmOptions) ? nu : nextRmOptions,
      currentRm === nextRm ? nu : nextRm,
      sameValue(currentWindowOptions, nextWindowOptions) ? nu : nextWindowOptions,
      currentWindow === nextWindow ? nu : nextWindow,
      currentWindowDisabled === nextWindowDisabled ? nu : nextWindowDisabled,
      sameValue(currentSignature, nextSignature) ? nu : nextSignature
    ];
  }

  function portoptSyncCalendarControls(
    triggerPayload,
    selectedPortfolio,
    periodicity,
    viewMode,
    currentDisabled,
    currentOptions,
    currentValue,
    returnsBasis,
    partialMode,
    rawDataIdentity,
    resultsMeta,
    benchmarkAssignments,
    longShortAssignments,
    dateRange,
    volScaler,
    volScalingAssignments,
    activeEntry,
    currentSignature
  ) {
    const nu = noUpdate();
    if (!triggerPayload || typeof triggerPayload !== "object" || String(triggerPayload.tab || "") !== "calendar") {
      return [nu, nu, nu, nu];
    }

    let nextDisabled = true;
    let nextOptions = [];
    let nextValue = null;
    if (selectedPortfolio && activeEntry && typeof activeEntry === "object") {
      const runInputs = activeEntry.run_inputs && typeof activeEntry.run_inputs === "object" ? activeEntry.run_inputs : {};
      const config = activeEntry.config && typeof activeEntry.config === "object" ? activeEntry.config : {};
      const orderedCols = [selectedPortfolio];
      const selectedSeries = Array.isArray(runInputs.selected_series)
        ? runInputs.selected_series
        : (Array.isArray(config.selected_series) ? config.selected_series : []);
      selectedSeries.forEach(function(name) {
        if (name && orderedCols.indexOf(name) === -1) {
          orderedCols.push(name);
        }
      });
      nextOptions = orderedCols.map(function(name) {
        return { value: name, label: name };
      });
      if (String(viewMode || "annual") === "monthly" && orderedCols.length) {
        nextDisabled = false;
        nextValue = orderedCols.indexOf(currentValue) !== -1 ? currentValue : orderedCols[0];
      }
    }
    const datasetKey = rawDataIdentity && typeof rawDataIdentity === "object"
      ? (rawDataIdentity.dataset_key || null)
      : null;
    const nextSignature = {
      tab: "calendar",
      portfolio: selectedPortfolio || null,
      view: viewMode || "annual",
      monthlySeries: nextValue || null,
      periodicity: periodicity || null,
      returnsBasis: returnsBasis || "total",
      partialMode: partialMode || "partial",
      datasetKey: datasetKey,
      benchmarkAssignments: benchmarkAssignments || {},
      longShortAssignments: longShortAssignments || {},
      dateRange: dateRange || null,
      volScaler: volScaler == null ? 0 : volScaler,
      volScalingAssignments: volScalingAssignments || {},
      resultsMeta: resultsMeta || null
    };

    return [
      currentDisabled === nextDisabled ? nu : nextDisabled,
      sameValue(currentOptions, nextOptions) ? nu : nextOptions,
      currentValue === nextValue ? nu : nextValue,
      sameValue(currentSignature, nextSignature) ? nu : nextSignature
    ];
  }

  function analyticsRollingReturnTypeState(metric, currentDisabled, currentStyle) {
    const nextDisabled = !(metric === "total_return" || metric === "excess_return");
    const nextStyle = nextDisabled ? { opacity: 0.5, pointerEvents: "none" } : {};
    return [
      currentDisabled === nextDisabled ? noUpdate() : nextDisabled,
      sameValue(currentStyle, nextStyle) ? noUpdate() : nextStyle
    ];
  }

  function analyticsSyncCalendarControls(
    triggerPayload,
    monthlyView,
    selectedSeries,
    storedMonthlySeries,
    currentDisabled,
    currentOptions,
    currentValue,
    datasetKey,
    originalPeriodicity,
    periodicity,
    returnsType,
    benchmarkAssignments,
    longShortAssignments,
    dateRange,
    volScaler,
    volScalingAssignments,
    partialMode,
    currentSignature
  ) {
    const nu = noUpdate();
    if (!triggerPayload || typeof triggerPayload !== "object" || String(triggerPayload.tab || "") !== "calendar") {
      return [nu, nu, nu, nu];
    }

    const series = Array.isArray(selectedSeries) ? selectedSeries.slice() : [];
    const nextOptions = series.map(function (name) {
      return { value: name, label: name };
    });
    let nextDisabled = true;
    let nextValue = null;
    if (String(monthlyView || "annual") === "monthly" && series.length) {
      nextDisabled = false;
      if (currentValue && series.indexOf(currentValue) !== -1) {
        nextValue = currentValue;
      } else if (storedMonthlySeries && series.indexOf(storedMonthlySeries) !== -1) {
        nextValue = storedMonthlySeries;
      } else {
        nextValue = series[0];
      }
    }

    const nextSignature = {
      tab: "calendar",
      datasetKey: datasetKey || null,
      periodicity: periodicity || null,
      originalPeriodicity: originalPeriodicity || null,
      selectedSeries: series,
      returnsType: returnsType || "total",
      benchmarkAssignments: benchmarkAssignments || {},
      longShortAssignments: longShortAssignments || {},
      dateRange: dateRange || null,
      monthlyView: monthlyView || "annual",
      monthlySeries: nextValue || null,
      volScaler: volScaler == null ? 0 : volScaler,
      volScalingAssignments: volScalingAssignments || {},
      partialMode: partialMode || "partial"
    };

    return [
      currentDisabled === nextDisabled ? nu : nextDisabled,
      sameValue(currentOptions, nextOptions) ? nu : nextOptions,
      currentValue === nextValue ? nu : nextValue,
      sameValue(currentSignature, nextSignature) ? nu : nextSignature
    ];
  }

  function analyticsRestoreSecondaryControls(
    activeTab,
    stateReady,
    storedRollWin,
    storedRollMetric,
    storedRollType,
    storedRollChart,
    storedDdChart,
    storedGrChart,
    storedFactorMode,
    storedFactorQuantiles,
    storedFactorTransform,
    storedFactorQqReference,
    storedConditionalView,
    storedConditionalComparator,
    storedConditionalThreshold,
    storedConditionalWindowConversion,
    storedConditionalStep,
    storedConditionalStepUnit,
    storedConditionalDisplayMode,
    storedRegimeDisplayMode,
    storedMonthlyView,
    currentRollWin,
    currentRollMetric,
    currentRollType,
    currentRollTypeDisabled,
    currentRollTypeStyle,
    currentRollChart,
    currentDdChart,
    currentGrChart,
    currentFactorMode,
    currentFactorQuantiles,
    currentFactorTransform,
    currentFactorQqReference,
    currentConditionalView,
    currentConditionalComparator,
    currentConditionalThreshold,
    currentConditionalWindowConversion,
    currentConditionalStep,
    currentConditionalStepUnit,
    currentConditionalDisplayMode,
    currentRegimeDisplayMode,
    currentMonthlyView
  ) {
    const nu = noUpdate();
    const outputs = new Array(21).fill(nu);
    if (!stateReady) {
      return outputs;
    }

    function sync(currentValue, nextValue) {
      return sameValue(currentValue, nextValue) ? nu : nextValue;
    }

    function coercePositiveInt(value, defaultValue) {
      const parsed = parseInt(value, 10);
      if (!Number.isFinite(parsed)) {
        return defaultValue;
      }
      return Math.max(defaultValue, parsed);
    }

    const tab = String(activeTab || "");
    const rollMetric = storedRollMetric || "total_return";
    const rollTypeDisabled = !(rollMetric === "total_return" || rollMetric === "excess_return");
    const rollTypeStyle = rollTypeDisabled ? { opacity: 0.5, pointerEvents: "none" } : {};

    if (tab === "rolling") {
      outputs[0] = sync(currentRollWin, storedRollWin || "1y");
      outputs[1] = sync(currentRollMetric, rollMetric);
      outputs[2] = sync(currentRollType, storedRollType || "annualized");
      outputs[3] = sync(currentRollTypeDisabled, rollTypeDisabled);
      outputs[4] = sync(currentRollTypeStyle, rollTypeStyle);
      outputs[5] = sync(currentRollChart, storedRollChart == null ? "chart" : storedRollChart);
    } else if (tab === "drawdown") {
      outputs[6] = sync(currentDdChart, storedDdChart == null ? "chart" : storedDdChart);
    } else if (tab === "growth") {
      outputs[7] = sync(currentGrChart, storedGrChart == null ? "chart" : storedGrChart);
    } else if (tab === "factor_analysis") {
      outputs[8] = sync(currentFactorMode, ["box", "scatter", "detail", "qq"].indexOf(storedFactorMode) !== -1 ? storedFactorMode : "box");
      outputs[9] = sync(currentFactorQuantiles, coercePositiveInt(storedFactorQuantiles, 5));
      outputs[10] = sync(currentFactorTransform, storedFactorTransform === "zscore" ? "zscore" : "raw");
      outputs[11] = sync(currentFactorQqReference, storedFactorQqReference === "reference" ? "reference" : "normal");
    } else if (tab === "conditional_returns") {
      outputs[12] = sync(currentConditionalView, ["coincident", "forward"].indexOf(storedConditionalView) !== -1 ? storedConditionalView : "forward");
      outputs[13] = sync(currentConditionalComparator, ["le", "ge"].indexOf(storedConditionalComparator) !== -1 ? storedConditionalComparator : "le");
      outputs[14] = sync(currentConditionalThreshold, storedConditionalThreshold == null ? 0 : storedConditionalThreshold);
      outputs[15] = sync(currentConditionalWindowConversion, ["compound", "end", "average", "sum"].indexOf(storedConditionalWindowConversion) !== -1 ? storedConditionalWindowConversion : "compound");
      outputs[16] = sync(currentConditionalStep, coercePositiveInt(storedConditionalStep, 1));
      outputs[17] = sync(currentConditionalStepUnit, ["periods", "months"].indexOf(storedConditionalStepUnit) !== -1 ? storedConditionalStepUnit : "months");
      outputs[18] = sync(currentConditionalDisplayMode, ["summary", "detail"].indexOf(storedConditionalDisplayMode) !== -1 ? storedConditionalDisplayMode : "summary");
    } else if (tab === "regime_analysis") {
      outputs[19] = sync(currentRegimeDisplayMode, ["summary", "detail"].indexOf(storedRegimeDisplayMode) !== -1 ? storedRegimeDisplayMode : "summary");
    } else if (tab === "calendar") {
      outputs[20] = sync(currentMonthlyView, storedMonthlyView == null ? "annual" : storedMonthlyView);
    }

    return outputs;
  }

  function analyticsResetStatisticsLoadedOnHydration(stateReady, currentLoaded, currentRenderedKey) {
    if (stateReady) {
      return [noUpdate(), noUpdate()];
    }
    const nextLoaded = currentLoaded === false ? noUpdate() : false;
    const nextRenderedKey = currentRenderedKey == null ? noUpdate() : null;
    return [nextLoaded, nextRenderedKey];
  }

  function analyticsFactorDefinitionLoadTrigger(activeTab, initialTabReady, stateReady, loaded) {
    if (loaded) {
      return noUpdate();
    }
    if (initialTabReady === false || !stateReady) {
      return noUpdate();
    }
    const tab = String(activeTab || "statistics");
    if (tab !== "factor_analysis" && tab !== "conditional_returns") {
      return noUpdate();
    }
    return {
      tab: tab,
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsRegimeDefinitionLoadTrigger(activeTab, initialTabReady, stateReady, loaded) {
    if (loaded) {
      return noUpdate();
    }
    if (initialTabReady === false || !stateReady) {
      return noUpdate();
    }
    const tab = String(activeTab || "statistics");
    if (tab !== "regime_analysis") {
      return noUpdate();
    }
    return {
      tab: tab,
      stamp: Date.now(),
      reason: triggeredId() || "unknown"
    };
  }

  function analyticsCorrelogramLoadingDisplay(activeTab, targetKey, renderedKey) {
    if (activeTab !== "correlogram") {
      return "auto";
    }
    if (targetKey && targetKey !== renderedKey) {
      return "show";
    }
    return "auto";
  }

  function analyticsConditionalReturnsLoadingDisplay(activeTab, stateReady, initialTabReady, targetKey, renderedKey) {
    if (activeTab !== "conditional_returns") {
      return "hide";
    }
    if (!initialTabReady || !stateReady) {
      return "show";
    }
    if (targetKey && targetKey !== renderedKey) {
      return "show";
    }
    return "hide";
  }

  function regressionSyncAnovaWindowOptions(selected, results, currentOptions, currentWindow, currentDisabled) {
    let nextOptions = [];
    let nextValue = null;
    let nextDisabled = true;
    const entry = (selected && results && results[selected]) ? results[selected] : null;
    const windowResults = entry && Array.isArray(entry.window_results) ? entry.window_results : [];
    if (windowResults.length) {
      nextOptions = windowResults.map(function (wr, idx) {
        const applyStart = String(((wr || {}).apply_start) || "").slice(0, 10);
        const applyEnd = String(((wr || {}).apply_end) || "").slice(0, 10);
        return {
          value: String(idx),
          label: "Window " + String(idx + 1) + ": " + applyStart + " to " + applyEnd
        };
      });
      const validValues = nextOptions.map(function (opt) { return opt.value; });
      nextValue = validValues.indexOf(currentWindow) !== -1 ? currentWindow : String(windowResults.length - 1);
      nextDisabled = false;
    }
    return [
      sameValue(currentOptions, nextOptions) ? noUpdate() : nextOptions,
      currentWindow === nextValue ? noUpdate() : nextValue,
      currentDisabled === nextDisabled ? noUpdate() : nextDisabled
    ];
  }

  function regressionSyncScatterXOptions(selected, results, mode, currentX) {
    const entry = (selected && results && results[selected]) ? results[selected] : null;
    if (!entry) {
      return [[], null, true];
    }
    const indepVars = Array.isArray(entry.independent_vars)
      ? entry.independent_vars.filter(Boolean)
      : [];
    const options = indepVars.map(function (name) {
      return { value: name, label: name };
    });
    const needsX = mode === "actual_vs_x" || mode === "predicted_vs_x";
    if (!needsX) {
      return [options, indepVars.indexOf(currentX) !== -1 ? currentX : null, true];
    }
    if (!indepVars.length) {
      return [[], null, true];
    }
    return [
      options,
      indepVars.indexOf(currentX) !== -1 ? currentX : indepVars[0],
      false
    ];
  }

  function regressionProjectActiveResultEntry(selected, results, currentEntry) {
    const nextEntry = (selected && results && results[selected]) ? results[selected] : null;
    const hasProjection = nextEntry
      && typeof nextEntry.display_json === "string"
      && nextEntry.display_json.length > 0
      && Array.isArray(nextEntry.display_columns)
      && nextEntry.display_columns.length > 0;
    if (nextEntry && !hasProjection) {
      return noUpdate();
    }
    if (sameValue(currentEntry, nextEntry)) {
      return noUpdate();
    }
    return nextEntry;
  }

  function portoptProjectActivePerformanceEntry(selectedPortfolio, results, currentEntry) {
    const sourceEntry = (selectedPortfolio && results && results[selectedPortfolio]) ? results[selectedPortfolio] : null;
    if (!sourceEntry || typeof sourceEntry !== "object") {
      return currentEntry === null ? noUpdate() : null;
    }
    const nextEntry = {
      reporting_returns_json: typeof sourceEntry.reporting_returns_json === "string" ? sourceEntry.reporting_returns_json : "",
      benchmark_returns_json: typeof sourceEntry.benchmark_returns_json === "string" ? sourceEntry.benchmark_returns_json : "",
      run_inputs: sourceEntry.run_inputs && typeof sourceEntry.run_inputs === "object" ? sourceEntry.run_inputs : {},
      config: sourceEntry.config && typeof sourceEntry.config === "object" ? sourceEntry.config : {},
      risk_free_meta: sourceEntry.risk_free_meta && typeof sourceEntry.risk_free_meta === "object" ? sourceEntry.risk_free_meta : {}
    };
    if (sameValue(currentEntry, nextEntry)) {
      return noUpdate();
    }
    return nextEntry;
  }

  function portoptProjectActiveAnalysisEntry(selectedPortfolio, results, currentEntry) {
    const sourceEntry = (selectedPortfolio && results && results[selectedPortfolio]) ? results[selectedPortfolio] : null;
    if (!sourceEntry || typeof sourceEntry !== "object") {
      return currentEntry === null ? noUpdate() : null;
    }
    const nextEntry = {
      window_weights: Array.isArray(sourceEntry.window_weights) ? sourceEntry.window_weights : [],
      config: sourceEntry.config && typeof sourceEntry.config === "object" ? sourceEntry.config : {},
      run_inputs: sourceEntry.run_inputs && typeof sourceEntry.run_inputs === "object" ? sourceEntry.run_inputs : {},
      risk_free_meta: sourceEntry.risk_free_meta && typeof sourceEntry.risk_free_meta === "object" ? sourceEntry.risk_free_meta : {},
      frontier_cache: sourceEntry.frontier_cache && typeof sourceEntry.frontier_cache === "object" ? sourceEntry.frontier_cache : {}
    };
    if (sameValue(currentEntry, nextEntry)) {
      return noUpdate();
    }
    return nextEntry;
  }

  function patchPlotlyTheme(colorScheme) {
    var isDark = colorScheme === "dark";
    var template = isDark ? "plotly_dark" : "plotly_white";
    var hoverBg = isDark ? "#25262b" : "#ffffff";
    var hoverFont = isDark ? "#f8f9fa" : "#1f2933";
    var hoverBorder = isDark ? "#5c5f66" : "#ced4da";
    var update = {
      template: template,
      "paper_bgcolor": "rgba(0,0,0,0)",
      "plot_bgcolor": "rgba(0,0,0,0)",
      "hoverlabel.bgcolor": hoverBg,
      "hoverlabel.bordercolor": hoverBorder,
      "hoverlabel.font.color": hoverFont
    };
    var plots = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < plots.length; i++) {
      if (plots[i].data && typeof Plotly !== "undefined") {
        try { Plotly.relayout(plots[i], update); } catch (e) { /* skip */ }
      }
    }
    return noUpdate();
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    dashmat_callbacks: {
      patchPlotlyTheme: patchPlotlyTheme,
      analyticsControlSync: analyticsControlSync,
      bulkUpdateSeriesSelection: bulkUpdateSeriesSelection,
      captureAnalyticsSeriesSnapshot: captureAnalyticsSeriesSnapshot,
      capturePortoptSeriesSnapshot: capturePortoptSeriesSnapshot,
      captureRegressionSeriesSnapshot: captureRegressionSeriesSnapshot,
      enforceRegressionSingleY: enforceRegressionSingleY,
      openAnalyticsSeriesModal: openAnalyticsSeriesModal,
      analyticsInitialSeriesBlocker: analyticsInitialSeriesBlocker,
      analyticsSeriesSelectionRenderTrigger: analyticsSeriesSelectionRenderTrigger,
      analyticsFactorRegimeSync: analyticsFactorRegimeSync,
      analyticsTabTrigger: analyticsTabTrigger,
      analyticsBootstrapCandidateTrigger: analyticsBootstrapCandidateTrigger,
      analyticsCandidateRefreshTrigger: analyticsCandidateRefreshTrigger,
      analyticsModalPreviewTrigger: analyticsModalPreviewTrigger,
      analyticsDownloadExcelDisabled: analyticsDownloadExcelDisabled,
      analyticsViewSync: analyticsViewSync,
      analyticsInitDateRange: analyticsInitDateRange,
      analyticsResolveButtonRange: analyticsResolveButtonRange,
      analyticsDateRangeButtons: analyticsDateRangeButtons,
      analyticsDateRangeStoreUpdate: analyticsDateRangeStoreUpdate,
      analyticsResolveInitialRange: analyticsResolveInitialRange,
      analyticsRestoreSecondaryControls: analyticsRestoreSecondaryControls,
      analyticsSyncReturnsTypeMirrors: analyticsSyncReturnsTypeMirrors,
      analyticsResetStatisticsLoadedOnHydration: analyticsResetStatisticsLoadedOnHydration,
      analyticsRollingReturnTypeState: analyticsRollingReturnTypeState,
      analyticsSyncCalendarControls: analyticsSyncCalendarControls,
      analyticsFactorDefinitionLoadTrigger: analyticsFactorDefinitionLoadTrigger,
      analyticsRegimeDefinitionLoadTrigger: analyticsRegimeDefinitionLoadTrigger,
      analyticsCorrelogramLoadingDisplay: analyticsCorrelogramLoadingDisplay,
      analyticsConditionalReturnsLoadingDisplay: analyticsConditionalReturnsLoadingDisplay,
      accountListDismissNotice: accountListDismissNotice,
      accountListPruneDbImportProvenance: accountListPruneDbImportProvenance,
      accountListRenderNotice: accountListRenderNotice,
      accountListRowsRefreshTrigger: accountListRowsRefreshTrigger,
      accountListSaveState: accountListSaveState,
      validateAnalyticsDbAddSelection: validateAnalyticsDbAddSelection,
      clearWorkspaceSession: clearWorkspaceSession,
      commonDailyButtonDisabled: commonDailyButtonDisabled,
      loadWorkspaceSession: loadWorkspaceSession,
      loadWorkspaceSessionDialog: loadWorkspaceSessionDialog,
      navigateAnalytics: navigateAnalytics,
      navigatePortopt: navigatePortopt,
      navigateRegression: navigateRegression,
      openPortoptSeriesModal: openPortoptSeriesModal,
      applyPortoptSeriesSnapshot: applyPortoptSeriesSnapshot,
      portoptActiveVisTrigger: portoptActiveVisTrigger,
      portoptClearRawDbRows: portoptClearRawDbRows,
      portoptClearReturns: portoptClearReturns,
      portoptInitLinearConstraintsGrid: portoptInitLinearConstraintsGrid,
      portoptInitTau: portoptInitTau,
      portoptReportingBasisControl: portoptReportingBasisControl,
      portoptSyncNameWithModel: portoptSyncNameWithModel,
      portoptSyncSaveSeriesUi: portoptSyncSaveSeriesUi,
      portoptSyncCalendarControls: portoptSyncCalendarControls,
      portoptSyncFrontierControls: portoptSyncFrontierControls,
      portoptSyncReturnsBasisFromMirrors: portoptSyncReturnsBasisFromMirrors,
      portoptSyncReturnsBasisMirrors: portoptSyncReturnsBasisMirrors,
      portoptSyncTau: portoptSyncTau,
      portoptToggleRawDivideBy: portoptToggleRawDivideBy,
      portoptToggleRollingReturnType: portoptToggleRollingReturnType,
      portoptToggleUiElements: portoptToggleUiElements,
      portoptLinearConstraintColumnDefs: portoptLinearConstraintColumnDefs,
      portoptMatrixGridData: portoptMatrixGridData,
      portoptReturnsGridData: portoptReturnsGridData,
      portoptUpdateDateRangeStore: portoptUpdateDateRangeStore,
      portoptUpdateMatrixUi: portoptUpdateMatrixUi,
      portoptUpdateOptStepOnUnitChange: portoptUpdateOptStepOnUnitChange,
      openRegressionSeriesModal: openRegressionSeriesModal,
      portoptBootstrapRestore: portoptBootstrapRestore,
      portoptControlSync: portoptControlSync,
      portoptInitialSeriesBlocker: portoptInitialSeriesBlocker,
      portoptMarkVisitedTabLoaded: portoptMarkVisitedTabLoaded,
      portoptProjectActiveAnalysisEntry: portoptProjectActiveAnalysisEntry,
      portoptProjectActivePerformanceEntry: portoptProjectActivePerformanceEntry,
      regressionClearRawDbRows: regressionClearRawDbRows,
      syncPortoptSeriesModalGrid: syncPortoptSeriesModalGrid,
      portoptViewSync: portoptViewSync,
      regressionDeleteRawDbRow: regressionDeleteRawDbRow,
      regressionInitialSeriesBlocker: regressionInitialSeriesBlocker,
      regressionControlSync: regressionControlSync,
      regressionModelSelectSync: regressionModelSelectSync,
      regressionProjectActiveResultEntry: regressionProjectActiveResultEntry,
      regressionSyncCalendarControls: regressionSyncCalendarControls,
      regressionSyncAnovaWindowOptions: regressionSyncAnovaWindowOptions,
      regressionSyncSaveSeriesUi: regressionSyncSaveSeriesUi,
      regressionSyncScatterXOptions: regressionSyncScatterXOptions,
      regressionToggleFileMenuActions: regressionToggleFileMenuActions,
      regressionToggleRollingReturnType: regressionToggleRollingReturnType,
      regressionToggleSheetSelectDisabled: regressionToggleSheetSelectDisabled,
      regressionToggleWindowControls: regressionToggleWindowControls,
      saveWorkspaceSession: saveWorkspaceSession,
      syncAnalyticsPeriodicity: syncAnalyticsPeriodicity,
      syncPortoptPeriodicity: syncPortoptPeriodicity,
      triggerAnalyticsUpload: triggerAnalyticsUpload,
      triggerPortoptUpload: triggerPortoptUpload,
      triggerRegressionUpload: triggerRegressionUpload,
      uiBlockerEnable: uiBlockerEnable,
      uiBlockerRelease: uiBlockerRelease,
      releaseBlockerOnSeriesGridReady: releaseBlockerOnSeriesGridReady
    }
  });
})();
