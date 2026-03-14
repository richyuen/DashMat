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

  const dashmatModuleRoutes = ["/analyticstool", "/portopt", "/regression"];

  function normalizePath(pathname) {
    return String(pathname || "").split("?")[0].replace(/\/$/, "") || "/";
  }

  function isDashmatModuleRoute(pathname) {
    return dashmatModuleRoutes.indexOf(normalizePath(pathname)) !== -1;
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

  function buildModuleRouteReadyPayload(targetPath, routeBlockerState, ready) {
    const state = routeBlockerState && typeof routeBlockerState === "object" ? routeBlockerState : {};
    const statePath = normalizePath(state.pathname);
    const requestId = Number.isFinite(state.requestId) ? state.requestId : 0;
    return {
      pathname: targetPath,
      requestId: statePath === targetPath ? requestId : 0,
      ready: statePath === targetPath ? !!ready : false
    };
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

  function modulePageRouteReady(pathname, pageLoadReady, modalOpened, modalStillNeeded, virtualRows, targetPath, routeBlockerState) {
    const localBlocker = buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      modalStillNeeded,
      virtualRows,
      targetPath
    );
    const pagePath = normalizePath(pathname);
    let routeReady = false;
    if (pagePath === targetPath && pageLoadReady) {
      routeReady = localBlocker === true || modalOpened === true || !modalStillNeeded || Array.isArray(virtualRows);
    }
    return buildModuleRouteReadyPayload(targetPath, routeBlockerState, routeReady);
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

  function analyticsInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      analyticsInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/analyticstool"
    );
  }

  function analyticsRouteReady(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, poOriginSeries) {
    return modulePageRouteReady(
      pathname,
      pageLoadReady,
      modalOpened,
      analyticsInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/analyticstool",
      routeBlockerState
    );
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
    const knownColumns = new Set(resolveStoredList(currentOrder, "po-series-order-store").filter(function (series) {
      return columnSet.has(series);
    }));
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

  function portoptInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      portoptInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/portopt"
    );
  }

  function portoptRouteReady(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, poOriginSeries) {
    return modulePageRouteReady(
      pathname,
      pageLoadReady,
      modalOpened,
      portoptInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, poOriginSeries, pageVisited),
      virtualRows,
      "/portopt",
      routeBlockerState
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
    try {
      const api = await window.dash_ag_grid.getApiAsync(gridId);
      if (!api) {
        return noUpdate();
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
      return noUpdate();
    }
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

    if (trigger === "dashmat-raw-data-meta-store") {
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
    const knownColumns = new Set(resolveStoredList(currentOrder, "po-series-order-store").filter(function (series) {
      return columnSet.has(series);
    }));
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
    if (!resolveStoredBool(pageVisited, "po-page-visited-store") && !selectedValid.length) {
      tempSelect = columns.filter(function (series) {
        return !poOriginSet.has(series);
      });
      shouldOpen = tempSelect.length > 0;
    } else if (trigger !== "dashmat-raw-data-meta-store" && genericNew.length) {
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
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
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
    pathname,
    rawMeta,
    currentSelect,
    currentBench,
    currentLs,
    currentOrder,
    currentVolScaling,
    poOriginSeries,
    pageVisited
  ) {
    const trigger = triggeredId();
    if (trigger !== "at-open-series-modal-button" && trigger !== "at-page-load-trigger") {
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

    if (pageLoadIntervals === null || pageLoadIntervals === undefined) {
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
      return [
        noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), noUpdate(), true, false
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
    if (!pageVisited && !selectedValid.length) {
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

  function regressionInitialSeriesBlocker(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, currentDepVar, poOriginSeries) {
    return buildLocalBlockerState(
      pathname,
      pageLoadReady,
      modalOpened,
      regressionInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, currentDepVar, poOriginSeries, pageVisited),
      virtualRows,
      "/regression"
    );
  }

  function regressionRouteReady(pathname, rawMeta, currentSelect, pageLoadReady, modalOpened, virtualRows, routeBlockerState, pageVisited, currentOrder, currentDepVar, poOriginSeries) {
    return modulePageRouteReady(
      pathname,
      pageLoadReady,
      modalOpened,
      regressionInitialSeriesModalPending(rawMeta, currentSelect, currentOrder, currentDepVar, poOriginSeries, pageVisited),
      virtualRows,
      "/regression",
      routeBlockerState
    );
  }

  function moduleRouteBlockerState(pathname, currentState) {
    const nextPath = normalizePath(pathname);
    const state = currentState && typeof currentState === "object" ? currentState : {};
    const prevPath = normalizePath(state.pathname);
    const prevRequestId = Number.isFinite(state.requestId) ? state.requestId : 0;
    if (!isDashmatModuleRoute(nextPath)) {
      return { active: false, pathname: nextPath, requestId: prevRequestId };
    }
    if (!prevPath || prevPath === "/" || nextPath === prevPath || !isDashmatModuleRoute(prevPath)) {
      return { active: false, pathname: nextPath, requestId: prevRequestId };
    }
    return { active: true, pathname: nextPath, requestId: prevRequestId + 1 };
  }

  function moduleRouteBlockerPresentation(routeBlockerState, atReady, poReady, regReady) {
    const state = routeBlockerState && typeof routeBlockerState === "object" ? routeBlockerState : {};
    const path = normalizePath(state.pathname);
    if (!state.active || !isDashmatModuleRoute(path)) {
      return [{ display: "none" }, false];
    }
    const requestId = Number.isFinite(state.requestId) ? state.requestId : 0;
    let readyPayload = null;
    if (path === "/analyticstool") {
      readyPayload = atReady;
    } else if (path === "/portopt") {
      readyPayload = poReady;
    } else if (path === "/regression") {
      readyPayload = regReady;
    }
    const readyState = readyPayload && typeof readyPayload === "object" ? readyPayload : {};
    const isReady =
      normalizePath(readyState.pathname) === path &&
      Number(readyState.requestId || 0) === requestId &&
      readyState.ready === true;
    if (isReady) {
      return [{ display: "none" }, false];
    }
    return [{ position: "fixed", inset: 0, zIndex: 2400 }, true];
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
    storedReportingBasis
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
    const modelDefaults = {
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
    const trimmedName = typeof storedPortfolioName === "string" ? storedPortfolioName.trim() : "";
    const resolvedPortfolioName = trimmedName || modelDefaults[resolvedOptModel] || "Port";

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

    return [
      periodicityOptions,
      resolvedPeriodicity,
      resolvedVolScaler,
      selectedSeries,
      resolvedActiveTab,
      normalizeView(storedWeightView),
      normalizeView(storedAttributionView),
      normalizeView(storedRiskView),
      normalizeView(storedTurnoverView),
      normalizeView(storedFrontierView),
      resolvedOptWindow,
      resolvedWindowSize,
      resolvedOptStep,
      resolvedOptStepUnit,
      resolvedOptModel,
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
      resolvedReturnsBasis,
      resolvedReportingBasis,
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

  function portoptControlSync(periodicity, volScaler, activeTab, seriesSelect, fillInSample, optStepUnit, optWindow, windowSize, optStep, optModel, portfolioName, expWtCov, halflife, covShrinkage, covShrinkageTarget, missingData, objective, blTau, exAnteMode, useRiskFree, returnsBasis, reportingBasis) {
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
      exAnteMode || "ret_cov",
      useRiskFree === "zero" ? false : true,
      returnsBasis === "excess" ? "excess" : "total",
      reportingBasis === "split"
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
      captureAnalyticsSeriesSnapshot: captureAnalyticsSeriesSnapshot,
      capturePortoptSeriesSnapshot: capturePortoptSeriesSnapshot,
      captureRegressionSeriesSnapshot: captureRegressionSeriesSnapshot,
      enforceRegressionSingleY: enforceRegressionSingleY,
      openAnalyticsSeriesModal: openAnalyticsSeriesModal,
      analyticsInitialSeriesBlocker: analyticsInitialSeriesBlocker,
      analyticsFactorRegimeSync: analyticsFactorRegimeSync,
      analyticsViewSync: analyticsViewSync,
      clearWorkspaceSession: clearWorkspaceSession,
      commonDailyButtonDisabled: commonDailyButtonDisabled,
      loadWorkspaceSession: loadWorkspaceSession,
      loadWorkspaceSessionDialog: loadWorkspaceSessionDialog,
      moduleRouteBlockerPresentation: moduleRouteBlockerPresentation,
      moduleRouteBlockerState: moduleRouteBlockerState,
      analyticsRouteReady: analyticsRouteReady,
      navigateAnalytics: navigateAnalytics,
      navigatePortopt: navigatePortopt,
      portoptRouteReady: portoptRouteReady,
      regressionRouteReady: regressionRouteReady,
      navigateRegression: navigateRegression,
      openPortoptSeriesModal: openPortoptSeriesModal,
      openRegressionSeriesModal: openRegressionSeriesModal,
      portoptBootstrapRestore: portoptBootstrapRestore,
      portoptControlSync: portoptControlSync,
      portoptInitialSeriesBlocker: portoptInitialSeriesBlocker,
      portoptMarkVisitedTabLoaded: portoptMarkVisitedTabLoaded,
      portoptViewSync: portoptViewSync,
      regressionInitialSeriesBlocker: regressionInitialSeriesBlocker,
      regressionControlSync: regressionControlSync,
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
