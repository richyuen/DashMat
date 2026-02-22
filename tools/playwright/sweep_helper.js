(() => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  window.__sweep = {
    sleep,
    setProps: (map) => {
      for (const [id, props] of Object.entries(map || {})) {
        try {
          window.dash_clientside.set_props(id, props);
        } catch (_) {}
      }
    },
    setSeries: async (y, xs) => {
      const order = [
        'SPX',
        'RMID',
        'R2000',
        'EAFE',
        'EM',
        'MSCIUSREIT',
        'BCAgg',
        'BCHY',
        'BCGAgg',
        'BCGC13',
      ];
      window.__sweep.setProps({
        'reg-series-select': { data: xs || [] },
        'reg-series-select-value-store': { data: xs || [] },
        'reg-dependent-var-store': { data: y || null },
        'reg-series-order-store': { data: order },
        'reg-benchmark-assignments-store': { data: {} },
        'reg-long-short-store': { data: {} },
        'reg-vol-scaling-assignments-store': { data: {} },
      });
      await sleep(350);
    },
    setRun: async (cfg) => {
      const c = cfg || {};
      window.__sweep.setProps({
        'reg-model-store': { data: c.model || 'ols' },
        'reg-regression-name-store': { data: (c.model || 'ols').toUpperCase() },
        'reg-force-zero-intercept-store': { data: !!c.forceZero },
        'reg-robust-se-store': { data: c.robustSE !== false },
        'reg-exp-wt-store': { data: !!c.expWt },
        'reg-halflife-store': { data: c.halfLife || 63 },
        'reg-window-type-store': { data: c.windowType || 'full' },
        'reg-window-size-store': { data: c.windowSize || 120 },
        'reg-opt-step-store': { data: c.optStep || 1 },
        'reg-opt-step-unit-store': { data: c.optStepUnit || 'months' },
        'reg-fill-in-sample-store': { data: c.fillInSample || 'off' },
        'reg-missing-data-store': { data: c.missingData || 'fill_na' },
        'reg-alpha-store': { data: c.alpha || 1 },
        'reg-l1-ratio-store': { data: c.l1Ratio || 0.5 },
        'reg-linear-constraints-store': { data: [] },
        'reg-arima-p-input': { value: c.arimaP || 0 },
        'reg-arima-d-input': { value: c.arimaD || 0 },
        'reg-arima-q-input': { value: c.arimaQ || 0 },
        'reg-garch-p-input': { value: c.garchP || 0 },
        'reg-garch-q-input': { value: c.garchQ || 0 },
      });
      await sleep(400);
    },
    run: async () => {
      const status = document.querySelector('#reg-run-status-text');
      const prev = (status?.textContent || '').trim();
      document.querySelector('#reg-run-button')?.click();
      for (let i = 0; i < 180; i++) {
        const txt = (document.querySelector('#reg-run-status-text')?.textContent || '').trim();
        if (txt.length > 0 && txt !== prev) {
          return txt;
        }
        await sleep(400);
      }
      return (document.querySelector('#reg-run-status-text')?.textContent || '').trim();
    },
    tab: async (name) => {
      const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
      const t = tabs.find((el) => (el.textContent || '').trim() === name);
      if (!t) {
        return false;
      }
      t.click();
      await sleep(700);
      return true;
    },
    hasGraph: (sel) => !!document.querySelector(sel + ' .js-plotly-plot'),
    hasGrid: (sel) => !!document.querySelector(sel + ' .ag-root-wrapper'),
    headers: (sel) =>
      Array.from(document.querySelectorAll(sel + ' .ag-header-cell-text')).map((e) =>
        (e.textContent || '').trim()
      ),
    currentResult: () => document.querySelector('#reg-result-select')?.value || '',
    periodicityValue: () => document.querySelector('#reg-periodicity-select')?.value || '',
    deleteResult: async () => {
      const before = window.__sweep.currentResult();
      document.querySelector('#reg-delete-result-btn')?.click();
      for (let i = 0; i < 40; i++) {
        const after = window.__sweep.currentResult();
        if (after !== before) {
          return { before, after };
        }
        await sleep(300);
      }
      return { before, after: window.__sweep.currentResult() };
    },
    scatterTraceNames: () => {
      const g = document.querySelector('#reg-scatter-content .js-plotly-plot');
      return g?.data ? g.data.map((t) => t.name) : [];
    },
  };
  return true;
})();
