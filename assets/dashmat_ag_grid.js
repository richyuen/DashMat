(function () {
  const dagfuncs = (window.dashAgGridFunctions =
    window.dashAgGridFunctions || {});

  function dashmatNormalizeMinus(value) {
    if (typeof value !== "string") {
      return value;
    }
    return value.replace(/\u2212/g, "-");
  }

  function dashmatFormatNumber(value, spec) {
    if (value === null || value === undefined || value === "") {
      return "";
    }
    if (typeof value === "boolean") {
      return value ? "True" : "False";
    }
    if (typeof value === "string") {
      return dashmatNormalizeMinus(value);
    }

    const numeric = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(numeric)) {
      return dashmatNormalizeMinus(String(value));
    }

    if (!spec) {
      return dashmatNormalizeMinus(String(numeric));
    }

    const match = String(spec).match(/^(,)?(?:\.(\d+))?([fd%])$/);
    if (!match) {
      return dashmatNormalizeMinus(String(numeric));
    }

    const useGrouping = match[1] === ",";
    const precision = match[2] ? parseInt(match[2], 10) : 0;
    const type = match[3];
    const scaled = type === "%" ? numeric * 100 : numeric;
    const decimals = type === "d" ? 0 : precision;
    const formatted = new Intl.NumberFormat("en-US", {
      useGrouping,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(scaled);
    return dashmatNormalizeMinus(type === "%" ? `${formatted}%` : formatted);
  }

  function dashmatProcessCellForClipboard(params) {
    if (!params) {
      return "";
    }
    if (typeof params.value === "number" && Number.isFinite(params.value)) {
      return String(params.value);
    }
    if (typeof params.value === "boolean") {
      return params.value ? "True" : "False";
    }
    if (typeof params.value === "string") {
      return dashmatNormalizeMinus(params.value);
    }
    if (params.valueFormatted !== null && params.valueFormatted !== undefined) {
      return dashmatNormalizeMinus(String(params.valueFormatted));
    }
    return "";
  }

  function dashmatProcessFormattedCellForClipboard(params) {
    if (!params) {
      return "";
    }
    if (params.valueFormatted !== null && params.valueFormatted !== undefined) {
      return dashmatNormalizeMinus(String(params.valueFormatted));
    }
    if (typeof params.value === "boolean") {
      return params.value ? "True" : "False";
    }
    if (typeof params.value === "string") {
      return dashmatNormalizeMinus(params.value);
    }
    if (typeof params.value === "number" && Number.isFinite(params.value)) {
      return String(params.value);
    }
    return "";
  }

  window.dashmatNormalizeMinus = dashmatNormalizeMinus;
  window.dashmatFormatNumber = dashmatFormatNumber;
  window.dashmatProcessCellForClipboard = dashmatProcessCellForClipboard;
  window.dashmatProcessFormattedCellForClipboard =
    dashmatProcessFormattedCellForClipboard;
  dagfuncs.dashmatNormalizeMinus = dashmatNormalizeMinus;
  dagfuncs.dashmatFormatNumber = dashmatFormatNumber;
  dagfuncs.dashmatProcessCellForClipboard = dashmatProcessCellForClipboard;
  dagfuncs.dashmatProcessFormattedCellForClipboard =
    dashmatProcessFormattedCellForClipboard;
})();
