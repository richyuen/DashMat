(function () {
  const SELECTOR = ".dashmat-tooltip-trigger-width[data-tooltip-width-intent='1']";
  let rafId = null;

  function contentWidth(node) {
    if (!node) {
      return 0;
    }
    const styles = window.getComputedStyle(node);
    const paddingLeft = parseFloat(styles.paddingLeft || "0") || 0;
    const paddingRight = parseFloat(styles.paddingRight || "0") || 0;
    return Math.max(0, Math.round(node.clientWidth - paddingLeft - paddingRight));
  }

  function syncWidths() {
    rafId = null;
    document.querySelectorAll(SELECTOR).forEach((node) => {
      const slot = node.parentElement && node.parentElement.parentElement;
      if (!slot) {
        return;
      }
      const width = contentWidth(slot);
      if (width > 0) {
        node.style.width = width + "px";
      }
    });
  }

  function scheduleSync() {
    if (rafId !== null) {
      window.cancelAnimationFrame(rafId);
    }
    rafId = window.requestAnimationFrame(function () {
      window.requestAnimationFrame(syncWidths);
    });
  }

  window.addEventListener("load", scheduleSync);
  window.addEventListener("resize", scheduleSync);
  document.addEventListener("DOMContentLoaded", scheduleSync);

  if (document.body && "MutationObserver" in window) {
    const observer = new MutationObserver(scheduleSync);
    observer.observe(document.body, { childList: true, subtree: true });
  }

  scheduleSync();
})();
