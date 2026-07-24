// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — Chart Factories
//  Reusable Chart.js rendering functions.
// ═══════════════════════════════════════════════════════════════════════════

// Chart.js dark theme defaults
function applyChartDefaults() {
  if (!window.Chart) return;
  const isDark = !document.documentElement.classList.contains('light');
  Chart.defaults.color = isDark ? '#71717a' : '#94a3b8';
  Chart.defaults.borderColor = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)';
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.tooltip.backgroundColor = isDark ? '#27272a' : '#ffffff';
  Chart.defaults.plugins.tooltip.titleColor = isDark ? '#fafafa' : '#0f172a';
  Chart.defaults.plugins.tooltip.bodyColor = isDark ? '#a1a1aa' : '#475569';
  Chart.defaults.plugins.tooltip.borderColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.legend.display = false;
}

// Store chart instances for cleanup
const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

/**
 * Create a line chart (e.g., latency over time).
 */
export function createLineChart(canvasId, labels, datasets) {
  applyChartDefaults();
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, maxTicksLimit: 10 } },
        y: {
          beginAtZero: true,
          grid: { color: document.documentElement.classList.contains('light') ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.03)' },
        },
      },
    },
  });
  return chartInstances[canvasId];
}

/**
 * Create a doughnut chart (e.g., guardrails breakdown).
 */
export function createDoughnutChart(canvasId, labels, data, colors) {
  applyChartDefaults();
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;

  const isDark = !document.documentElement.classList.contains('light');

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: isDark ? '#18181b' : '#ffffff',
        borderWidth: 3,
        hoverBorderColor: isDark ? '#18181b' : '#ffffff',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
              return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
            },
          },
        },
      },
    },
  });
  return chartInstances[canvasId];
}

/**
 * Create a stacked bar chart (e.g., latency breakdown per step).
 */
export function createStackedBarChart(canvasId, labels, datasets) {
  applyChartDefaults();
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 0, maxTicksLimit: 10 } },
        y: {
          stacked: true,
          beginAtZero: true,
          grid: { color: document.documentElement.classList.contains('light') ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.03)' },
          title: { display: true, text: 'ms', font: { size: 10 }, color: '#71717a' },
        },
      },
    },
  });
  return chartInstances[canvasId];
}

/**
 * Create a horizontal bar chart (e.g., attribution breakdown).
 */
export function createHorizontalBarChart(canvasId, labels, data, colors) {
  applyChartDefaults();
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderRadius: 6,
        maxBarThickness: 28,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.03)' } },
        y: { grid: { display: false } },
      },
    },
  });
  return chartInstances[canvasId];
}

/**
 * Refresh chart defaults on theme change.
 */
export function refreshChartTheme() {
  applyChartDefaults();
  Object.values(chartInstances).forEach(c => c.update());
}
