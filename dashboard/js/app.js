// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — Main Application
//  Tab routing, state management, auto-refresh.
// ═══════════════════════════════════════════════════════════════════════════

import { fetchMetrics, fetchTraces, fetchEvalResults, fetchHealth } from './api.js';
import { renderKPICards } from './kpi.js';
import { createLineChart, createDoughnutChart, refreshChartTheme } from './charts.js';
import { renderTraceTable } from './traces.js';
import { renderGuardrailsTab } from './guardrails.js';
import { renderEvaluationTab } from './evaluation.js';

// ── State ────────────────────────────────────────────────────────────────
let currentTab = 'overview';
let metricsData = null;
let tracesData = null;
let refreshInterval = null;

// ── Tab Routing ──────────────────────────────────────────────────────────
function switchTab(tabId) {
  currentTab = tabId;

  // Update nav buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  // Update content panels
  document.querySelectorAll('.tab-content').forEach(panel => {
    panel.classList.toggle('active', panel.id === `tab-${tabId}`);
  });

  // Lazy-load tab data
  loadTabData(tabId);
}

async function loadTabData(tabId) {
  switch (tabId) {
    case 'overview':
      await loadOverview();
      break;
    case 'traces':
      await loadTraces();
      break;
    case 'guardrails':
      await loadGuardrails();
      break;
    case 'evaluation':
      await loadEvaluation();
      break;
    case 'system':
      await loadSystem();
      break;
  }
}

// ── Overview Tab ─────────────────────────────────────────────────────────
async function loadOverview() {
  const data = await fetchMetrics();
  if (!data) return;
  metricsData = data;

  // KPI cards
  renderKPICards('overview-kpis', data.summary);

  // Attribution donut
  const ab = data.summary.attribution_breakdown;
  createDoughnutChart(
    'attribution-donut',
    ['RAG Grounded', 'Parametric Risk', 'Unknown'],
    [ab.rag_grounded, ab.parametric_risk, ab.unknown],
    ['#10b981', '#f59e0b', '#71717a']
  );
  // Update legend
  const attrLegend = document.getElementById('attribution-legend');
  if (attrLegend) {
    attrLegend.innerHTML = [
      { label: 'RAG Grounded', count: ab.rag_grounded, color: '#10b981' },
      { label: 'Parametric Risk', count: ab.parametric_risk, color: '#f59e0b' },
      { label: 'Unknown', count: ab.unknown, color: '#71717a' },
    ].map(l => `
      <span class="chart-legend-item">
        <span class="chart-legend-dot" style="background:${l.color}"></span>
        ${l.label}: <strong style="color:var(--text-primary)">${l.count}</strong>
      </span>
    `).join('');
  }

  // Latency timeline from recent traces
  const traces = data.recent_traces || [];
  if (traces.length > 1) {
    const labels = traces.map(t => {
      const d = new Date(t.timestamp);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    });
    createLineChart('latency-timeline', labels, [
      {
        label: 'Latency (ms)',
        data: traces.map(t => t.latency_ms),
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99,102,241,0.08)',
        fill: true,
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: '#6366f1',
      },
    ]);
  }

  // Recent traces mini-table
  renderRecentTable(traces.slice(0, 8));
}

function renderRecentTable(traces) {
  const tbody = document.getElementById('overview-recent-tbody');
  if (!tbody) return;

  if (!traces.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:2rem">No queries yet</td></tr>';
    return;
  }

  tbody.innerHTML = traces.map(t => {
    const latencyColor = t.latency_ms < 3000 ? 'var(--accent-emerald)' :
                         t.latency_ms < 8000 ? 'var(--accent-amber)' : 'var(--accent-rose)';
    return `
      <tr>
        <td style="font-size:0.75rem;color:var(--text-muted);white-space:nowrap">${new Date(t.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}</td>
        <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t.query?.slice(0, 50) || '—'}</td>
        <td class="mono right" style="color:${latencyColor}">${(t.latency_ms / 1000).toFixed(2)}s</td>
        <td class="mono right">${t.steps}</td>
        <td class="center">${t.attribution === 'rag_grounded'
          ? '<span class="badge badge-success">RAG</span>'
          : t.attribution === 'parametric_risk'
          ? '<span class="badge badge-warning">Param</span>'
          : '<span class="badge badge-neutral">—</span>'}</td>
      </tr>`;
  }).join('');
}

// ── Traces Tab ───────────────────────────────────────────────────────────
async function loadTraces() {
  const data = await fetchTraces(100);
  if (!data) return;
  tracesData = data.traces;
  renderTraceTable('trace-tbody', tracesData);
  const countEl = document.getElementById('trace-count');
  if (countEl) countEl.textContent = `(${tracesData.length})`;
}

// ── Guardrails Tab ───────────────────────────────────────────────────────
async function loadGuardrails() {
  const data = metricsData || await fetchMetrics();
  if (!data) return;
  renderGuardrailsTab(data.summary, data.recent_traces || []);
  if (window.lucide) lucide.createIcons();
}

// ── Evaluation Tab ───────────────────────────────────────────────────────
async function loadEvaluation() {
  const data = await fetchEvalResults();
  if (!data) return;
  renderEvaluationTab(data.results);
}

// ── System Tab ───────────────────────────────────────────────────────────
async function loadSystem() {
  const container = document.getElementById('system-content');
  if (!container) return;

  const health = await fetchHealth();

  if (!health) {
    container.innerHTML = `
      <div class="empty-state">
        <i data-lucide="server-off"></i>
        <p>Cannot reach backend</p>
        <p class="hint">Make sure the FastAPI server is running</p>
      </div>`;
    if (window.lucide) lucide.createIcons();
    return;
  }

  const statusIcon = health.status === 'healthy' ? 'check-circle' : 'alert-triangle';
  const statusColor = health.status === 'healthy' ? 'var(--accent-emerald)' : 'var(--accent-amber)';

  container.innerHTML = `
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem">
      <i data-lucide="${statusIcon}" style="width:24px;height:24px;color:${statusColor}"></i>
      <span style="font-size:1.1rem;font-weight:600;color:var(--text-primary)">System ${health.status}</span>
    </div>

    <div class="stat-grid">
      ${(health.components || []).map(c => `
        <div class="stat-item">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <div class="stat-label">${c.name}</div>
            <span class="badge ${c.healthy ? 'badge-success' : 'badge-danger'}">${c.healthy ? 'Healthy' : 'Down'}</span>
          </div>
          <div style="font-size:0.75rem;color:var(--text-tertiary);margin-top:0.25rem">
            ${c.latency_ms ? `Latency: ${c.latency_ms.toFixed(0)}ms` : c.detail || ''}
          </div>
        </div>
      `).join('')}
    </div>`;

  if (window.lucide) lucide.createIcons();
}

// ── Theme Toggle ─────────────────────────────────────────────────────────
function toggleTheme() {
  document.documentElement.classList.toggle('light');
  refreshChartTheme();
  // Re-render current tab for theme consistency
  loadTabData(currentTab);
}

// ── Refresh ──────────────────────────────────────────────────────────────
async function refreshDashboard() {
  const btn = document.getElementById('refresh-btn');
  if (btn) {
    btn.style.animation = 'spin 0.5s linear';
    setTimeout(() => btn.style.animation = '', 500);
  }
  await loadTabData(currentTab);
}

// ── Init ─────────────────────────────────────────────────────────────────
function init() {
  // Tab click handlers
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Theme toggle
  window.toggleTheme = toggleTheme;
  window.refreshDashboard = refreshDashboard;

  // Initialize Lucide icons
  if (window.lucide) lucide.createIcons();

  // Load initial tab
  switchTab('overview');

  // Auto-refresh every 30s
  refreshInterval = setInterval(() => {
    if (currentTab === 'overview' || currentTab === 'traces') {
      loadTabData(currentTab);
    }
  }, 30000);
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
