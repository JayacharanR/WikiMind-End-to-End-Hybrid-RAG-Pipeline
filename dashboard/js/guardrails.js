// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — Guardrails Panel
// ═══════════════════════════════════════════════════════════════════════════

import { createDoughnutChart, createHorizontalBarChart } from './charts.js';

function escapeHtml(str) {
  if (typeof str !== 'string') return str ?? '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

/**
 * Render the Guardrails tab content.
 * @param {Object} summary - metrics summary from /api/metrics
 * @param {Array} traces - recent traces
 */
export function renderGuardrailsTab(summary, traces) {
  // 1. Guardrails donut chart
  const gs = summary.guardrails_stats;
  createDoughnutChart(
    'guardrails-donut',
    ['Applied', 'Bypassed'],
    [gs.applied, gs.bypassed],
    ['#10b981', '#71717a']
  );

  // Update legend
  const legend = document.getElementById('guardrails-legend');
  if (legend) {
    legend.innerHTML = [
      { label: 'Applied', count: gs.applied, color: '#10b981' },
      { label: 'Bypassed', count: gs.bypassed, color: '#71717a' },
    ].map(l => `
      <span class="chart-legend-item">
        <span class="chart-legend-dot" style="background:${l.color}"></span>
        ${l.label}: <strong style="color:var(--text-primary)">${l.count}</strong>
      </span>
    `).join('');
  }

  // 2. Grade breakdown bar chart
  const gb = summary.grade_breakdown;
  createHorizontalBarChart(
    'grades-bar',
    ['Grounded', 'Hallucinated', 'Useful', 'Not Useful'],
    [gb.grounded, gb.hallucinated, gb.useful, gb.not_useful],
    ['#10b981', '#f43f5e', '#6366f1', '#f59e0b']
  );

  // 3. Guardrails event log
  renderGuardrailsLog(traces);
}

function renderGuardrailsLog(traces) {
  const container = document.getElementById('guardrails-log');
  if (!container) return;

  // Filter traces that have guardrails applied or notable events
  const events = traces.filter(t =>
    t.guardrails_applied ||
    t.hallucination_grade === 'hallucinated' ||
    t.answer_grade === 'not_useful'
  );

  if (events.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding:2rem">
        <i data-lucide="shield-off"></i>
        <p>No guardrail events recorded</p>
        <p class="hint">Events appear when queries are filtered, blocked, or flagged</p>
      </div>`;
    if (window.lucide) lucide.createIcons();
    return;
  }

  container.innerHTML = `
    <table class="trace-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Query</th>
          <th class="center">Guardrails</th>
          <th class="center">Hallucination</th>
          <th class="center">Quality</th>
        </tr>
      </thead>
      <tbody>
        ${events.slice(0, 30).map(t => `
          <tr>
            <td style="font-size:0.75rem;white-space:nowrap">${new Date(t.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}</td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(t.query?.slice(0, 60)) || '—'}</td>
            <td class="center">${t.guardrails_applied
              ? '<span class="badge badge-success">Applied</span>'
              : '<span class="badge badge-neutral">Bypassed</span>'}</td>
            <td class="center">${t.hallucination_grade === 'hallucinated'
              ? '<span class="badge badge-danger">Hallucinated</span>'
              : '<span class="badge badge-success">Grounded</span>'}</td>
            <td class="center">${t.answer_grade === 'not_useful'
              ? '<span class="badge badge-warning">Not Useful</span>'
              : '<span class="badge badge-success">Useful</span>'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}
