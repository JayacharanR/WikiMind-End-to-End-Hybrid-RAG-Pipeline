// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — Evaluation Results Viewer
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Render evaluation benchmark results.
 * @param {Array} results - from /api/eval-results
 */
export function renderEvaluationTab(results) {
  const container = document.getElementById('eval-content');
  if (!container) return;

  if (!results || results.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i data-lucide="flask-conical"></i>
        <p>No evaluation results found</p>
        <p class="hint">Run the evaluation harness to see benchmark metrics here</p>
      </div>`;
    if (window.lucide) lucide.createIcons();
    return;
  }

  container.innerHTML = results.map(result => {
    const agg = result.aggregate || {};
    const perQuery = result.per_query || [];
    const filename = result.filename || 'unknown';

    return `
      <div class="card section-gap">
        <div class="card-header">
          <div class="card-title">
            <i data-lucide="file-bar-chart"></i>
            ${filename}
          </div>
          <span style="font-size:0.7rem;color:var(--text-muted)">${perQuery.length} queries</span>
        </div>
        <div class="card-body">
          <!-- Aggregate Metrics -->
          <div class="stat-grid" style="margin-bottom:1.25rem">
            ${renderMetricStat('Recall@5', agg.mean_recall_at_5, '#6366f1')}
            ${renderMetricStat('MRR', agg.mean_mrr, '#8b5cf6')}
            ${renderMetricStat('Accuracy', agg.mean_accuracy, '#10b981')}
            ${renderMetricStat('Latency P50', agg.latency_p50 ? `${agg.latency_p50.toFixed(2)}s` : '—', '#f59e0b')}
            ${renderMetricStat('Avg Steps', agg.mean_steps, '#38bdf8')}
          </div>

          <!-- Per-Query Table -->
          ${perQuery.length > 0 ? `
            <div style="overflow-x:auto">
              <table class="trace-table">
                <thead>
                  <tr>
                    <th>Question</th>
                    <th class="right">Recall@5</th>
                    <th class="right">MRR</th>
                    <th class="right">Accuracy</th>
                    <th class="right">Latency</th>
                    <th class="right">Steps</th>
                  </tr>
                </thead>
                <tbody>
                  ${perQuery.map(q => `
                    <tr>
                      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${q.question || '—'}</td>
                      <td class="mono right">${q.recall_at_5?.toFixed(2) ?? '—'}</td>
                      <td class="mono right">${q.mrr?.toFixed(3) ?? '—'}</td>
                      <td class="mono right">${q.answer_accuracy?.toFixed(2) ?? '—'}</td>
                      <td class="mono right">${q.latency ? q.latency.toFixed(2) + 's' : '—'}</td>
                      <td class="mono right">${q.steps ?? '—'}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          ` : ''}
        </div>
      </div>`;
  }).join('');

  if (window.lucide) lucide.createIcons();
}

function renderMetricStat(label, value, color) {
  const display = typeof value === 'number' ? value.toFixed(3) : (value || '—');
  return `
    <div class="stat-item">
      <div class="stat-label">${label}</div>
      <div class="stat-value" style="color:${color}">${display}</div>
    </div>`;
}
