// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — KPI Card Rendering
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Render KPI cards into the given container.
 * @param {string} containerId - DOM element ID
 * @param {Object} summary - metrics summary from /api/metrics
 */
export function renderKPICards(containerId, summary) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const ragPct = summary.total_queries > 0
    ? Math.round((summary.attribution_breakdown.rag_grounded / summary.total_queries) * 100)
    : 0;

  const cards = [
    {
      label: 'Total Queries',
      value: summary.total_queries.toLocaleString(),
      sub: `Cache hit rate: ${(summary.cache_hit_rate * 100).toFixed(1)}%`,
      icon: 'activity',
      accent: '#6366f1',
      accentBg: 'rgba(99,102,241,0.12)',
    },
    {
      label: 'P50 Latency',
      value: `${(summary.p50_latency_ms / 1000).toFixed(2)}s`,
      sub: `P95: ${(summary.p95_latency_ms / 1000).toFixed(2)}s`,
      icon: 'timer',
      accent: '#f59e0b',
      accentBg: 'rgba(245,158,11,0.12)',
    },
    {
      label: 'Avg Steps',
      value: summary.avg_steps.toFixed(1),
      sub: 'Agent graph transitions',
      icon: 'git-branch',
      accent: '#8b5cf6',
      accentBg: 'rgba(139,92,246,0.12)',
    },
    {
      label: 'Provenance Score',
      value: `${(summary.avg_provenance_score * 100).toFixed(0)}%`,
      sub: 'Avg citation verification',
      icon: 'shield-check',
      accent: '#10b981',
      accentBg: 'rgba(16,185,129,0.12)',
    },
    {
      label: 'RAG Grounded',
      value: `${ragPct}%`,
      sub: `${summary.attribution_breakdown.rag_grounded} of ${summary.total_queries}`,
      icon: 'database',
      accent: '#38bdf8',
      accentBg: 'rgba(56,189,248,0.12)',
    },
    {
      label: 'Guardrails',
      value: summary.guardrails_stats.applied.toLocaleString(),
      sub: `Bypassed: ${summary.guardrails_stats.bypassed}`,
      icon: 'shield',
      accent: '#f43f5e',
      accentBg: 'rgba(244,63,94,0.12)',
    },
  ];

  container.innerHTML = cards.map(c => `
    <div class="card kpi-card" style="--kpi-accent: ${c.accent}">
      <div class="card-body">
        <div class="kpi-top">
          <span class="kpi-label">${c.label}</span>
          <div class="kpi-icon" style="background: ${c.accentBg}; color: ${c.accent}">
            <i data-lucide="${c.icon}"></i>
          </div>
        </div>
        <div class="kpi-value">${c.value}</div>
        <div class="kpi-sub">${c.sub}</div>
      </div>
    </div>
  `).join('');

  // Re-init Lucide icons
  if (window.lucide) lucide.createIcons();
}
