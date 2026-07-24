// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — Trace Table + Waterfall
// ═══════════════════════════════════════════════════════════════════════════

const expandedTraces = new Set();

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function truncate(str, len = 60) {
  if (!str) return '—';
  return str.length > len ? str.slice(0, len) + '…' : str;
}

function attrBadge(attr) {
  switch (attr) {
    case 'rag_grounded': return '<span class="badge badge-success">RAG Grounded</span>';
    case 'parametric_risk': return '<span class="badge badge-warning">Parametric Risk</span>';
    default: return '<span class="badge badge-neutral">Unknown</span>';
  }
}

function gradeBadge(grade, type) {
  if (type === 'hallucination') {
    return grade === 'grounded'
      ? '<span class="badge badge-success">Grounded</span>'
      : '<span class="badge badge-danger">Hallucinated</span>';
  }
  return grade === 'useful'
    ? '<span class="badge badge-success">Useful</span>'
    : '<span class="badge badge-warning">Not Useful</span>';
}

/**
 * Render the trace table.
 * @param {string} tbodyId - ID of the <tbody> element
 * @param {Array} traces - trace objects from API
 */
export function renderTraceTable(tbodyId, traces) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;

  if (!traces || traces.length === 0) {
    tbody.innerHTML = `
      <tr><td colspan="7">
        <div class="empty-state">
          <i data-lucide="inbox"></i>
          <p>No traces yet</p>
          <p class="hint">Send a query through the chat UI to see traces here</p>
        </div>
      </td></tr>`;
    if (window.lucide) lucide.createIcons();
    return;
  }

  tbody.innerHTML = traces.map((t, i) => {
    const isExpanded = expandedTraces.has(i);
    const latencyColor = t.latency_ms < 3000 ? 'var(--accent-emerald)' :
                         t.latency_ms < 8000 ? 'var(--accent-amber)' : 'var(--accent-rose)';
    const provPct = (t.provenance_score * 100).toFixed(0);

    return `
      <tr class="trace-row ${isExpanded ? 'expanded' : ''}" onclick="window.__toggleTrace(${i})">
        <td style="width:32px; padding-left:1rem">
          <span class="chevron"><i data-lucide="chevron-down" style="width:14px;height:14px"></i></span>
        </td>
        <td>
          <div style="font-size:0.75rem;color:var(--text-muted)">${formatDate(t.timestamp)}</div>
          <div style="font-size:0.8rem">${formatTime(t.timestamp)}</div>
        </td>
        <td style="max-width:280px">
          <div style="font-size:0.8rem;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${truncate(t.query, 55)}</div>
        </td>
        <td class="mono right" style="color:${latencyColor}">${(t.latency_ms / 1000).toFixed(2)}s</td>
        <td class="mono right">${t.steps}</td>
        <td class="center">${provPct}%</td>
        <td class="center">${attrBadge(t.attribution)}</td>
      </tr>
      <tr>
        <td colspan="7" style="padding:0; border:none">
          <div class="trace-detail ${isExpanded ? 'open' : ''}" id="trace-detail-${i}">
            <div class="trace-detail-inner">
              ${renderDetailContent(t)}
            </div>
          </div>
        </td>
      </tr>`;
  }).join('');

  if (window.lucide) lucide.createIcons();
}

function renderDetailContent(trace) {
  const citationEntries = trace.citation_map ? Object.entries(trace.citation_map) : [];

  return `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">
      <!-- Left: Query & Generation -->
      <div>
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.5rem">Query</div>
        <div style="font-size:0.8rem;color:var(--text-primary);margin-bottom:1rem;line-height:1.5">${trace.query || '—'}</div>

        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.5rem">Generation</div>
        <div style="font-size:0.8rem;color:var(--text-secondary);line-height:1.6;max-height:200px;overflow-y:auto">${trace.generation || '—'}</div>
      </div>

      <!-- Right: Metadata -->
      <div>
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.5rem">Metadata</div>
        <div class="stat-grid" style="grid-template-columns:repeat(2,1fr);gap:0.5rem;margin-bottom:1rem">
          <div class="stat-item" style="padding:0.5rem 0.75rem">
            <div class="stat-label">Latency</div>
            <div class="stat-value" style="font-size:1rem">${(trace.latency_ms / 1000).toFixed(2)}s</div>
          </div>
          <div class="stat-item" style="padding:0.5rem 0.75rem">
            <div class="stat-label">Steps</div>
            <div class="stat-value" style="font-size:1rem">${trace.steps}</div>
          </div>
          <div class="stat-item" style="padding:0.5rem 0.75rem">
            <div class="stat-label">Documents</div>
            <div class="stat-value" style="font-size:1rem">${trace.document_count || 0}</div>
          </div>
          <div class="stat-item" style="padding:0.5rem 0.75rem">
            <div class="stat-label">Provenance</div>
            <div class="stat-value" style="font-size:1rem">${(trace.provenance_score * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem">
          ${attrBadge(trace.attribution)}
          ${trace.guardrails_applied ? '<span class="badge badge-info">Guardrails ✓</span>' : '<span class="badge badge-neutral">No Guardrails</span>'}
          ${gradeBadge(trace.hallucination_grade, 'hallucination')}
          ${gradeBadge(trace.answer_grade, 'answer')}
        </div>

        ${trace.expanded_queries && trace.expanded_queries.length > 0 ? `
          <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.25rem;margin-top:0.5rem">Expanded Queries</div>
          <ul style="font-size:0.75rem;color:var(--text-tertiary);list-style:disc;padding-left:1rem">
            ${trace.expanded_queries.map(q => `<li>${truncate(q, 80)}</li>`).join('')}
          </ul>
        ` : ''}

        ${citationEntries.length > 0 ? `
          <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);margin-bottom:0.25rem;margin-top:0.75rem">Citations</div>
          <div style="font-size:0.75rem;color:var(--text-tertiary)">
            ${citationEntries.map(([k, v]) => `<div>[${k}] ${truncate(v.title || v.content || '', 60)}</div>`).join('')}
          </div>
        ` : ''}
      </div>
    </div>`;
}

// Global toggle handler (needed for onclick in innerHTML)
window.__toggleTrace = function(index) {
  if (expandedTraces.has(index)) {
    expandedTraces.delete(index);
  } else {
    expandedTraces.add(index);
  }
  // Re-render just the affected row
  const row = document.querySelectorAll('.trace-row')[index];
  const detail = document.getElementById(`trace-detail-${index}`);
  if (row && detail) {
    row.classList.toggle('expanded');
    detail.classList.toggle('open');
  }
};
