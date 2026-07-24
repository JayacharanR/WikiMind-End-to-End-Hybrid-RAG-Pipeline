// ═══════════════════════════════════════════════════════════════════════════
//  WikiMind Dashboard — API Client
//  Fetch wrappers for backend endpoints.
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = window.location.origin;

/**
 * Fetch aggregated dashboard metrics.
 * @returns {Promise<Object>} { summary, recent_traces }
 */
export async function fetchMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch /api/metrics:', err);
    return null;
  }
}

/**
 * Fetch full trace list.
 * @param {number} limit - max traces to return
 * @returns {Promise<Object>} { traces }
 */
export async function fetchTraces(limit = 100) {
  try {
    const res = await fetch(`${API_BASE}/api/traces?limit=${limit}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch /api/traces:', err);
    return null;
  }
}

/**
 * Fetch evaluation benchmark results.
 * @returns {Promise<Object>} { results }
 */
export async function fetchEvalResults() {
  try {
    const res = await fetch(`${API_BASE}/api/eval-results`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch /api/eval-results:', err);
    return null;
  }
}

/**
 * Fetch component health status.
 * @returns {Promise<Object>} { status, components }
 */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch /health:', err);
    return null;
  }
}
