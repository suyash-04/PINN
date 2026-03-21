/* API client — all calls go through the Vite proxy to FastAPI */

const BASE = '/api';

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

function post(path, body) {
  return request(path, { method: 'POST', body: JSON.stringify(body) });
}

export const api = {
  /* ── GET endpoints ─── */
  defaults:       () => request('/defaults'),
  health:         () => request('/health'),
  getModelInfo:   () => request('/model-info'),
  getValidation:  (body) => body ? post('/validation', body) : request('/validation'),
  hydrusData:     () => request('/hydrus-data'),
  getLossHistory: () => request('/loss-history'),

  /* ── POST endpoints ─── */
  predict:        (body) => post('/predict', body),
  predictGrid:    (body) => post('/predict-grid', body),
  factorOfSafety: (body) => post('/factor-of-safety', body),
  fsGrid:         (body) => post('/fs-grid', body),
  getSoilProps:   (body) => post('/soil-properties', body),
  soilProperties: (body) => post('/soil-properties', body),
  hydrusComparison: (body) => post('/hydrus-comparison', body),
  getPDEResidual: (body) => post('/pde-residual', body),
  getUncertainty: (body) => post('/uncertainty', body),
  getCriticalSlip:(body) => post('/critical-slip', body),
  sensitivity:    (body) => post('/sensitivity', body),
  exportData:     (body) => post('/export', body),
};

