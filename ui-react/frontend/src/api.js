const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

const post = (path, body) =>
  request(path, { method: 'POST', body: JSON.stringify(body) });

export const api = {
  // GET
  defaults:         ()     => request('/defaults'),
  health:           ()     => request('/health'),
  getModelInfo:     ()     => request('/model-info'),
  getValidation:    ()     => request('/validation'),
  getLossHistory:   ()     => request('/loss-history'),

  // POST
  predict:          (b)    => post('/predict', b),
  predictGrid:      (b)    => post('/predict-grid', b),
  factorOfSafety:   (b)    => post('/factor-of-safety', b),
  fsGrid:           (b)    => post('/fs-grid', b),
  soilProperties:   (b)    => post('/soil-properties', b),
  hydrusComparison: (b)    => post('/hydrus-comparison', b),
  getPDEResidual:   (b)    => post('/pde-residual', b),
  getUncertainty:   (b)    => post('/uncertainty', b),
  getCriticalSlip:  (b)    => post('/critical-slip', b),
  sensitivity:      (b)    => post('/sensitivity', b),
  exportData:       (b)    => post('/export', b),
};
