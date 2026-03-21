import { useState } from 'react';
import { useApp } from '../context';
import { api } from '../api';
import { PageHeader, Card, Spinner } from '../components/ui';

const FORMATS = [
  { key: 'csv', label: 'CSV', icon: '📊', desc: 'Comma-separated values for spreadsheets' },
  { key: 'json', label: 'JSON', icon: '🔧', desc: 'Structured data for programmatic use' },
  { key: 'numpy', label: 'NumPy', icon: '🐍', desc: 'NumPy arrays for Python analysis' },
];

const DATASETS = [
  { key: 'predictions', label: 'PINN Predictions', desc: 'Full ψ(t,z) prediction grid' },
  { key: 'factor_of_safety', label: 'Factor of Safety', desc: 'FS grid over all times and depths' },
  { key: 'comparison', label: 'HYDRUS Comparison', desc: 'PINN vs HYDRUS-1D data with error metrics' },
  { key: 'soil_properties', label: 'Soil Properties', desc: 'Van Genuchten curves (θ, K, C vs ψ)' },
];

export default function Export() {
  const { geo, norm } = useApp();
  const [format, setFormat] = useState('csv');
  const [dataset, setDataset] = useState('predictions');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const handleExport = async () => {
    setLoading(true);
    setStatus(null);
    try {
      const result = await api.exportData({
        z_min: 0.5, z_max: 40, z_res: 50,
        t_min: 0, t_max: defaults.norm.t_max, t_res: 50,
        geo, norm,
      });
      // Create download from the response
      if (result.data) {
        let content, ext, mimeType;
        if (format === 'json') {
          content = JSON.stringify(result.data, null, 2);
          ext = 'json';
          mimeType = 'application/json';
        } else {
          // Convert array of objects to CSV
          const rows = result.data;
          const headers = Object.keys(rows[0]);
          const csvLines = [headers.join(','), ...rows.map(r => headers.map(h => r[h]).join(','))];
          content = csvLines.join('\n');
          ext = 'csv';
          mimeType = 'text/csv';
        }
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pinn_export_${new Date().toISOString().slice(0, 10)}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        setStatus({ type: 'success', msg: `Exported ${result.n_rows} rows!` });
      } else {
        setStatus({ type: 'success', msg: `Export ready: ${result.message || 'Success'}` });
      }
    } catch (e) {
      setStatus({ type: 'error', msg: `Export failed: ${e.message}` });
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <PageHeader title="Export Data" subtitle="Download PINN predictions and analysis results in various formats" icon="📤" />

      {/* Format selection */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Output Format</h3>
        <div className="grid grid-cols-3 gap-3">
          {FORMATS.map(f => (
            <button key={f.key} onClick={() => setFormat(f.key)}
              className={`p-4 rounded-xl border-2 text-left transition ${
                format === f.key ? 'border-blue-500 bg-blue-50' : 'border-slate-100 hover:border-slate-200'
              }`}>
              <div className="text-2xl mb-1">{f.icon}</div>
              <p className="text-sm font-semibold text-slate-800">{f.label}</p>
              <p className="text-[10px] text-slate-500">{f.desc}</p>
            </button>
          ))}
        </div>
      </Card>

      {/* Dataset selection */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Dataset</h3>
        <div className="space-y-2">
          {DATASETS.map(d => (
            <button key={d.key} onClick={() => setDataset(d.key)}
              className={`w-full p-3 rounded-lg border text-left flex items-center gap-3 transition ${
                dataset === d.key ? 'border-blue-500 bg-blue-50' : 'border-slate-100 hover:border-slate-200'
              }`}>
              <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                dataset === d.key ? 'border-blue-500' : 'border-slate-300'
              }`}>
                {dataset === d.key && <div className="w-2 h-2 rounded-full bg-blue-500" />}
              </div>
              <div>
                <p className="text-sm font-medium text-slate-700">{d.label}</p>
                <p className="text-[10px] text-slate-500">{d.desc}</p>
              </div>
            </button>
          ))}
        </div>
      </Card>

      {/* Current parameters summary */}
      <Card className="bg-slate-50">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Export Parameters</h3>
        <div className="grid grid-cols-4 gap-2 text-xs text-slate-600">
          <div><span className="text-slate-400">β:</span> {geo.beta}°</div>
          <div><span className="text-slate-400">c':</span> {geo.c_prime} Pa</div>
          <div><span className="text-slate-400">φ':</span> {geo.phi_prime}°</div>
          <div><span className="text-slate-400">γ:</span> {geo.gamma} N/m³</div>
        </div>
      </Card>

      {/* Export button */}
      <div className="flex items-center gap-4">
        <button onClick={handleExport} disabled={loading}
          className="px-8 py-3 bg-blue-600 text-white rounded-xl text-sm font-semibold hover:bg-blue-700 disabled:opacity-50 transition shadow-lg shadow-blue-200">
          {loading ? (
            <span className="flex items-center gap-2"><Spinner /> Exporting…</span>
          ) : (
            `📤 Export ${DATASETS.find(d => d.key === dataset)?.label} as ${format.toUpperCase()}`
          )}
        </button>

        {status && (
          <div className={`text-sm font-medium ${status.type === 'success' ? 'text-green-600' : 'text-red-600'}`}>
            {status.type === 'success' ? '✅' : '❌'} {status.msg}
          </div>
        )}
      </div>
    </div>
  );
}
