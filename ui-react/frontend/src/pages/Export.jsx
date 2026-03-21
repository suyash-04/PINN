import { useState } from 'react';
import { useApp } from '../context';
import { api } from '../api';
import { PageHeader, Card, Spinner, SectionTitle, Button } from '../components/ui';

const FORMATS = [
  { key: 'csv',  label: 'CSV',    desc: 'Comma-separated — spreadsheets & R/pandas' },
  { key: 'json', label: 'JSON',   desc: 'Structured — programmatic consumption' },
];

const DATASETS = [
  { key: 'predictions',    label: 'PINN Predictions',  desc: 'Full ψ(t, z) grid' },
  { key: 'factor_of_safety', label: 'Factor of Safety', desc: 'FS(t, z) grid' },
  { key: 'comparison',    label: 'HYDRUS Comparison',  desc: 'PINN vs HYDRUS + error metrics' },
];

export default function Export() {
  // Pull defaults from context — NOT bare variable
  const { geo, norm, defaults } = useApp();
  const tMax = defaults.norm.t_max;

  const [format,  setFormat]  = useState('csv');
  const [dataset, setDataset] = useState('predictions');
  const [loading, setLoading] = useState(false);
  const [status,  setStatus]  = useState(null);

  const handleExport = async () => {
    setLoading(true);
    setStatus(null);
    try {
      const result = await api.exportData({
        z_min: 0.5, z_max: 40, z_res: 50,
        t_min: 0, t_max: tMax, t_res: 50,
        geo, norm,
      });

      if (result.data) {
        let content, ext, mime;
        if (format === 'json') {
          content = JSON.stringify(result.data, null, 2);
          ext = 'json'; mime = 'application/json';
        } else {
          const rows    = result.data;
          const headers = Object.keys(rows[0]);
          const lines   = [headers.join(','), ...rows.map(r => headers.map(h => r[h]).join(','))];
          content = lines.join('\n');
          ext = 'csv'; mime = 'text/csv';
        }
        const blob = new Blob([content], { type: mime });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `pinn_${dataset}_${new Date().toISOString().slice(0,10)}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        setStatus({ ok: true, msg: `Exported ${result.n_rows?.toLocaleString()} rows` });
      } else {
        setStatus({ ok: true, msg: 'Export complete' });
      }
    } catch (e) {
      setStatus({ ok: false, msg: e.message });
    }
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 800 }}>
      <PageHeader
        title="Export Data"
        subtitle="Download PINN predictions and analysis results"
        badge="Export"
      />

      {/* Format */}
      <Card>
        <SectionTitle>Output format</SectionTitle>
        <div style={{ display: 'flex', gap: 10 }}>
          {FORMATS.map(f => (
            <button key={f.key} onClick={() => setFormat(f.key)} style={{
              flex: 1, padding: '12px 16px', borderRadius: 8, cursor: 'pointer',
              border: format === f.key ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: format === f.key ? 'rgba(47,129,247,.1)' : 'transparent',
              textAlign: 'left',
            }}>
              <div style={{ fontSize: 13, fontWeight: 600,
                color: format === f.key ? 'var(--accent)' : 'var(--text)',
                fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
                {f.label}
              </div>
              <div style={{ fontSize: 10, color: 'var(--muted)' }}>{f.desc}</div>
            </button>
          ))}
        </div>
      </Card>

      {/* Dataset */}
      <Card>
        <SectionTitle>Dataset</SectionTitle>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {DATASETS.map(d => (
            <button key={d.key} onClick={() => setDataset(d.key)} style={{
              display: 'flex', alignItems: 'center', gap: 12,
              padding: '10px 14px', borderRadius: 6, cursor: 'pointer',
              border: dataset === d.key ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: dataset === d.key ? 'rgba(47,129,247,.07)' : 'transparent',
              textAlign: 'left',
            }}>
              <div style={{
                width: 14, height: 14, borderRadius: '50%', flexShrink: 0,
                border: `2px solid ${dataset === d.key ? 'var(--accent)' : 'var(--border2)'}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {dataset === d.key && (
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)' }} />
                )}
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)', marginBottom: 2 }}>
                  {d.label}
                </div>
                <div style={{ fontSize: 10, color: 'var(--muted)' }}>{d.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </Card>

      {/* Parameter summary */}
      <Card style={{ background: 'rgba(255,255,255,.02)' }}>
        <SectionTitle>Grid parameters</SectionTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8,
          fontSize: 11, fontFamily: 'var(--font-mono)' }}>
          {[
            ['z range', `0.5 – 40 m,  50 pts`],
            ['t range', `0 – ${tMax} d,  50 pts`],
            ['β',       `${geo.beta}°`],
            ["c′",      `${geo.c_prime} kPa`],
          ].map(([k, v]) => (
            <div key={k} style={{ padding: '6px 10px', background: 'var(--border)',
              borderRadius: 5 }}>
              <div style={{ color: 'var(--muted)', fontSize: 9, marginBottom: 2 }}>{k}</div>
              <div style={{ color: 'var(--text)' }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Export button */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <Button onClick={handleExport} disabled={loading} variant="primary"
          style={{ padding: '10px 28px', fontSize: 13 }}>
          {loading ? 'Exporting…' : `Export ${DATASETS.find(d => d.key === dataset)?.label} → ${format.toUpperCase()}`}
        </Button>
        {status && (
          <span style={{ fontSize: 12,
            color: status.ok ? 'var(--accent-2)' : 'var(--danger)' }}>
            {status.ok ? '✓' : '✗'} {status.msg}
          </span>
        )}
      </div>
    </div>
  );
}
