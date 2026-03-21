import { useState, useEffect } from 'react';
import { api } from '../api';
import { PageHeader, Spinner, Card, MetricCard, Tabs, SectionTitle, DataTable } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function Validation() {
  const [data, setData] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    api.getValidation().then(d => {
      const overall  = d.overall  ?? {};
      const perTime  = d.per_time ?? [];
      setData({ overall, perTime });
    });
  }, []);

  if (!data) return <Spinner text="Running validation…" />;

  const { overall, perTime } = data;
  const maxRMSE = Math.max(...perTime.map(p => p.RMSE ?? 0));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1200 }}>
      <PageHeader
        title="Model Validation"
        subtitle="Comprehensive performance against HYDRUS-1D reference — Moriasi et al. 2007 criteria"
        badge="Validation"
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
        <MetricCard label="Overall R²"   value={overall.R2?.toFixed(5)   ?? 'N/A'} color="blue" />
        <MetricCard label="Overall RMSE" value={`${overall.RMSE?.toFixed(4) ?? 'N/A'} m`} color="amber" />
        <MetricCard label="Overall MAE"  value={`${overall.MAE?.toFixed(4)  ?? 'N/A'} m`} color="green" />
        <MetricCard label="Overall NSE"  value={overall.NSE?.toFixed(5)  ?? 'N/A'} color="purple" />
      </div>

      <Card>
        <Tabs
          tabs={[
            { key: 'overview',  label: 'R² & RMSE' },
            { key: 'temporal',  label: 'Temporal errors' },
            { key: 'table',     label: 'Full table' },
          ]}
          active={tab} onChange={setTab}
        />

        {tab === 'overview' && perTime.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Chart
              data={[
                { x: perTime.map(p => p.time), y: perTime.map(p => p.R2 ?? 0),
                  type: 'scatter', mode: 'lines+markers', name: 'R²',
                  line: { color: '#2f81f7', width: 2 }, marker: { size: 5 } },
                { x: [perTime[0].time, perTime.at(-1).time], y: [0.99, 0.99],
                  type: 'scatter', mode: 'lines', name: 'R² = 0.99',
                  line: { color: '#3fb950', dash: 'dash', width: 1 } },
              ]}
              layout={{
                xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                yaxis: { title: { text: 'R²', font: { size: 11 } } },
                legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
              }}
              height={360}
            />
            <Chart
              data={[{
                x: perTime.map(p => `Day ${p.time}`),
                y: perTime.map(p => p.RMSE ?? 0),
                type: 'bar',
                marker: { color: perTime.map(p =>
                  (p.RMSE ?? 0) < maxRMSE * 0.3 ? '#3fb950'
                  : (p.RMSE ?? 0) < maxRMSE * 0.6 ? '#d29922'
                  : '#f85149'
                )},
                name: 'RMSE',
              }]}
              layout={{
                xaxis: { title: { text: 'Timestep', font: { size: 11 } } },
                yaxis: { title: { text: 'RMSE (m)', font: { size: 11 } } },
              }}
              height={300}
            />
          </div>
        )}

        {tab === 'temporal' && (
          <Chart
            data={[
              { x: perTime.map(p => p.time), y: perTime.map(p => p.MAE ?? 0),
                type: 'scatter', mode: 'lines+markers', name: 'MAE',
                line: { color: '#2f81f7', width: 2 }, marker: { size: 5 } },
              { x: perTime.map(p => p.time), y: perTime.map(p => p.RMSE ?? 0),
                type: 'scatter', mode: 'lines+markers', name: 'RMSE',
                line: { color: '#f85149', width: 2, dash: 'dash' }, marker: { size: 5 } },
            ]}
            layout={{
              xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
              yaxis: { title: { text: 'Error magnitude (m)', font: { size: 11 } } },
              legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
            }}
            height={440}
          />
        )}

        {tab === 'table' && (
          <DataTable
            columns={[
              { key: 'time', label: 'Day' },
              { key: 'R2',   label: 'R²',   render: v => v?.toFixed(6) },
              { key: 'RMSE', label: 'RMSE (m)', render: v => v?.toFixed(5) },
              { key: 'MAE',  label: 'MAE (m)',  render: v => v?.toFixed(5) },
              { key: 'NSE',  label: 'NSE',   render: v => v?.toFixed(5) },
              { key: 'KGE',  label: 'KGE',   render: v => v?.toFixed(5) },
              { key: 'NSE',  label: 'Rating (NSE)',
                render: v => v > 0.75 ? '✓ Very Good' : v > 0.65 ? '~ Good'
                  : v > 0.5 ? '~ Satisfactory' : '✗ Unsatisfactory' },
            ]}
            rows={perTime}
            keyFn={r => r.time}
          />
        )}
      </Card>

      {/* Benchmark card */}
      <Card>
        <SectionTitle>Moriasi et al. 2007 — performance thresholds</SectionTitle>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, fontSize: 11 }}>
          {[
            ['Very Good',     'NSE > 0.75',  '#3fb950'],
            ['Good',          'NSE > 0.65',  '#3fb950'],
            ['Satisfactory',  'NSE > 0.50',  '#d29922'],
            ['Unsatisfactory','NSE ≤ 0.50',  '#f85149'],
          ].map(([label, crit, color]) => (
            <div key={label} style={{ padding: '8px 12px', borderRadius: 6,
              border: `1px solid ${color}33`, background: `${color}0a` }}>
              <div style={{ color, fontWeight: 600, marginBottom: 2 }}>{label}</div>
              <div style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 10 }}>{crit}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
