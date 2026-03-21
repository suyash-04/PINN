import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, TimeToggle, SectionTitle, DataTable } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function ErrorAnalysis() {
  const { defaults } = useApp();
  const defaultTimes = defaults.available_times.slice(0, 5);
  const [selectedTimes, setSelectedTimes] = useState(defaultTimes);
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!selectedTimes.length) { setData(null); return; }
    api.hydrusComparison({ times: selectedTimes }).then(d => {
      const allHydrus = d.comparisons.flatMap(c => c.psi_hydrus);
      const allPinn   = d.comparisons.flatMap(c => c.psi_pinn);
      const allErr    = allPinn.map((p, i) => p - allHydrus[i]);
      const mean      = allHydrus.reduce((a, b) => a + b, 0) / allHydrus.length;
      const ssRes     = allErr.reduce((a, e) => a + e * e, 0);
      const ssTot     = allHydrus.reduce((a, h) => a + (h - mean) ** 2, 0);
      const r2        = 1 - ssRes / Math.max(ssTot, 1e-10);
      setData({ ...d, overall: { r2 } });
    });
  }, [selectedTimes]);

  if (!data) return <Spinner text="Computing errors…" />;

  const allErrors = data.comparisons.flatMap(c => c.abs_error);
  const maxErr  = Math.max(...allErrors);
  const meanErr = allErrors.reduce((a, b) => a + b, 0) / allErrors.length;
  const sorted  = [...allErrors].sort((a, b) => a - b);
  const p95     = sorted[Math.floor(sorted.length * 0.95)];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="Error Analysis"
        subtitle="Spatial and statistical breakdown of PINN prediction errors vs HYDRUS-1D"
        badge="Diagnostics"
      />

      <Card>
        <SectionTitle>Timestep selection</SectionTitle>
        <TimeToggle times={defaults.available_times} selected={selectedTimes}
          onChange={setSelectedTimes} label="Day" />
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
        <MetricCard label="Overall R²"    value={data.overall.r2.toFixed(5)} color="blue" />
        <MetricCard label="Max error"     value={`${maxErr.toFixed(3)} m`}   color="red" />
        <MetricCard label="Mean error"    value={`${meanErr.toFixed(3)} m`}  color="amber" />
        <MetricCard label="95th pct error" value={`${p95.toFixed(3)} m`}    color="purple" />
      </div>

      <Card>
        <SectionTitle>Absolute error profiles — ψ residual vs depth</SectionTitle>
        <Chart
          data={data.comparisons.map((c, i) => ({
            x: c.abs_error, y: c.z, type: 'scatter', mode: 'lines+markers',
            name: `Day ${c.time}`,
            line: { color: COLORS[i % COLORS.length], width: 1.8 },
            marker: { size: 3 },
          }))}
          layout={{
            xaxis: { title: { text: '|ψ_PINN − ψ_HYDRUS| (m)', font: { size: 11 } } },
            yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
            legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
          }}
          height={420}
        />
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card>
          <SectionTitle>Error distribution</SectionTitle>
          <Chart
            data={[{
              x: allErrors, type: 'histogram', nbinsx: 45,
              marker: { color: 'rgba(47,129,247,.5)' }, name: '|error|',
            }]}
            layout={{
              xaxis: { title: { text: '|Error| (m)', font: { size: 11 } } },
              yaxis: { title: { text: 'Count', font: { size: 11 } } },
              bargap: 0.04,
            }}
            height={320}
          />
        </Card>

        <Card>
          <SectionTitle>RMSE by timestep</SectionTitle>
          <Chart
            data={[{
              x: data.comparisons.map(c => `Day ${c.time}`),
              y: data.comparisons.map(c => c.rmse),
              type: 'bar',
              marker: { color: data.comparisons.map((_, i) => COLORS[i % COLORS.length]) },
              name: 'RMSE',
            }]}
            layout={{
              xaxis: { title: { text: 'Timestep', font: { size: 11 } } },
              yaxis: { title: { text: 'RMSE (m)', font: { size: 11 } } },
            }}
            height={320}
          />
        </Card>
      </div>

      <Card>
        <SectionTitle>Relative error heatmap — |ε| / |ψ_HYDRUS| (%)</SectionTitle>
        <Chart
          data={[{
            z: data.comparisons.map(c =>
              c.abs_error.map((e, i) =>
                Math.abs(c.psi_hydrus[i]) > 1 ? (e / Math.abs(c.psi_hydrus[i])) * 100 : 0
              )
            ),
            x: data.comparisons[0]?.z ?? [],
            y: data.comparisons.map(c => `Day ${c.time}`),
            type: 'heatmap', colorscale: 'YlOrRd',
            colorbar: { title: { text: '%', side: 'right' }, thickness: 14,
              tickfont: { size: 10, color: '#7d8590' } },
          }]}
          layout={{
            xaxis: { title: { text: 'Depth (m)', font: { size: 11 } } },
            margin: { l: 70, r: 80, t: 20, b: 50 },
          }}
          height={280}
        />
      </Card>

      <Card>
        <SectionTitle>Per-timestep statistics table</SectionTitle>
        <DataTable
          columns={[
            { key: 'time', label: 'Day' },
            { key: 'r2',   label: 'R²',   render: v => v?.toFixed(5) },
            { key: 'rmse', label: 'RMSE (m)', render: v => v?.toFixed(4) },
            { key: 'mae',  label: 'MAE (m)',  render: v => v?.toFixed(4) },
            { key: 'r2',   label: 'Rating',
              render: v => v > 0.999 ? '✓ Excellent' : v > 0.99 ? '~ Very good' : v > 0.95 ? '~ Good' : '✗ Review' },
          ]}
          rows={data.comparisons}
          keyFn={r => r.time}
        />
      </Card>
    </div>
  );
}
