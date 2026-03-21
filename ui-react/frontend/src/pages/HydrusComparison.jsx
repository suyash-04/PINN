import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs, TimeToggle, SectionTitle, DataTable } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

function computeMetrics(hydrus, pinn) {
  const err = pinn.map((p, i) => p - hydrus[i]);
  const mean = hydrus.reduce((a, b) => a + b, 0) / hydrus.length;
  const ssRes = err.reduce((a, e) => a + e * e, 0);
  const ssTot = hydrus.reduce((a, h) => a + (h - mean) ** 2, 0);
  return {
    r2:   1 - ssRes / Math.max(ssTot, 1e-10),
    rmse: Math.sqrt(ssRes / err.length),
    mae:  err.reduce((a, e) => a + Math.abs(e), 0) / err.length,
  };
}

export default function HydrusComparison() {
  const { defaults } = useApp();
  // Use first 5 available timesteps by default — no hardcoding
  const defaultTimes = defaults.available_times.slice(0, 5);
  const [times, setTimes] = useState(defaultTimes);
  const [data, setData]   = useState(null);
  const [tab, setTab]     = useState('overlay');

  useEffect(() => {
    if (!times.length) { setData(null); return; }
    api.hydrusComparison({ times }).then(d => {
      const allHydrus = d.comparisons.flatMap(c => c.psi_hydrus);
      const allPinn   = d.comparisons.flatMap(c => c.psi_pinn);
      const overall   = computeMetrics(allHydrus, allPinn);
      setData({ ...d, overall, scatter: { hydrus: allHydrus, pinn: allPinn } });
    });
  }, [times]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="HYDRUS-1D vs PINN"
        subtitle="Quantitative comparison of PINN predictions against reference numerical solutions"
        badge="Validation"
      />

      <Card>
        <SectionTitle>Select timesteps</SectionTitle>
        <TimeToggle times={defaults.available_times} selected={times} onChange={setTimes} label="Day" />
      </Card>

      {!data
        ? <Spinner text="Running comparison…" />
        : <>
            {/* Overall metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
              <MetricCard label="Overall R²"   value={data.overall.r2.toFixed(5)}   color="blue" />
              <MetricCard label="RMSE (m)"     value={data.overall.rmse.toFixed(4)}  color="amber" />
              <MetricCard label="MAE (m)"      value={data.overall.mae.toFixed(4)}   color="purple" />
            </div>

            <Card>
              <Tabs
                tabs={[
                  { key: 'overlay', label: 'Profile overlay' },
                  { key: 'error',   label: 'Error profiles' },
                  { key: 'scatter', label: 'Scatter plot' },
                  { key: 'table',   label: 'Statistics table' },
                ]}
                active={tab} onChange={setTab}
              />

              {tab === 'overlay' && (
                <Chart
                  data={data.comparisons.flatMap((c, i) => [
                    { x: c.psi_hydrus, y: c.z, type: 'scatter', mode: 'markers',
                      name: `HYDRUS day ${c.time}`,
                      marker: { color: COLORS[i % COLORS.length], size: 4, symbol: 'circle-open' },
                      legendgroup: `t${c.time}` },
                    { x: c.psi_pinn, y: c.z, type: 'scatter', mode: 'lines',
                      name: `PINN day ${c.time}`,
                      line: { color: COLORS[i % COLORS.length], width: 2 },
                      legendgroup: `t${c.time}` },
                  ])}
                  layout={{
                    xaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
                    yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                    legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
                  }}
                  height={520}
                />
              )}

              {tab === 'error' && (
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
                  }}
                  height={520}
                />
              )}

              {tab === 'scatter' && (() => {
                const psiMin = Math.min(...data.scatter.hydrus);
                const psiMax = Math.max(...data.scatter.hydrus);
                return (
                  <Chart
                    data={[
                      { x: data.scatter.hydrus, y: data.scatter.pinn, type: 'scatter', mode: 'markers',
                        marker: { color: '#2f81f7', size: 3, opacity: 0.45 }, name: 'Points' },
                      { x: [psiMin, psiMax], y: [psiMin, psiMax], type: 'scatter', mode: 'lines',
                        line: { color: '#f85149', dash: 'dash', width: 1.5 }, name: '1:1 line' },
                    ]}
                    layout={{
                      xaxis: { title: { text: 'ψ HYDRUS (m)', font: { size: 11 } } },
                      yaxis: { title: { text: 'ψ PINN (m)', font: { size: 11 } } },
                      annotations: [{
                        x: 0.05, y: 0.97, xref: 'paper', yref: 'paper', showarrow: false,
                        text: `R² = ${data.overall.r2.toFixed(5)}<br>RMSE = ${data.overall.rmse.toFixed(3)} m`,
                        font: { size: 11, color: '#e6edf3', family: "'IBM Plex Mono', monospace" },
                        align: 'left', bgcolor: 'rgba(22,27,34,.85)',
                        bordercolor: '#30363d', borderpad: 6,
                      }],
                    }}
                    height={500}
                  />
                );
              })()}

              {tab === 'table' && (
                <DataTable
                  columns={[
                    { key: 'time', label: 'Day' },
                    { key: 'r2',   label: 'R²',   render: v => v?.toFixed(5) },
                    { key: 'rmse', label: 'RMSE (m)', render: v => v?.toFixed(4) },
                    { key: 'mae',  label: 'MAE (m)',  render: v => v?.toFixed(4) },
                    { key: 'r2',   label: 'Quality',
                      render: v => v > 0.99 ? '✓ Excellent' : v > 0.95 ? '~ Good' : '✗ Review' },
                  ]}
                  rows={data.comparisons}
                  keyFn={r => r.time}
                />
              )}
            </Card>
          </>
      }
    </div>
  );
}
