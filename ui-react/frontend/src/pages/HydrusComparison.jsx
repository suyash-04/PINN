import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function HydrusComparison() {
  const { defaults } = useApp();
  const [times, setTimes] = useState([0, 60, 90, 96, 123]);
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('overlay');

  useEffect(() => {
    if (!times.length) return;
    api.hydrusComparison({ times }).then(d => {
      // Compute overall metrics from per-time comparisons
      const allHydrus = d.comparisons.flatMap(c => c.psi_hydrus);
      const allPinn = d.comparisons.flatMap(c => c.psi_pinn);
      const allErr = allPinn.map((p, i) => p - allHydrus[i]);
      const meanObs = allHydrus.reduce((a, b) => a + b, 0) / allHydrus.length;
      const ssRes = allErr.reduce((a, e) => a + e * e, 0);
      const ssTot = allHydrus.reduce((a, h) => a + (h - meanObs) ** 2, 0);
      const r2 = 1 - ssRes / Math.max(ssTot, 1e-10);
      const rmse = Math.sqrt(ssRes / allErr.length);
      const mae = allErr.reduce((a, e) => a + Math.abs(e), 0) / allErr.length;
      setData({
        ...d,
        overall: { r2, rmse, mae },
        scatter: { hydrus: allHydrus, pinn: allPinn },
      });
    });
  }, [times]);

  const toggle = t => setTimes(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="HYDRUS-1D vs PINN Comparison" subtitle="Quantitative comparison of simulation data against neural network predictions" icon="🔬" />

      <Card>
        <p className="text-xs font-semibold text-slate-500 mb-2">SELECT TIMESTEPS</p>
        <div className="flex flex-wrap gap-2">
          {defaults.available_times.map(t => (
            <button key={t} onClick={() => toggle(t)}
              className={`px-3 py-1 text-xs rounded-full font-medium transition
                ${times.includes(t) ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
              Day {t}
            </button>
          ))}
        </div>
      </Card>

      {!data ? <Spinner /> : (
        <>
          {/* Overall metrics */}
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="R²" value={data.overall.r2.toFixed(4)} color="blue" />
            <MetricCard label="RMSE (m)" value={data.overall.rmse.toFixed(2)} color="amber" />
            <MetricCard label="MAE (m)" value={data.overall.mae.toFixed(2)} color="purple" />
          </div>

          <Card>
            <Tabs tabs={[
              { key: 'overlay', label: '📈 Overlay' },
              { key: 'error', label: '📊 Error Profile' },
              { key: 'scatter', label: '🔘 Scatter Plot' },
            ]} active={tab} onChange={setTab} />

            {tab === 'overlay' && (
              <Chart
                data={data.comparisons.flatMap((c, i) => [
                  { x: c.psi_hydrus, y: c.z, type: 'scatter', mode: 'markers',
                    name: `HYDRUS t=${c.time}d`, marker: { color: COLORS[i], size: 5, symbol: 'circle-open' },
                    legendgroup: `t${c.time}` },
                  { x: c.psi_pinn, y: c.z, type: 'scatter', mode: 'lines',
                    name: `PINN t=${c.time}d`, line: { color: COLORS[i], width: 2.5 },
                    legendgroup: `t${c.time}` },
                ])}
                layout={{
                  xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 520, legend: { orientation: 'h', y: 1.08, xanchor: 'center', x: 0.5 },
                }}
              />
            )}

            {tab === 'error' && (
              <Chart
                data={data.comparisons.map((c, i) => ({
                  x: c.abs_error, y: c.z, type: 'scatter', mode: 'lines+markers',
                  name: `t=${c.time}d`, line: { color: COLORS[i] }, marker: { size: 3 },
                }))}
                layout={{
                  xaxis: { title: '|ψ_PINN − ψ_HYDRUS| (m)' },
                  yaxis: { title: 'Depth (m)', autorange: 'reversed' }, height: 520,
                }}
              />
            )}

            {tab === 'scatter' && (
              <Chart
                data={[
                  { x: data.scatter.hydrus, y: data.scatter.pinn, type: 'scatter', mode: 'markers',
                    marker: { color: '#2563eb', size: 4, opacity: 0.5 }, name: 'Data' },
                  { x: [Math.min(...data.scatter.hydrus), Math.max(...data.scatter.hydrus)],
                    y: [Math.min(...data.scatter.hydrus), Math.max(...data.scatter.hydrus)],
                    type: 'scatter', mode: 'lines', line: { color: '#dc2626', dash: 'dash' }, name: '1:1' },
                ]}
                layout={{
                  xaxis: { title: 'ψ HYDRUS (m)' }, yaxis: { title: 'ψ PINN (m)' }, height: 500,
                  annotations: [{ x: 0.05, y: 0.95, xref: 'paper', yref: 'paper', showarrow: false,
                    text: `R²=${data.overall.r2.toFixed(4)}<br>RMSE=${data.overall.rmse.toFixed(2)}m`,
                    font: { size: 13 }, bgcolor: '#f1f5f9', bordercolor: '#cbd5e1', borderpad: 6 }],
                }}
              />
            )}
          </Card>

          {/* Per-time stats table */}
          <Card>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">📋 Per-Timestep Statistics</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-left text-slate-500">
                    <th className="p-2">Time (d)</th><th className="p-2">R²</th><th className="p-2">RMSE</th><th className="p-2">MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {data.comparisons.map(c => (
                    <tr key={c.time} className="border-b border-slate-100">
                      <td className="p-2 font-medium">{c.time}</td>
                      <td className="p-2">{c.r2.toFixed(4)}</td>
                      <td className="p-2">{c.rmse.toFixed(3)}</td>
                      <td className="p-2">{c.mae.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
