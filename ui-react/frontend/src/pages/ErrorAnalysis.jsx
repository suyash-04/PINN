import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function ErrorAnalysis() {
  const { geo, norm, defaults } = useApp();
  const [data, setData] = useState(null);
  const [selectedTimes, setSelectedTimes] = useState([0, 60, 90, 96, 123]);

  useEffect(() => {
    api.hydrusComparison({ times: selectedTimes }).then(d => {
      // Compute overall metrics
      const allHydrus = d.comparisons.flatMap(c => c.psi_hydrus);
      const allPinn = d.comparisons.flatMap(c => c.psi_pinn);
      const allErr = allPinn.map((p, i) => p - allHydrus[i]);
      const meanObs = allHydrus.reduce((a, b) => a + b, 0) / allHydrus.length;
      const ssRes = allErr.reduce((a, e) => a + e * e, 0);
      const ssTot = allHydrus.reduce((a, h) => a + (h - meanObs) ** 2, 0);
      const r2 = 1 - ssRes / Math.max(ssTot, 1e-10);
      setData({ ...d, overall: { r2 } });
    });
  }, [selectedTimes, geo, norm]);

  const toggle = t => setSelectedTimes(p => p.includes(t) ? p.filter(x => x !== t) : [...p, t]);

  if (!data) return <Spinner />;

  const allErrors = data.comparisons.flatMap(c => c.abs_error);
  const maxErr = Math.max(...allErrors);
  const meanErr = allErrors.reduce((a, b) => a + b, 0) / allErrors.length;
  const pct95 = [...allErrors].sort((a, b) => a - b)[Math.floor(allErrors.length * 0.95)];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Error Analysis" subtitle="Detailed error metrics and spatial/temporal error distribution" icon="🔍" />

      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="R²" value={data.overall.r2.toFixed(4)} color="blue" />
        <MetricCard label="Max Error" value={`${maxErr.toFixed(2)} m`} color="red" />
        <MetricCard label="Mean Error" value={`${meanErr.toFixed(2)} m`} color="amber" />
        <MetricCard label="95th Pctile" value={`${pct95.toFixed(2)} m`} color="purple" />
      </div>

      <Card>
        <p className="text-xs font-semibold text-slate-500 mb-2">TIMESTEPS</p>
        <div className="flex flex-wrap gap-2">
          {defaults.available_times.map(t => (
            <button key={t} onClick={() => toggle(t)}
              className={`px-3 py-1 text-xs rounded-full font-medium transition
                ${selectedTimes.includes(t) ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
              {t}d
            </button>
          ))}
        </div>
      </Card>

      {/* Error profiles per time */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Error Profiles by Depth</h3>
        <Chart
          data={data.comparisons.map((c, i) => ({
            x: c.abs_error, y: c.z, type: 'scatter', mode: 'lines+markers',
            name: `t=${c.time}d`, line: { color: COLORS[i % COLORS.length] }, marker: { size: 3 },
          }))}
          layout={{
            xaxis: { title: '|Error| (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
            height: 450,
          }}
        />
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {/* Error histogram */}
        <Card>
          <h3 className="text-sm font-semibold text-slate-700 mb-2">Error Distribution</h3>
          <Chart
            data={[{
              x: allErrors, type: 'histogram', nbinsx: 40,
              marker: { color: '#6366f1' }, name: '|Error|',
            }]}
            layout={{
              xaxis: { title: '|Error| (m)' }, yaxis: { title: 'Count' }, height: 350, bargap: 0.05,
            }}
          />
        </Card>

        {/* Per-time bar chart */}
        <Card>
          <h3 className="text-sm font-semibold text-slate-700 mb-2">RMSE by Timestep</h3>
          <Chart
            data={[{
              x: data.comparisons.map(c => `t=${c.time}d`),
              y: data.comparisons.map(c => c.rmse),
              type: 'bar', marker: { color: data.comparisons.map((_, i) => COLORS[i % COLORS.length]) },
            }]}
            layout={{
              xaxis: { title: 'Timestep' }, yaxis: { title: 'RMSE (m)' }, height: 350,
            }}
          />
        </Card>
      </div>

      {/* Relative error heatmap */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Relative Error (%) by Time & Depth</h3>
        <Chart
          data={[{
            z: data.comparisons.map(c => c.abs_error.map((e, i) => Math.abs(c.psi_hydrus[i]) > 1 ? (e / Math.abs(c.psi_hydrus[i])) * 100 : 0)),
            x: data.comparisons[0]?.z || [],
            y: data.comparisons.map(c => `Hour ${c.time}`),
            type: 'heatmap', colorscale: 'YlOrRd',
            colorbar: { title: { text: '%', side: 'right' }, thickness: 15 },
          }]}
          layout={{
            xaxis: { title: 'Depth (m)' }, yaxis: { title: '' }, height: 350,
          }}
        />
      </Card>

      {/* Detailed stats table */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">📋 Detailed Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="p-2">Time</th><th className="p-2">R²</th><th className="p-2">RMSE</th>
                <th className="p-2">MAE</th><th className="p-2">Max Err</th><th className="p-2">Quality</th>
              </tr>
            </thead>
            <tbody>
              {data.comparisons.map(c => {
                const mx = Math.max(...c.abs_error);
                const q = c.r2 > 0.99 ? '🟢 Excellent' : c.r2 > 0.95 ? '🟡 Good' : '🔴 Needs Work';
                return (
                  <tr key={c.time} className="border-b border-slate-50">
                    <td className="p-2 font-medium">Hour {c.time}</td>
                    <td className="p-2">{c.r2.toFixed(4)}</td>
                    <td className="p-2">{c.rmse.toFixed(3)}</td>
                    <td className="p-2">{c.mae.toFixed(3)}</td>
                    <td className="p-2">{mx.toFixed(3)}</td>
                    <td className="p-2">{q}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
