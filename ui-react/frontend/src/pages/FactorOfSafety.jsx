import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function FactorOfSafety() {
  const { geo, defaults } = useApp();
  const [zRes, setZRes] = useState(50);
  const [tRes, setTRes] = useState(50);
  const [zMax, setZMax] = useState(40);
  const [fsClip, setFsClip] = useState(5);
  const [data, setData] = useState(null);
  const [history, setHistory] = useState(null);
  const [selDepths, setSelDepths] = useState([2, 5, 10, 20]);

  useEffect(() => {
    api.fsGrid({ z_min: 0.5, z_max: zMax, z_res: zRes, t_res: tRes, geo }).then(setData);
  }, [geo, zRes, tRes, zMax]);

  useEffect(() => {
    if (!selDepths.length) return;
    Promise.all(selDepths.map(d => {
      const times = Array.from({ length: 200 }, (_, i) => i * (defaults.norm.t_max / 199));
      return api.factorOfSafety({
        z: Array(200).fill(d), t: times, geo
      }).then(r => ({ depth: d, times, fs: r.fs }));
    })).then(setHistory);
  }, [selDepths, geo, defaults]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Factor of Safety Contour Map" subtitle="2-D heatmap of slope stability across depth and time" icon="🗺️" />

      {/* Controls */}
      <Card className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          ['Depth res', zRes, setZRes, 20, 100, 10],
          ['Time res', tRes, setTRes, 20, 100, 10],
          ['Max depth (m)', zMax, setZMax, 5, 55, 5],
          ['FS clip', fsClip, setFsClip, 1.5, 10, 0.5],
        ].map(([label, val, setter, min, max, step]) => (
          <div key={label} className="space-y-1">
            <label className="text-xs font-medium text-slate-500">{label}: <strong>{val}</strong></label>
            <input type="range" min={min} max={max} step={step} value={val}
              onChange={e => setter(+e.target.value)} className="w-full accent-blue-600" />
          </div>
        ))}
      </Card>

      {/* Heatmap */}
      {!data ? <Spinner text="Computing FS grid…" /> : (
        <>
          <Card>
            <Chart
              data={[{
                x: data.t, y: data.z, z: data.fs, type: 'heatmap',
                colorscale: [[0, '#7f1d1d'], [0.15, '#ef4444'], [0.2, '#f97316'],
                  [0.3, '#eab308'], [0.5, '#22c55e'], [1, '#0ea5e9']],
                colorbar: { title: { text: 'FS', side: 'right' } },
                hovertemplate: 'Time: %{x:.1f}d<br>Depth: %{y:.1f}m<br>FS: %{z:.3f}<extra></extra>',
                zmin: 0, zmax: fsClip,
              }]}
              layout={{
                xaxis: { title: 'Time (days)' },
                yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                height: 520,
              }}
            />
          </Card>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard label="Total Cells" value={data.stats.total.toLocaleString()} color="blue" />
            <MetricCard label="🔴 Unstable (FS<1)" value={`${data.stats.unstable} (${(100 * data.stats.unstable / data.stats.total).toFixed(1)}%)`} color="red" />
            <MetricCard label="🟡 Marginal" value={`${data.stats.marginal} (${(100 * data.stats.marginal / data.stats.total).toFixed(1)}%)`} color="amber" />
            <MetricCard label="🟢 Safe (FS≥1.5)" value={`${data.stats.safe} (${(100 * data.stats.safe / data.stats.total).toFixed(1)}%)`} color="green" />
          </div>
        </>
      )}

      {/* Time history */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">📉 FS Time History at Specific Depths</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {[1, 2, 5, 10, 15, 20, 30, 40].map(d => (
            <button key={d} onClick={() => setSelDepths(prev => prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d])}
              className={`px-3 py-1 text-xs rounded-full font-medium transition
                ${selDepths.includes(d) ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
              {d}m
            </button>
          ))}
        </div>
        {!history ? <Spinner /> : (
          <Chart
            data={[
              ...history.map((h, i) => ({
                x: h.times, y: h.fs, type: 'scatter', mode: 'lines',
                name: `z=${h.depth}m`, line: { color: COLORS[i % COLORS.length], width: 2.5 },
              })),
              { x: [0, defaults.norm.t_max], y: [1, 1], type: 'scatter', mode: 'lines',
                line: { color: '#dc2626', dash: 'dash' }, name: 'FS=1', showlegend: true },
            ]}
            layout={{
              xaxis: { title: 'Time (days)' }, yaxis: { title: 'Factor of Safety', range: [0, Math.min(fsClip, 8)] },
              height: 420, legend: { orientation: 'h', y: 1.08 },
            }}
          />
        )}
      </Card>
    </div>
  );
}
