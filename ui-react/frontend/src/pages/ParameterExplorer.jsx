import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function ParameterExplorer() {
  const { geo, defaults } = useApp();
  const [time, setTime] = useState(96);
  const [zMax, setZMax] = useState(40);
  const [profiles, setProfiles] = useState(null);
  const [sensitivity, setSensitivity] = useState(null);
  const [queryResult, setQueryResult] = useState(null);

  useEffect(() => {
    const depths = Array.from({ length: 200 }, (_, i) => 0.5 + i * ((zMax - 0.5) / 199));
    const defGeo = { ...defaults.geo };
    Promise.all([
      api.factorOfSafety({ z: depths, t: Array(200).fill(time), geo }),
      api.factorOfSafety({ z: depths, t: Array(200).fill(time), geo: defGeo }),
      api.predict({ z: depths, t: Array(200).fill(time) }),
    ]).then(([fsCur, fsDef, psiData]) => {
      setProfiles({ depths, fsCurrent: fsCur.fs, fsDefault: fsDef.fs, psi: psiData.psi });
    });
    api.sensitivity({ z: 10, t: time, geo }).then(setSensitivity);
  }, [geo, time, zMax, defaults]);

  const query = (z, t) => {
    api.factorOfSafety({ z: [z], t: [t], geo }).then(r => {
      api.predict({ z: [z], t: [t] }).then(p => {
        setQueryResult({ z, t, psi: p.psi[0], fs: r.fs[0] });
      });
    });
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Interactive Parameter Explorer" subtitle="Adjust sidebar parameters to observe real-time effects on stability" icon="🎛️" />

      <Card className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-500">Time: <strong>Day {time}</strong></label>
          <input type="range" min={0} max={defaults.norm.t_max} step={1} value={time}
            onChange={e => setTime(+e.target.value)} className="w-full accent-blue-600" />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-slate-500">Max depth: <strong>{zMax} m</strong></label>
          <input type="range" min={5} max={55} step={5} value={zMax}
            onChange={e => setZMax(+e.target.value)} className="w-full accent-blue-600" />
        </div>
      </Card>

      {/* Dual profile plot */}
      {!profiles ? <Spinner /> : (
        <Card>
          <div className="grid lg:grid-cols-2 gap-4">
            <Chart
              data={[{ x: profiles.psi, y: profiles.depths, type: 'scatter', mode: 'lines',
                name: 'ψ (PINN)', line: { color: '#2563eb', width: 2.5 } }]}
              layout={{ xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                height: 450, title: { text: 'Pressure Head ψ(z)', font: { size: 14 } } }}
            />
            <Chart
              data={[
                { x: profiles.fsCurrent, y: profiles.depths, type: 'scatter', mode: 'lines',
                  name: 'Current params', line: { color: '#16a34a', width: 2.5 } },
                { x: profiles.fsDefault, y: profiles.depths, type: 'scatter', mode: 'lines',
                  name: 'Default params', line: { color: '#94a3b8', width: 2, dash: 'dash' } },
                { x: [1, 1], y: [0, zMax], mode: 'lines', name: 'FS=1',
                  line: { color: '#dc2626', dash: 'dash', width: 1.5 }, showlegend: true, type: 'scatter' },
              ]}
              layout={{ xaxis: { title: 'Factor of Safety', range: [0, 10] },
                yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                height: 450, title: { text: 'Factor of Safety FS(z)', font: { size: 14 } },
                legend: { orientation: 'h', y: 1.1 } }}
            />
          </div>
        </Card>
      )}

      {/* Sensitivity */}
      {sensitivity && (
        <Card>
          <h3 className="text-sm font-semibold text-slate-700 mb-2">📊 Sensitivity (FS at 10m depth)</h3>
          <p className="text-xs text-slate-500 mb-3">Baseline FS = {sensitivity.fs_base.toFixed(3)}</p>
          <Chart
            data={[
              { y: sensitivity.sensitivity.map(s => s.param), x: sensitivity.sensitivity.map(s => s.fs_low),
                type: 'bar', orientation: 'h', name: '↓ param', marker: { color: '#dc2626' } },
              { y: sensitivity.sensitivity.map(s => s.param), x: sensitivity.sensitivity.map(s => s.fs_high),
                type: 'bar', orientation: 'h', name: '↑ param', marker: { color: '#16a34a' } },
            ]}
            layout={{ barmode: 'relative', xaxis: { title: 'ΔFS' }, height: 300,
              margin: { l: 130 }, shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: 5.5,
                line: { color: '#64748b', width: 1 } }] }}
          />
        </Card>
      )}

      {/* Point query */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">🔢 Point Query</h3>
        <div className="flex gap-3 items-end mb-3">
          <div>
            <label className="text-xs text-slate-500">Depth (m)</label>
            <input type="number" defaultValue={5} min={0.5} max={55} step={0.5} id="qz"
              className="block w-24 border rounded-lg px-2 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-slate-500">Time (days)</label>
            <input type="number" defaultValue={96} min={0} max={123} step={1} id="qt"
              className="block w-24 border rounded-lg px-2 py-1.5 text-sm" />
          </div>
          <button onClick={() => query(+document.getElementById('qz').value, +document.getElementById('qt').value)}
            className="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-blue-700 transition">
            Query
          </button>
        </div>
        {queryResult && (
          <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
            <MetricCard label="ψ (m)" value={queryResult.psi.toFixed(2)} color="blue" />
            <MetricCard label="FS" value={queryResult.fs.toFixed(3)} color={queryResult.fs >= 1.5 ? 'green' : queryResult.fs >= 1 ? 'amber' : 'red'} />
            <MetricCard label="Status" value={queryResult.fs >= 1.5 ? '🟢 Safe' : queryResult.fs >= 1 ? '🟡 Marginal' : '🔴 Unstable'}
              color={queryResult.fs >= 1.5 ? 'green' : queryResult.fs >= 1 ? 'amber' : 'red'} />
          </div>
        )}
      </Card>
    </div>
  );
}
