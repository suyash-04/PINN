import { useState } from 'react';
import { useApp } from '../context';
import { api } from '../api';
import { PageHeader, Card, Spinner, MetricCard, SliderField } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

const PRESETS = {
  'Clay (stiff)': { c_prime: 15, phi_prime: 25, gamma: 19, alpha: 0.5, n: 1.3, Ks: 1e-8 },
  'Silt (soft)': { c_prime: 5, phi_prime: 28, gamma: 18, alpha: 2.0, n: 1.5, Ks: 1e-6 },
  'Sand (loose)': { c_prime: 0, phi_prime: 33, gamma: 17, alpha: 5.0, n: 2.5, Ks: 1e-4 },
  'Residual soil': { c_prime: 8, phi_prime: 30, gamma: 18.5, alpha: 1.0, n: 1.8, Ks: 5e-7 },
};

export default function ScenarioComparator() {
  const { geo, norm } = useApp();
  const [scenarios, setScenarios] = useState([
    { name: 'Current', params: { ...geo }, color: COLORS[0] },
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [time, setTime] = useState(96);

  const addPreset = (name) => {
    const preset = PRESETS[name];
    setScenarios(prev => [...prev, {
      name, params: { ...geo, ...preset }, color: COLORS[prev.length % COLORS.length],
    }]);
  };

  const removeScenario = idx => setScenarios(prev => prev.filter((_, i) => i !== idx));

  const compare = async () => {
    setLoading(true);
    const depths = Array.from({ length: 50 }, (_, i) => 0.5 + (i / 49) * 39.5);
    const res = await Promise.all(
      scenarios.map(async s => {
        const [predRes, fsRes] = await Promise.all([
          api.predict({ z: depths, t: Array(depths.length).fill(time) }),
          api.factorOfSafety({ z: depths, t: Array(depths.length).fill(time), geo: s.params }),
        ]);
        return {
          name: s.name, color: s.color, depths,
          psi: predRes.psi,
          fs: fsRes.fs,
          min_fs: Math.min(...fsRes.fs),
        };
      })
    );
    setResults(res);
    setLoading(false);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Scenario Comparator" subtitle="Compare slope stability across different soil and geometry configurations" icon="⚖️" />

      {/* Scenario management */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-700">Scenarios ({scenarios.length})</h3>
          <div className="flex gap-2">
            {Object.keys(PRESETS).map(name => (
              <button key={name} onClick={() => addPreset(name)}
                className="px-3 py-1 text-xs rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 font-medium transition">
                + {name}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          {scenarios.map((s, i) => (
            <div key={i} className="flex items-center gap-3 p-2 bg-slate-50 rounded-lg">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: s.color }} />
              <span className="text-xs font-medium text-slate-700 flex-1">{s.name}</span>
              <span className="text-[10px] text-slate-400 font-mono">
                c'={s.params.c_prime} φ'={s.params.phi_prime}° Ks={s.params.Ks?.toExponential(1)}
              </span>
              {i > 0 && (
                <button onClick={() => removeScenario(i)} className="text-xs text-red-500 hover:text-red-700">✕</button>
              )}
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <SliderField label={`Analysis Time: Day ${time}`} value={time} min={0} max={123} step={1} onChange={setTime} />
          </div>
          <button onClick={compare} disabled={loading || scenarios.length < 2}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition">
            {loading ? 'Computing…' : '⚖️ Compare'}
          </button>
        </div>
      </Card>

      {results && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {results.map(r => (
              <Card key={r.name} className="text-center">
                <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ backgroundColor: r.color }} />
                <p className="text-[10px] text-slate-400">{r.name}</p>
                <p className="text-xl font-bold text-slate-800">{r.min_fs.toFixed(3)}</p>
                <p className="text-[10px] text-slate-400">Min FS</p>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* FS comparison */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">Factor of Safety Comparison</h3>
              <Chart
                data={[
                  ...results.map(r => ({
                    x: r.fs, y: r.depths, type: 'scatter', mode: 'lines',
                    name: r.name, line: { color: r.color, width: 2.5 },
                  })),
                  { x: [1, 1], y: [0, norm.z_max], type: 'scatter', mode: 'lines',
                    name: 'FS=1', line: { color: '#dc2626', dash: 'dash', width: 1.5 } },
                ]}
                layout={{
                  xaxis: { title: 'Factor of Safety' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 420,
                }}
              />
            </Card>

            {/* ψ comparison */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">Pressure Head Comparison</h3>
              <Chart
                data={results.map(r => ({
                  x: r.psi, y: r.depths, type: 'scatter', mode: 'lines',
                  name: r.name, line: { color: r.color, width: 2.5 },
                }))}
                layout={{
                  xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 420,
                }}
              />
            </Card>
          </div>

          {/* Ranking table */}
          <Card>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">📋 Scenario Ranking</h3>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b text-left text-slate-500">
                  <th className="p-2">Rank</th><th className="p-2">Scenario</th><th className="p-2">Min FS</th>
                  <th className="p-2">c'</th><th className="p-2">φ'</th><th className="p-2">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {[...results].sort((a, b) => a.min_fs - b.min_fs).map((r, i) => (
                  <tr key={r.name} className="border-b border-slate-50">
                    <td className="p-2 font-medium">#{i + 1}</td>
                    <td className="p-2 flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color }} />
                      {r.name}
                    </td>
                    <td className="p-2 font-mono">{r.min_fs.toFixed(4)}</td>
                    <td className="p-2">{results.find(x => x.name === r.name) && scenarios.find(s => s.name === r.name)?.params.c_prime}</td>
                    <td className="p-2">{scenarios.find(s => s.name === r.name)?.params.phi_prime}°</td>
                    <td className="p-2">{r.min_fs > 1.5 ? '🟢 Safe' : r.min_fs > 1 ? '🟡 Marginal' : '🔴 Unstable'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}
