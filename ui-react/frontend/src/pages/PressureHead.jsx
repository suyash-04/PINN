import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function PressureHead() {
  const { defaults } = useApp();
  const [selectedTimes, setSelectedTimes] = useState([0, 30, 60, 90, 96, 123]);
  const [depth, setDepth] = useState(5);
  const [data, setData] = useState(null);
  const [timeData, setTimeData] = useState(null);

  useEffect(() => {
    if (!selectedTimes.length) return;
    const depths = Array.from({ length: 200 }, (_, i) => 0.5 + i * (50 / 200));
    Promise.all(selectedTimes.map(t =>
      api.predict({ z: depths, t: Array(200).fill(t) })
    )).then(profiles => setData({ profiles, depths }));
  }, [selectedTimes]);

  useEffect(() => {
    const times = Array.from({ length: 200 }, (_, i) => i * (defaults.norm.t_max / 199));
    api.predict({ z: Array(200).fill(depth), t: times }).then(d =>
      setTimeData({ times, psi: d.psi })
    );
  }, [depth, defaults]);

  const toggle = (t) => setSelectedTimes(prev =>
    prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Pressure Head Profiles — ψ(z, t)" subtitle="Pore-water pressure vs depth at selected timesteps" icon="📈" />

      {/* Time selector */}
      <Card>
        <p className="text-xs font-semibold text-slate-500 mb-2">SELECT TIMESTEPS</p>
        <div className="flex flex-wrap gap-2">
          {defaults.available_times.map(t => (
            <button key={t} onClick={() => toggle(t)}
              className={`px-3 py-1 text-xs rounded-full font-medium transition
                ${selectedTimes.includes(t) ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
              Day {t}
            </button>
          ))}
        </div>
      </Card>

      {/* Profile plot */}
      {!data ? <Spinner /> : (
        <Card>
          <Chart
            data={data.profiles.map((p, i) => ({
              x: p.psi, y: data.depths, type: 'scatter', mode: 'lines',
              name: `Day ${selectedTimes[i]}`, line: { color: COLORS[i % COLORS.length], width: 2.5 },
            }))}
            layout={{
              xaxis: { title: 'Pressure Head ψ (m)' },
              yaxis: { title: 'Depth z (m)', autorange: 'reversed' },
              height: 550, legend: { orientation: 'h', y: 1.08, xanchor: 'center', x: 0.5 },
            }}
          />
        </Card>
      )}

      {/* ψ vs Time */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">ψ vs Time at Fixed Depth</h3>
        <input type="range" min={0.5} max={50} step={0.5} value={depth}
          onChange={e => setDepth(+e.target.value)}
          className="w-full accent-blue-600 mb-1" />
        <p className="text-xs text-slate-500 mb-3">Depth: <strong>{depth} m</strong></p>
        {!timeData ? <Spinner /> : (
          <Chart
            data={[{
              x: timeData.times, y: timeData.psi, type: 'scatter', mode: 'lines',
              line: { color: '#2563eb', width: 2.5 }, fill: 'tozeroy',
              fillcolor: 'rgba(37,99,235,0.06)', name: `ψ at z=${depth}m`,
            }]}
            layout={{
              xaxis: { title: 'Time (days)' }, yaxis: { title: 'ψ (m)' }, height: 380,
            }}
          />
        )}
      </Card>
    </div>
  );
}
