import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card } from '../components/ui';
import Chart from '../components/Chart';

export default function Animation() {
  const { geo, norm, defaults } = useApp();
  const times = defaults.available_times;
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [mode, setMode] = useState('psi');
  const [cache, setCache] = useState({});
  const timer = useRef(null);

  // Pre-fetch all frames
  useEffect(() => {
    (async () => {
      const c = {};
      const depths = Array.from({ length: 50 }, (_, i) => 0.5 + (i / 49) * (geo.z_max || norm.z_max));
      for (const t of times) {
        const [predRes, fsRes] = await Promise.all([
          api.predict({ z: depths, t: Array(depths.length).fill(t) }),
          api.factorOfSafety({ z: depths, t: Array(depths.length).fill(t), geo }),
        ]);
        c[t] = { depths, psi: predRes.psi, fs: fsRes.fs };
      }
      setCache(c);
    })();
  }, [geo, norm, times]);

  const play = useCallback(() => {
    setPlaying(true);
    timer.current = setInterval(() => {
      setFrame(f => {
        if (f >= times.length - 1) { setPlaying(false); clearInterval(timer.current); return 0; }
        return f + 1;
      });
    }, speed);
  }, [speed, times]);

  const pause = () => { setPlaying(false); clearInterval(timer.current); };

  useEffect(() => () => clearInterval(timer.current), []);

  const t = times[frame];
  const d = cache[t];
  const zMax = geo.z_max || norm.z_max;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Time-Lapse Animation" subtitle="Animated evolution of pressure head and factor of safety over simulation time" icon="🎬" />

      {Object.keys(cache).length < times.length ? (
        <Card className="text-center py-12">
          <Spinner />
          <p className="text-xs text-slate-500 mt-2">Loading frames ({Object.keys(cache).length}/{times.length})…</p>
        </Card>
      ) : (
        <>
          {/* Controls */}
          <Card>
            <div className="flex items-center gap-4 flex-wrap">
              <button onClick={playing ? pause : play}
                className="px-5 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition">
                {playing ? '⏸ Pause' : '▶ Play'}
              </button>
              <button onClick={() => setFrame(0)} className="px-3 py-2 text-xs rounded bg-slate-100 hover:bg-slate-200 text-slate-600">
                ⏮ Reset
              </button>
              <div className="flex gap-2">
                {['psi', 'fs'].map(m => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-3 py-1.5 text-xs rounded-lg font-medium ${mode === m ? 'bg-slate-800 text-white' : 'bg-slate-100 text-slate-600'}`}>
                    {m === 'psi' ? 'ψ Profile' : 'FS Profile'}
                  </button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-2">
                <span className="text-xs text-slate-500">Speed</span>
                <input type="range" min={100} max={1500} step={100} value={speed}
                  onChange={e => setSpeed(+e.target.value)} className="w-24" />
                <span className="text-xs text-slate-500 w-12">{speed}ms</span>
              </div>
            </div>

            {/* Timeline scrubber */}
            <div className="mt-4">
              <input type="range" min={0} max={times.length - 1} value={frame}
                onChange={e => { setFrame(+e.target.value); pause(); }}
                className="w-full accent-blue-600" />
              <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-1">
                {times.filter((_, i) => i % Math.ceil(times.length / 10) === 0 || i === times.length - 1).map(t => (
                  <span key={t}>{t}d</span>
                ))}
              </div>
            </div>
          </Card>

          {/* Current frame info */}
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-bold">Day {t}</div>
            <div className="text-xs text-slate-500">Frame {frame + 1} / {times.length}</div>
            {d && (
              <>
                <div className="text-xs text-slate-500">ψ range: [{Math.min(...d.psi).toFixed(1)}, {Math.max(...d.psi).toFixed(1)}] m</div>
                <div className="text-xs text-slate-500">FS min: {Math.min(...d.fs).toFixed(3)}</div>
              </>
            )}
          </div>

          {/* Chart */}
          {d && (
            <Card>
              {mode === 'psi' ? (
                <Chart
                  data={[{
                    x: d.psi, y: d.depths, type: 'scatter', mode: 'lines',
                    line: { color: '#2563eb', width: 3 }, fill: 'tozerox', fillcolor: 'rgba(37,99,235,0.08)',
                    name: `ψ at t=${t}d`,
                  }]}
                  layout={{
                    xaxis: { title: 'ψ (m)', range: [norm.psi_min * 1.05, norm.psi_max * 0.5] },
                    yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                    height: 500, title: { text: `Pressure Head Profile — Day ${t}`, font: { size: 14 } },
                  }}
                />
              ) : (
                <Chart
                  data={[{
                    x: d.fs, y: d.depths, type: 'scatter', mode: 'lines',
                    line: { color: '#059669', width: 3 }, name: `FS at t=${t}d`,
                  }, {
                    x: [1, 1], y: [0, zMax], type: 'scatter', mode: 'lines',
                    line: { color: '#dc2626', width: 2, dash: 'dash' }, name: 'FS = 1',
                  }]}
                  layout={{
                    xaxis: { title: 'Factor of Safety', range: [0, Math.max(5, Math.max(...d.fs) * 1.1)] },
                    yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                    height: 500, title: { text: `Factor of Safety — Day ${t}`, font: { size: 14 } },
                  }}
                />
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
