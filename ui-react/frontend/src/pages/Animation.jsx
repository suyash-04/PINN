import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, SectionTitle, SliderField } from '../components/ui';
import Chart, { DANGER } from '../components/Chart';

export default function Animation() {
  const { geo, norm, defaults } = useApp();
  const times = defaults.available_times;
  const zMax  = norm.z_max; // fixed: was incorrectly reading geo.z_max

  const [frame,   setFrame]   = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed,   setSpeed]   = useState(600);
  const [mode,    setMode]    = useState('psi'); // 'psi' | 'fs'
  const [cache,   setCache]   = useState({});
  const [loadedCount, setLoadedCount] = useState(0);
  const timerRef = useRef(null);

  // Sequential pre-fetch (avoids parallel flood)
  useEffect(() => {
    setCache({});
    setLoadedCount(0);
    let cancelled = false;
    const depths = Array.from({ length: 60 }, (_, i) => 0.5 + (i / 59) * (zMax - 0.5));

    (async () => {
      for (const t of times) {
        if (cancelled) break;
        const [predRes, fsRes] = await Promise.all([
          api.predict({ z: depths, t: Array(depths.length).fill(t) }),
          api.factorOfSafety({ z: depths, t: Array(depths.length).fill(t), geo }),
        ]);
        if (cancelled) break;
        setCache(prev => ({ ...prev, [t]: { depths, psi: predRes.psi, fs: fsRes.fs } }));
        setLoadedCount(prev => prev + 1);
      }
    })();

    return () => { cancelled = true; };
  }, [geo, norm]); // re-fetch when geo changes

  const pause = useCallback(() => {
    setPlaying(false);
    clearInterval(timerRef.current);
  }, []);

  const play = useCallback(() => {
    setPlaying(true);
    timerRef.current = setInterval(() => {
      setFrame(f => {
        if (f >= times.length - 1) {
          clearInterval(timerRef.current);
          setPlaying(false);
          return 0;
        }
        return f + 1;
      });
    }, speed);
  }, [speed, times.length]);

  useEffect(() => () => clearInterval(timerRef.current), []);

  const ready = loadedCount >= times.length;
  const t = times[frame];
  const d = cache[t];

  const psiMin = d ? Math.min(...d.psi) * 1.05 : -500;
  const psiMax = d ? Math.max(...d.psi) * 0.5  : -40;
  const fsMax  = d ? Math.min(8, Math.max(...d.fs) * 1.1) : 6;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1000 }}>
      <PageHeader
        title="Time-Lapse Animation"
        subtitle="Animated evolution of ψ and FS through the simulation period"
        badge="Animation"
      />

      {/* Loading progress */}
      {!ready && (
        <Card>
          <div style={{ marginBottom: 8, fontSize: 11, color: 'var(--muted)' }}>
            Pre-fetching frames — {loadedCount} / {times.length}
          </div>
          <div style={{ height: 4, background: 'var(--border)', borderRadius: 2 }}>
            <div style={{
              height: '100%', borderRadius: 2, background: 'var(--accent)',
              width: `${(loadedCount / times.length) * 100}%`,
              transition: 'width .3s',
            }} />
          </div>
        </Card>
      )}

      {/* Controls */}
      <Card>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
          <button
            onClick={playing ? pause : play}
            disabled={loadedCount === 0}
            style={{ padding: '6px 18px', fontSize: 12, fontWeight: 600, borderRadius: 6,
              background: playing ? 'var(--border2)' : 'var(--accent)', color: '#fff',
              border: 'none', cursor: loadedCount === 0 ? 'not-allowed' : 'pointer',
              opacity: loadedCount === 0 ? 0.4 : 1 }}>
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={() => { pause(); setFrame(0); }}
            style={{ padding: '6px 14px', fontSize: 12, borderRadius: 6,
              background: 'var(--border)', color: 'var(--muted)', border: 'none', cursor: 'pointer' }}>
            ⏮ Reset
          </button>

          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: 4, marginLeft: 8 }}>
            {[['psi', 'ψ Profile'], ['fs', 'FS Profile']].map(([m, lbl]) => (
              <button key={m} onClick={() => setMode(m)}
                style={{ padding: '5px 12px', fontSize: 11, fontWeight: 600, borderRadius: 5,
                  background: mode === m ? 'rgba(47,129,247,.2)' : 'transparent',
                  color: mode === m ? 'var(--accent)' : 'var(--muted)',
                  border: `1px solid ${mode === m ? 'var(--accent)' : 'var(--border)'}`,
                  cursor: 'pointer' }}>
                {lbl}
              </button>
            ))}
          </div>

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 10, color: 'var(--muted)' }}>Frame delay</span>
            <input type="range" min={150} max={1500} step={50} value={speed}
              onChange={e => setSpeed(+e.target.value)}
              style={{ width: 80, accentColor: 'var(--accent)' }} />
            <span style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'var(--font-mono)',
              width: 42 }}>{speed} ms</span>
          </div>
        </div>

        {/* Scrubber */}
        <div>
          <input type="range" min={0} max={times.length - 1} value={frame}
            onChange={e => { pause(); setFrame(+e.target.value); }}
            style={{ width: '100%', accentColor: 'var(--accent)' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between',
            fontSize: 9, color: 'var(--muted)', fontFamily: 'var(--font-mono)', marginTop: 3, padding: '0 2px' }}>
            {times.filter((_, i) => i % Math.ceil(times.length / 8) === 0).map(t => (
              <span key={t}>Day {t}</span>
            ))}
          </div>
        </div>
      </Card>

      {/* Frame status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ padding: '4px 14px', background: 'rgba(47,129,247,.15)',
          border: '1px solid rgba(47,129,247,.3)', borderRadius: 4,
          fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)' }}>
          Day {t}
        </div>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          Frame {frame + 1} / {times.length}
        </span>
        {d && <>
          <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
            ψ ∈ [{Math.min(...d.psi).toFixed(1)}, {Math.max(...d.psi).toFixed(1)}] m
          </span>
          <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
            FS_min = {Math.min(...d.fs).toFixed(3)}
          </span>
        </>}
      </div>

      {/* Chart */}
      <Card>
        {!d
          ? <Spinner text={ready ? 'Rendering…' : `Loading ${loadedCount}/${times.length} frames…`} />
          : mode === 'psi'
            ? (
              <Chart
                data={[{
                  x: d.psi, y: d.depths, type: 'scatter', mode: 'lines',
                  name: `ψ  Day ${t}`,
                  line: { color: '#2f81f7', width: 2.5 },
                  fill: 'tozerox', fillcolor: 'rgba(47,129,247,.07)',
                }]}
                layout={{
                  xaxis: { title: { text: 'ψ (m)', font: { size: 11 } }, range: [psiMin, psiMax] },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                  title: { text: `Pressure Head — Day ${t}`,
                    font: { size: 12, color: '#7d8590' }, x: 0.5 },
                }}
                height={500}
              />
            )
            : (
              <Chart
                data={[
                  { x: d.fs, y: d.depths, type: 'scatter', mode: 'lines',
                    name: `FS  Day ${t}`, line: { color: '#3fb950', width: 2.5 } },
                  { x: [1, 1], y: [0, zMax], type: 'scatter', mode: 'lines',
                    name: 'FS = 1', line: { color: DANGER, width: 1.5, dash: 'dash' } },
                ]}
                layout={{
                  xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } }, range: [0, fsMax] },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                  title: { text: `Factor of Safety — Day ${t}`,
                    font: { size: 12, color: '#7d8590' }, x: 0.5 },
                }}
                height={500}
              />
            )
        }
      </Card>
    </div>
  );
}
