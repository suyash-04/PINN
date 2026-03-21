import { useState } from 'react';
import { useApp } from '../context';
import { api } from '../api';
import { PageHeader, Card, Spinner, MetricCard, SliderField, SectionTitle, Button, StatusBadge } from '../components/ui';
import Chart, { COLORS, DANGER } from '../components/Chart';

const PRESETS = {
  'Clay (stiff)':   { c_prime: 15, phi_prime: 25, gamma: 19,   alpha: 0.5, n: 1.3, Ks: 1e-8 },
  'Silt (soft)':    { c_prime: 5,  phi_prime: 28, gamma: 18,   alpha: 2.0, n: 1.5, Ks: 1e-6 },
  'Sand (loose)':   { c_prime: 0,  phi_prime: 33, gamma: 17,   alpha: 5.0, n: 2.5, Ks: 1e-4 },
  'Residual soil':  { c_prime: 8,  phi_prime: 30, gamma: 18.5, alpha: 1.0, n: 1.8, Ks: 5e-7 },
};

export default function ScenarioComparator() {
  // Use useApp — do NOT reference `defaults` as a bare variable
  const { geo, norm, defaults } = useApp();
  const tMax = defaults.norm.t_max;
  const zMax = norm.z_max;

  const [scenarios, setScenarios] = useState([
    { name: 'Current', params: { ...geo }, color: COLORS[0] },
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [time, setTime]       = useState(Math.round(tMax * 0.78));

  const addPreset = name => {
    if (scenarios.find(s => s.name === name)) return;
    setScenarios(prev => [...prev, {
      name, params: { ...geo, ...PRESETS[name] },
      color: COLORS[prev.length % COLORS.length],
    }]);
  };

  const remove = idx => setScenarios(prev => prev.filter((_, i) => i !== idx));

  const compare = async () => {
    setLoading(true);
    const depths = Array.from({ length: 60 }, (_, i) => 0.5 + (i / 59) * (zMax - 0.5));
    const res = await Promise.all(scenarios.map(async s => {
      const [predRes, fsRes] = await Promise.all([
        api.predict({ z: depths, t: Array(depths.length).fill(time) }),
        api.factorOfSafety({ z: depths, t: Array(depths.length).fill(time), geo: s.params }),
      ]);
      return { name: s.name, color: s.color, depths,
        psi: predRes.psi, fs: fsRes.fs, min_fs: Math.min(...fsRes.fs) };
    }));
    setResults(res);
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1300 }}>
      <PageHeader
        title="Scenario Comparator"
        subtitle="Side-by-side stability comparison across soil types"
        badge="Scenarios"
      />

      {/* Scenario list */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
          <SectionTitle>Scenarios ({scenarios.length})</SectionTitle>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {Object.keys(PRESETS).map(name => (
              <button key={name} onClick={() => addPreset(name)} style={{
                padding: '3px 12px', fontSize: 11, borderRadius: 4,
                border: '1px solid var(--border)', background: 'transparent',
                color: 'var(--muted)', cursor: 'pointer',
              }}>
                + {name}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {scenarios.map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10,
              padding: '7px 12px', background: 'rgba(255,255,255,.025)',
              borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
                background: s.color }} />
              <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)', flex: 1 }}>
                {s.name}
              </span>
              <span style={{ fontSize: 10, color: 'var(--muted)',
                fontFamily: 'var(--font-mono)' }}>
                c′={s.params.c_prime} kPa  φ′={s.params.phi_prime}°  Ks={s.params.Ks?.toExponential(1)}
              </span>
              {i > 0 && (
                <button onClick={() => remove(i)} style={{ background: 'none', border: 'none',
                  cursor: 'pointer', color: 'var(--danger)', fontSize: 14, padding: 0 }}>×</button>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Time + run */}
      <Card style={{ maxWidth: 600 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 20, alignItems: 'end' }}>
          <SliderField label="Analysis time" value={time} onChange={setTime}
            min={0} max={tMax} step={1} fmt={v => `Day ${v.toFixed(0)}`} />
          <Button onClick={compare} disabled={loading || scenarios.length < 2} variant="primary">
            {loading ? 'Computing…' : 'Compare →'}
          </Button>
        </div>
        {scenarios.length < 2 && (
          <div style={{ fontSize: 10, color: 'var(--warn)', marginTop: 8 }}>
            Add at least one more scenario to compare.
          </div>
        )}
      </Card>

      {loading && <Spinner text="Running scenarios…" />}

      {results && !loading && (
        <>
          {/* Min FS summary */}
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${results.length}, 1fr)`, gap: 12 }}>
            {results.map(r => (
              <div key={r.name} style={{ padding: '12px 16px',
                borderTop: `2px solid ${r.color}`,
                background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8 }}>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}>{r.name}</div>
                <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: 'var(--text)' }}>{r.min_fs.toFixed(3)}</div>
                <div style={{ marginTop: 4 }}><StatusBadge fs={r.min_fs} /></div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card>
              <SectionTitle>Factor of Safety comparison</SectionTitle>
              <Chart
                data={[
                  ...results.map(r => ({
                    x: r.fs, y: r.depths, type: 'scatter', mode: 'lines',
                    name: r.name, line: { color: r.color, width: 2 },
                  })),
                  { x: [1, 1], y: [0, zMax], type: 'scatter', mode: 'lines',
                    name: 'FS = 1', line: { color: DANGER, dash: 'dash', width: 1.5 } },
                ]}
                layout={{
                  xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } } },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                  legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
                }}
                height={400}
              />
            </Card>

            <Card>
              <SectionTitle>Ranking — worst to best FS</SectionTitle>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 1, marginTop: 8 }}>
                {[...results].sort((a, b) => a.min_fs - b.min_fs).map((r, i) => {
                  const s = scenarios.find(x => x.name === r.name);
                  return (
                    <div key={r.name} style={{ display: 'grid',
                      gridTemplateColumns: '20px 12px 1fr auto auto',
                      gap: 10, padding: '8px 0', borderBottom: '1px solid var(--border)',
                      alignItems: 'center', fontSize: 11 }}>
                      <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>#{i+1}</span>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: r.color }} />
                      <span style={{ color: 'var(--text)', fontWeight: 500 }}>{r.name}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>
                        {r.min_fs.toFixed(4)}
                      </span>
                      <StatusBadge fs={r.min_fs} />
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
