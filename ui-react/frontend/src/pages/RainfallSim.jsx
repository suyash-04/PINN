// RainfallSim.jsx
import { useState } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Card, SliderField, MetricCard, SectionTitle, Button } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function RainfallSim() {
  const { geo, norm } = useApp();
  const [intensity, setIntensity] = useState(1e-6);
  const [duration,  setDuration]  = useState(48);
  const [results,   setResults]   = useState(null);
  const [loading,   setLoading]   = useState(false);

  const INTENSITIES = [
    [1e-7, '1×10⁻⁷  light drizzle'],
    [5e-7, '5×10⁻⁷  moderate rain'],
    [1e-6, '1×10⁻⁶  steady rain'],
    [5e-6, '5×10⁻⁶  heavy rain'],
    [1e-5, '1×10⁻⁵  extreme event'],
  ];

  const simulate = async () => {
    setLoading(true);
    const tMax  = norm.t_max;
    const times = [0, 20, 40, 60, 80, 100, tMax];
    const depths = Array.from({ length: 40 }, (_, i) => 0.5 + (i / 39) * 39.5);

    const profiles = await Promise.all(times.map(async t => {
      const predRes = await api.predict({ z: depths, t: Array(depths.length).fill(t) });
      const psiArr  = predRes.psi;
      const infiltration = Math.min(intensity * duration * 3600, geo.Ks * duration * 3600);
      const psiWet = psiArr.map((psi, i) => {
        const depthFactor = Math.exp(-depths[i] / 5);
        const timeFactor  = t < duration
          ? t / duration
          : Math.max(0, 1 - (t - duration) / (tMax - duration));
        return psi + infiltration * 1000 * depthFactor * timeFactor;
      });
      return { time: t, depths, psi_dry: psiArr, psi_wet: psiWet };
    }));

    setResults({ profiles, intensity, duration });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1200 }}>
      <PageHeader
        title="Rainfall Scenario Simulation"
        subtitle="Conceptual overlay — how different rainfall intensities perturb the baseline ψ field"
        badge="Conceptual"
      />

      <Card>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 20, alignItems: 'end' }}>
          <div>
            <SectionTitle>Rainfall intensity</SectionTitle>
            {INTENSITIES.map(([v, label]) => (
              <label key={v} style={{ display: 'flex', alignItems: 'center', gap: 8,
                fontSize: 11, color: intensity === v ? 'var(--accent)' : 'var(--muted)',
                cursor: 'pointer', marginBottom: 5 }}>
                <input type="radio" name="intensity" value={v}
                  checked={intensity === v} onChange={() => setIntensity(v)}
                  style={{ accentColor: 'var(--accent)' }} />
                <span style={{ fontFamily: 'var(--font-mono)' }}>{label}</span>
              </label>
            ))}
          </div>
          <SliderField label="Duration" value={duration} onChange={setDuration}
            min={1} max={120} step={1} fmt={v => `${v} hours`} />
          <Button onClick={simulate} disabled={loading} variant="primary">
            {loading ? 'Simulating…' : 'Simulate →'}
          </Button>
        </div>
      </Card>

      {!results
        ? (
          <Card style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--muted)' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>🌧</div>
            <div style={{ fontSize: 12 }}>Configure parameters and click <strong>Simulate</strong></div>
          </Card>
        )
        : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
              <MetricCard label="Intensity"
                value={results.intensity.toExponential(1)+' m/s'} color="blue" />
              <MetricCard label="Duration"
                value={`${results.duration} hours`} color="amber" />
              <MetricCard label="Cumulative"
                value={`${(results.intensity * results.duration * 3600 * 1000).toFixed(1)} mm`}
                color="green" />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Card>
                <SectionTitle>ψ profiles — dry vs wet (even timesteps)</SectionTitle>
                <Chart
                  data={results.profiles.filter((_, i) => i % 2 === 0).flatMap((p, i) => [
                    { x: p.psi_dry, y: p.depths, type: 'scatter', mode: 'lines',
                      name: `Dry t=${p.time}`, legendgroup: `t${p.time}`,
                      line: { color: COLORS[i % COLORS.length], width: 1.2, dash: 'dot' } },
                    { x: p.psi_wet, y: p.depths, type: 'scatter', mode: 'lines',
                      name: `Wet t=${p.time}`, legendgroup: `t${p.time}`,
                      line: { color: COLORS[i % COLORS.length], width: 2 } },
                  ])}
                  layout={{
                    xaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
                    yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                    legend: { font: { size: 9 } },
                  }}
                  height={400}
                />
              </Card>

              <Card>
                <SectionTitle>Δψ heatmap — wet − dry (m)</SectionTitle>
                <Chart
                  data={[{
                    z: results.profiles.map(p => p.psi_wet.map((w, i) => w - p.psi_dry[i])),
                    x: results.profiles[0].depths,
                    y: results.profiles.map(p => `Day ${p.time}`),
                    type: 'heatmap', colorscale: 'RdBu',
                    colorbar: { title: { text: 'Δψ (m)', side: 'right' }, thickness: 14,
                      tickfont: { size: 10, color: '#7d8590' } },
                  }]}
                  layout={{
                    xaxis: { title: { text: 'Depth (m)', font: { size: 11 } } },
                    margin: { l: 60, r: 80, t: 20, b: 50 },
                  }}
                  height={400}
                />
              </Card>
            </div>

            <Card>
              <SectionTitle>Note on methodology</SectionTitle>
              <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.7 }}>
                This is a conceptual overlay. The base ψ field is from the trained PINN.
                A simplified depth-attenuated, time-ramped infiltration perturbation is applied
                (exp decay with depth, linear with time) — not a coupled re-simulation.
                For rigorous rainfall-coupled analysis, retrain with time-varying flux boundary
                conditions.
              </div>
            </Card>
          </>
        )
      }
    </div>
  );
}
