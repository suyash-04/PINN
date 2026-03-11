import { useState } from 'react';
import { useApp } from '../context';
import { PageHeader, Card, SliderField, MetricCard } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';
import { api } from '../api';

export default function RainfallSim() {
  const { geo, norm } = useApp();
  const [intensity, setIntensity] = useState(1e-6);
  const [duration, setDuration] = useState(48);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const simulate = async () => {
    setLoading(true);
    const times = [0, 20, 40, 60, 80, 100, 123];
    const depths = Array.from({ length: 40 }, (_, i) => 0.5 + (i / 39) * 39.5);

    const profiles = await Promise.all(
      times.map(async t => {
        const predRes = await api.predict({ z: depths, t: Array(depths.length).fill(t) });
        const psiArr = predRes.psi;
        // Apply a simple rainfall infiltration perturbation (conceptual)
        const infiltration = Math.min(intensity * duration * 3600, geo.Ks * duration * 3600);
        const psiWet = psiArr.map((psi, i) => {
          const depthFactor = Math.exp(-depths[i] / 5);
          const timeFactor = t < duration ? t / duration : Math.max(0, 1 - (t - duration) / (123 - duration));
          return psi + infiltration * 1000 * depthFactor * timeFactor;
        });
        return { time: t, depths, psi_dry: psiArr, psi_wet: psiWet };
      })
    );

    setResults({ profiles, intensity, duration });
    setLoading(false);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Rainfall Scenario Simulation" subtitle="Explore how different rainfall intensities affect pore-water pressure and slope stability" icon="🌧️" />

      <Card>
        <div className="grid grid-cols-3 gap-4 items-end">
          <div>
            <label className="text-xs font-semibold text-slate-500 mb-1 block">Rainfall Intensity (m/s)</label>
            <select value={intensity} onChange={e => setIntensity(+e.target.value)}
              className="w-full p-2 rounded-lg bg-slate-50 border border-slate-200 text-sm">
              <option value={1e-7}>1×10⁻⁷ (light drizzle)</option>
              <option value={5e-7}>5×10⁻⁷ (moderate)</option>
              <option value={1e-6}>1×10⁻⁶ (steady rain)</option>
              <option value={5e-6}>5×10⁻⁶ (heavy rain)</option>
              <option value={1e-5}>1×10⁻⁵ (extreme)</option>
            </select>
          </div>
          <SliderField label={`Duration: ${duration} hours`} value={duration} min={1} max={120} step={1} onChange={setDuration} />
          <button onClick={simulate} disabled={loading}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition h-10">
            {loading ? 'Simulating…' : '🌧️ Simulate'}
          </button>
        </div>
      </Card>

      {!results ? (
        <Card className="text-center py-12 text-slate-400">
          <div className="text-4xl mb-3">🌧️</div>
          <p className="text-sm">Configure rainfall parameters and click <strong>Simulate</strong></p>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="Intensity" value={`${results.intensity.toExponential(1)} m/s`} color="blue" />
            <MetricCard label="Duration" value={`${results.duration} hrs`} color="amber" />
            <MetricCard label="Cumulative" value={`${(results.intensity * results.duration * 3600 * 1000).toFixed(1)} mm`} color="green" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Dry vs Wet comparison at selected times */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">ψ Profiles: Dry vs Wet</h3>
              <Chart
                data={results.profiles.filter((_, i) => i % 2 === 0).flatMap((p, i) => [
                  { x: p.psi_dry, y: p.depths, type: 'scatter', mode: 'lines',
                    name: `Dry t=${p.time}d`, line: { color: COLORS[i], width: 1.5, dash: 'dot' },
                    legendgroup: `t${p.time}` },
                  { x: p.psi_wet, y: p.depths, type: 'scatter', mode: 'lines',
                    name: `Wet t=${p.time}d`, line: { color: COLORS[i], width: 2.5 },
                    legendgroup: `t${p.time}` },
                ])}
                layout={{
                  xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 420, legend: { font: { size: 9 } },
                }}
              />
            </Card>

            {/* ψ change heatmap */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">ψ Change (Wet − Dry)</h3>
              <Chart
                data={[{
                  z: results.profiles.map(p => p.psi_wet.map((w, i) => w - p.psi_dry[i])),
                  x: results.profiles[0].depths,
                  y: results.profiles.map(p => `Day ${p.time}`),
                  type: 'heatmap', colorscale: 'RdBu',
                  colorbar: { title: { text: 'Δψ (m)', side: 'right' }, thickness: 15 },
                }]}
                layout={{
                  xaxis: { title: 'Depth (m)' }, height: 420,
                }}
              />
            </Card>
          </div>

          <Card>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">📘 Note</h3>
            <p className="text-xs text-slate-600">
              This is a conceptual scenario overlay. The base ψ field comes from the trained PINN.
              A simplified 1D infiltration perturbation is applied (exponential decay with depth, linear ramp with time).
              For rigorous coupled rainfall-infiltration analysis, the PINN should be retrained with time-varying flux boundary conditions.
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
