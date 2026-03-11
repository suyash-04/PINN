import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SliderField } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function Uncertainty() {
  const { geo, norm } = useApp();
  const [time, setTime] = useState(90);
  const [nSamples, setNSamples] = useState(100);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = () => {
    setLoading(true);
    api.getUncertainty({
      time, n_samples: nSamples, geo, norm,
      z_min: 0.5, z_max: 40.0, z_res: 80, cov_frac: 0.1,
    }).then(d => {
      // Compute derived stats from the response
      const meanFS = d.mean.reduce((a, b) => a + b, 0) / d.mean.length;
      const stdFS = d.std.reduce((a, b) => a + b, 0) / d.std.length;
      const minFS_samples = Math.min(...d.p5);
      setData({
        ...d,
        mean_fs: meanFS,
        std_fs: stdFS,
        fs_5th: minFS_samples,
        prob_failure: d.p5.filter(v => v < 1).length / d.p5.length,
        fs_samples: d.deterministic, // Use deterministic as a proxy
        depth_profiles: {
          depths: d.z,
          fs_mean: d.mean,
          fs_upper: d.p95,
          fs_lower: d.p5,
        },
      });
      setLoading(false);
    });
  };

  useEffect(() => { run(); }, []);

  if (!data && loading) return <Spinner />;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Uncertainty Quantification" subtitle="Monte Carlo analysis of factor of safety under parameter uncertainty" icon="🎲" />

      <Card>
        <div className="grid grid-cols-3 gap-4 items-end">
          <SliderField label="Time (days)" value={time} min={0} max={123} step={1} onChange={setTime} />
          <SliderField label="MC Samples" value={nSamples} min={50} max={500} step={50} onChange={setNSamples} />
          <button onClick={run} disabled={loading}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition h-10">
            {loading ? 'Running…' : '🎯 Run Analysis'}
          </button>
        </div>
      </Card>

      {data && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <MetricCard label="Mean FS" value={data.mean_fs?.toFixed(3)} color="blue" />
            <MetricCard label="Std FS" value={data.std_fs?.toFixed(3)} color="amber" />
            <MetricCard label="P(FS<1)" value={`${(data.prob_failure * 100).toFixed(1)}%`} color="red" />
            <MetricCard label="FS 5th %ile" value={data.fs_5th?.toFixed(3)} color="purple" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* FS Distribution */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">FS Distribution</h3>
              <Chart
                data={[
                  { x: data.fs_samples, type: 'histogram', nbinsx: 30,
                    marker: { color: 'rgba(37,99,235,0.6)' }, name: 'FS samples' },
                ]}
                layout={{
                  xaxis: { title: 'Factor of Safety' }, yaxis: { title: 'Count' }, height: 380,
                  shapes: [
                    { type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                      line: { color: '#dc2626', width: 2, dash: 'dash' } },
                    { type: 'line', x0: data.mean_fs, x1: data.mean_fs, y0: 0, y1: 1, yref: 'paper',
                      line: { color: '#2563eb', width: 2 } },
                  ],
                  annotations: [
                    { x: 1, y: 1, yref: 'paper', text: 'FS=1', showarrow: false, yanchor: 'bottom',
                      font: { color: '#dc2626', size: 10 } },
                  ],
                  bargap: 0.05,
                }}
              />
            </Card>

            {/* FS profile with uncertainty bands */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">FS Profile (mean ± σ)</h3>
              <Chart
                data={[
                  ...(data.depth_profiles ? [
                    { x: data.depth_profiles.fs_upper, y: data.depth_profiles.depths,
                      type: 'scatter', mode: 'lines', line: { width: 0 }, showlegend: false },
                    { x: data.depth_profiles.fs_lower, y: data.depth_profiles.depths,
                      type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(37,99,235,0.15)',
                      line: { width: 0 }, name: '±1σ band' },
                    { x: data.depth_profiles.fs_mean, y: data.depth_profiles.depths,
                      type: 'scatter', mode: 'lines', name: 'Mean FS',
                      line: { color: '#2563eb', width: 2.5 } },
                  ] : []),
                  { x: [1, 1], y: [0, norm.z_max], type: 'scatter', mode: 'lines',
                    line: { color: '#dc2626', dash: 'dash', width: 1.5 }, name: 'FS=1' },
                ]}
                layout={{
                  xaxis: { title: 'Factor of Safety' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 380,
                }}
              />
            </Card>
          </div>

          {/* Reliability info */}
          <Card>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">📘 Methodology</h3>
            <div className="text-xs text-slate-600 space-y-1">
              <p>Monte Carlo simulation perturbs soil parameters (c', φ', γ, Ks, α, n) with ±10% coefficient of variation (lognormal distribution).</p>
              <p>For each sample, the full FS profile is computed using the PINN-predicted pressure head field.</p>
              <p><strong>Probability of Failure:</strong> P(FS &lt; 1) = {(data.prob_failure * 100).toFixed(1)}% based on {nSamples} samples at t = {time} days.</p>
              <p>A reliability index β can be estimated as β = (μ_FS − 1) / σ_FS = {data.mean_fs && data.std_fs ? ((data.mean_fs - 1) / data.std_fs).toFixed(2) : 'N/A'}</p>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
