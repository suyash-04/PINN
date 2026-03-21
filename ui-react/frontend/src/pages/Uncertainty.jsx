// Uncertainty.jsx
import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SectionTitle, SliderField, Button } from '../components/ui';
import Chart, { DANGER } from '../components/Chart';

export default function Uncertainty() {
  const { geo, norm } = useApp();
  const [time,     setTime]     = useState(Math.round(norm.t_max * 0.78));
  const [nSamples, setNSamples] = useState(100);
  const [covFrac,  setCovFrac]  = useState(0.10);
  const [data,     setData]     = useState(null);
  const [loading,  setLoading]  = useState(false);

  const run = () => {
    setLoading(true);
    setData(null);
    api.getUncertainty({
      time, n_samples: nSamples, cov_frac: covFrac,
      z_min: 0.5, z_max: 40.0, z_res: 80, geo, norm,
    }).then(d => {
      const meanFS = d.mean.reduce((a, b) => a + b, 0) / d.mean.length;
      const stdFS  = d.std.reduce((a, b) => a + b, 0)  / d.std.length;
      setData({
        ...d,
        mean_fs:       meanFS,
        std_fs:        stdFS,
        fs_5th:        Math.min(...d.p5),
        prob_failure:  d.p5.filter(v => v < 1).length / d.p5.length,
      });
      setLoading(false);
    }).catch(() => setLoading(false));
  };

  // Auto-run on mount
  useEffect(() => { run(); }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1200 }}>
      <PageHeader
        title="Uncertainty Quantification"
        subtitle="Monte Carlo analysis of FS under ±CoV perturbation of soil parameters"
        badge="Monte Carlo"
      />

      <Card>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr) auto', gap: 20, alignItems: 'end' }}>
          <SliderField label="Analysis time" value={time} onChange={setTime}
            min={0} max={norm.t_max} step={1} fmt={v => `Day ${v.toFixed(0)}`} />
          <SliderField label="MC samples" value={nSamples} onChange={setNSamples}
            min={50} max={500} step={50} />
          <SliderField label="CoV fraction" value={covFrac} onChange={setCovFrac}
            min={0.02} max={0.30} step={0.01} fmt={v => `±${(v*100).toFixed(0)}%`} />
          <Button onClick={run} disabled={loading} variant="primary">
            {loading ? 'Running…' : 'Run →'}
          </Button>
        </div>
      </Card>

      {loading && <Spinner text={`Running ${nSamples} Monte Carlo samples…`} />}

      {data && !loading && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            <MetricCard label="Mean FS"    value={data.mean_fs.toFixed(4)}  color="blue" />
            <MetricCard label="Std FS"     value={data.std_fs.toFixed(4)}   color="amber" />
            <MetricCard label="P(FS < 1)"  value={`${(data.prob_failure*100).toFixed(1)}%`} color="red" />
            <MetricCard label="FS 5th %ile" value={data.fs_5th.toFixed(4)}  color="purple" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Uncertainty band */}
            <Card>
              <SectionTitle>FS profile — mean ± P5/P95 band</SectionTitle>
              <Chart
                data={[
                  // Upper bound (invisible, fill base)
                  { x: data.p95, y: data.z, type: 'scatter', mode: 'lines',
                    line: { width: 0 }, showlegend: false },
                  // Lower bound (fill to previous)
                  { x: data.p5, y: data.z, type: 'scatter', mode: 'lines',
                    fill: 'tonexty', fillcolor: 'rgba(47,129,247,.15)',
                    line: { width: 0 }, name: 'P5–P95 band' },
                  // Mean
                  { x: data.mean, y: data.z, type: 'scatter', mode: 'lines',
                    name: 'Mean FS', line: { color: '#2f81f7', width: 2.5 } },
                  // Deterministic
                  { x: data.deterministic, y: data.z, type: 'scatter', mode: 'lines',
                    name: 'Deterministic', line: { color: '#3fb950', width: 1.5, dash: 'dot' } },
                  // FS=1 line
                  { x: [1, 1], y: [data.z[0], data.z.at(-1)], type: 'scatter', mode: 'lines',
                    name: 'FS = 1', line: { color: DANGER, dash: 'dash', width: 1.5 } },
                ]}
                layout={{
                  xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } } },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                  legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
                }}
                height={420}
              />
            </Card>

            {/* Histogram of pointwise mean FS */}
            <Card>
              <SectionTitle>FS distribution across depth (mean values)</SectionTitle>
              <Chart
                data={[{
                  x: data.mean, type: 'histogram', nbinsx: 30,
                  marker: { color: 'rgba(47,129,247,.5)' }, name: 'Mean FS',
                }]}
                layout={{
                  xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } } },
                  yaxis: { title: { text: 'Count', font: { size: 11 } } },
                  shapes: [
                    { type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                      line: { color: DANGER, width: 1.5, dash: 'dash' } },
                    { type: 'line', x0: data.mean_fs, x1: data.mean_fs, y0: 0, y1: 1, yref: 'paper',
                      line: { color: '#2f81f7', width: 1.5 } },
                  ],
                  bargap: 0.04,
                }}
                height={420}
              />
            </Card>
          </div>

          <Card>
            <SectionTitle>Methodology</SectionTitle>
            <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.8 }}>
              Each of the {nSamples} samples perturbs soil parameters
              (c′, φ′, γ, Ks, α, n) by sampling from a lognormal distribution
              with CoV = ±{(covFrac*100).toFixed(0)}%.
              The PINN-predicted ψ(z,t) is held fixed; only the geomechanical
              layer is resampled, isolating parametric uncertainty in the
              stability model. Reliability index β = (μ_FS − 1) / σ_FS
              = {((data.mean_fs - 1) / data.std_fs).toFixed(3)}.
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
