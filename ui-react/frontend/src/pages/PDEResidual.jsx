// PDEResidual.jsx
import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SectionTitle } from '../components/ui';
import Chart from '../components/Chart';

export default function PDEResidual() {
  const { geo, norm } = useApp();
  const [res, setRes]       = useState(null);
  const [gridRes, setGridRes] = useState(20);

  useEffect(() => {
    setRes(null);
    api.getPDEResidual({
      z_min: 0.5, z_max: 40.0, z_res: gridRes,
      t_min: 0.0, t_max: norm.t_max, t_res: gridRes,
      geo, norm,
    }).then(setRes);
  }, [geo, norm, gridRes]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1100 }}>
      <PageHeader
        title="PDE Residual"
        subtitle="Richards' equation residual — how well the PINN satisfies the governing physics"
        badge="Physics check"
      />

      {/* Grid resolution buttons */}
      <Card>
        <SectionTitle>Grid resolution</SectionTitle>
        <div style={{ display: 'flex', gap: 8 }}>
          {[10, 15, 20, 30, 40].map(g => (
            <button key={g} onClick={() => setGridRes(g)} style={{
              padding: '4px 14px', fontSize: 11, fontFamily: 'var(--font-mono)', cursor: 'pointer',
              borderRadius: 4, border: gridRes === g ? '1px solid var(--accent)' : '1px solid var(--border)',
              background: gridRes === g ? 'rgba(47,129,247,.15)' : 'transparent',
              color: gridRes === g ? 'var(--accent)' : 'var(--muted)',
            }}>
              {g}×{g}
            </button>
          ))}
        </div>
      </Card>

      {!res
        ? <Spinner text="Computing PDE residuals…" />
        : <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
              <MetricCard label="Max |R|"    value={res.stats.max.toExponential(2)}    color="red" />
              <MetricCard label="Mean |R|"   value={res.stats.mean.toExponential(2)}   color="amber" />
              <MetricCard label="Median |R|" value={res.stats.median.toExponential(2)} color="green" />
            </div>

            <Card>
              <SectionTitle>Residual heatmap — log₁₀ scale</SectionTitle>
              <Chart
                data={[{
                  z: res.abs_residual.map(row => row.map(v => Math.log10(v + 1e-14))),
                  x: res.t, y: res.z, type: 'heatmap',
                  colorscale: 'RdYlGn', reversescale: true,
                  colorbar: { title: { text: 'log₁₀|R|', side: 'right' }, thickness: 14,
                    tickfont: { size: 10, color: '#7d8590' },
                    tickcolor: '#30363d', outlinecolor: '#21262d' },
                  hovertemplate: 'Day: %{x:.1f}<br>Depth: %{y:.1f} m<br>log|R|: %{z:.2f}<extra></extra>',
                }]}
                layout={{
                  xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                }}
                height={460}
              />
            </Card>

            <Card>
              <SectionTitle>Residual distribution</SectionTitle>
              <Chart
                data={[{
                  x: res.abs_residual.flat().map(v => Math.log10(v + 1e-14)),
                  type: 'histogram', nbinsx: 50,
                  marker: { color: 'rgba(47,129,247,.5)' }, name: 'log₁₀|R|',
                }]}
                layout={{
                  xaxis: { title: { text: 'log₁₀|Residual|', font: { size: 11 } } },
                  yaxis: { title: { text: 'Count', font: { size: 11 } } },
                  bargap: 0.04,
                }}
                height={300}
              />
            </Card>

            <Card>
              <SectionTitle>Interpretation guide</SectionTitle>
              <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.7,
                fontFamily: 'var(--font-mono)' }}>
                <div>R = C(ψ)·∂ψ/∂t − ∂/∂z[K(ψ)·(∂ψ/∂z + 1)]</div>
                <div style={{ marginTop: 8, fontFamily: 'var(--font-sans)' }}>
                  Green zones satisfy Richards' equation well (log|R| ≪ −3).
                  Red zones indicate higher PDE violation — typically near the wetting front where
                  gradients are steepest. Residuals below 10⁻³ are considered acceptable for a
                  well-trained PINN.
                </div>
              </div>
            </Card>
          </>
      }
    </div>
  );
}
