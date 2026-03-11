import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function PDEResidual() {
  const { geo, norm } = useApp();
  const [res, setRes] = useState(null);
  const [gridRes, setGridRes] = useState(20);

  useEffect(() => {
    api.getPDEResidual({
      z_min: 0.5, z_max: 40.0, z_res: gridRes,
      t_min: 0.0, t_max: 123.0, t_res: gridRes,
      geo, norm,
    }).then(setRes);
  }, [geo, norm, gridRes]);

  if (!res) return <Spinner />;

  const absGrid = res.abs_residual; // 2D array [z_res x t_res]
  const flatAbs = absGrid.flat();
  const maxR = res.stats.max;
  const meanR = res.stats.mean;
  const medR = res.stats.median;

  // Log10 transform for heatmap
  const heatZ = absGrid.map(row => row.map(v => Math.log10(v + 1e-12)));
  const tVals = res.t;
  const zVals = res.z;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="PDE Residual Analysis" subtitle="Richards' equation residual: how well the PINN satisfies the governing physics" icon="📐" />

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Max |R|" value={maxR.toExponential(2)} color="red" />
        <MetricCard label="Mean |R|" value={meanR.toExponential(2)} color="amber" />
        <MetricCard label="Median |R|" value={medR.toExponential(2)} color="green" />
      </div>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-700">Residual Heatmap (log₁₀ scale)</h3>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span>Grid:</span>
            {[10, 15, 20, 30].map(g => (
              <button key={g} onClick={() => setGridRes(g)}
                className={`px-2 py-1 rounded ${gridRes === g ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                {g}×{g}
              </button>
            ))}
          </div>
        </div>
        <Chart
          data={[{
            z: heatZ, x: tVals, y: zVals, type: 'heatmap',
            colorscale: 'RdYlGn', reversescale: true,
            colorbar: { title: { text: 'log₁₀|R|', side: 'right' }, thickness: 15 },
          }]}
          layout={{
            xaxis: { title: 'Time (days)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
            height: 480,
          }}
        />
      </Card>

      {/* Histogram */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Residual Distribution</h3>
        <Chart
          data={[{
            x: flatAbs.map(v => Math.log10(v + 1e-12)), type: 'histogram', nbinsx: 40,
            marker: { color: '#6366f1' }, name: 'log₁₀|R|',
          }]}
          layout={{
            xaxis: { title: 'log₁₀|Residual|' }, yaxis: { title: 'Count' }, height: 350,
            bargap: 0.05,
          }}
        />
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">📘 Interpretation</h3>
        <div className="text-xs text-slate-600 space-y-1">
          <p>The PDE residual measures how well the PINN satisfies Richards' equation: <code>R = C(ψ)·∂ψ/∂t − ∂/∂z[K(ψ)·(∂ψ/∂z + 1)]</code></p>
          <p>Lower values indicate better physics compliance. Green regions satisfy the PDE well; red regions have higher violation.</p>
          <p>Residuals should ideally be O(10⁻³) or smaller for a well-trained PINN.</p>
        </div>
      </Card>
    </div>
  );
}
