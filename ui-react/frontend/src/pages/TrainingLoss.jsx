import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function TrainingLoss() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('curve');

  useEffect(() => {
    api.getLossHistory()
      .then(res => {
        setHistory(res.history || []);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load loss history:", err);
        setLoading(false);
      });
  }, []);

  if (loading) return <Spinner />;
  if (history.length === 0) {
    return (
      <Card>
        <div className="p-8 text-center text-slate-500 italic">
          No training history found. Please run model training first.
        </div>
      </Card>
    );
  }

  // Pre-process history data for plotting
  // We'll use the 'step' as the x-axis. Since steps repeat for Adam/L-BFGS,
  // we'll create a continuous 'globalStep'.
  let globalStep = 0;
  const processed = history.map((h, i) => {
    const prev = history[i-1];
    if (prev && h.phase !== prev.phase) {
        // Phase transition
    }
    return {
      ...h,
      dataLoss: (h.anchor || 0) + (h.failure || 0),
      bcLoss: (h.boundary || 0) + (h.initial || 0),
      label: `${h.phase} ${h.step}`
    };
  });

  const steps = processed.map((_, i) => i + 1);
  const totalLoss = processed.map(h => h.total);
  const dataLoss = processed.map(h => h.dataLoss);
  const pdeLoss = processed.map(h => h.physics);
  const bcLoss = processed.map(h => h.bcLoss);

  const finalTotal = totalLoss[totalLoss.length - 1];
  const finalData = dataLoss[dataLoss.length - 1];
  const finalPDE = pdeLoss[pdeLoss.length - 1];
  const totalEpochs = processed.length;

  // Find the split point between Adam and L-BFGS
  const lbfgsIndex = history.findIndex(h => h.phase === 'LBFGS');

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Training Loss Analysis" subtitle="Loss evolution during PINN training (Adam + L-BFGS phases)" icon="📉" />

      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Final Total Loss" value={finalTotal.toExponential(3)} color="blue" />
        <MetricCard label="Final Data Loss" value={finalData.toExponential(3)} color="green" />
        <MetricCard label="Final PDE Loss" value={finalPDE.toExponential(3)} color="amber" />
        <MetricCard label="Logged steps" value={totalEpochs} color="purple" />
      </div>

      <Card>
        <Tabs tabs={[
          { key: 'curve', label: '📈 Loss Curves' },
          { key: 'components', label: '📊 Components' },
          { key: 'log', label: '🔢 Log Scale' },
        ]} active={tab} onChange={setTab} />

        {tab === 'curve' && (
          <Chart
            data={[
              { x: steps, y: totalLoss, type: 'scatter', mode: 'lines', name: 'Total',
                line: { color: '#2563eb', width: 2.5 } },
              { x: steps, y: dataLoss, type: 'scatter', mode: 'lines', name: 'Data',
                line: { color: '#059669', width: 2 } },
              { x: steps, y: pdeLoss, type: 'scatter', mode: 'lines', name: 'PDE',
                line: { color: '#d97706', width: 2 } },
              { x: steps, y: bcLoss, type: 'scatter', mode: 'lines', name: 'BC/IC',
                line: { color: '#7c3aed', width: 2 } },
            ]}
            layout={{
              xaxis: { title: 'Logging Step' }, yaxis: { title: 'Loss' }, height: 480,
              shapes: lbfgsIndex !== -1 ? [{
                type: 'line', x0: lbfgsIndex + 1, x1: lbfgsIndex + 1, y0: 0, y1: 1, yref: 'paper',
                line: { color: '#dc2626', dash: 'dot', width: 1.5 }
              }] : [],
              annotations: lbfgsIndex !== -1 ? [{
                x: lbfgsIndex + 1, y: 0.95, yref: 'paper', text: 'L-BFGS →', showarrow: false,
                font: { color: '#dc2626', size: 10 }, xanchor: 'left'
              }] : [],
            }}
          />
        )}

        {tab === 'components' && (
          <Chart
            data={[
              { x: steps, y: dataLoss.map((d, i) => d / totalLoss[i] * 100), type: 'scatter', mode: 'lines',
                name: 'Data %', line: { color: '#059669' }, stackgroup: 'one', fillcolor: 'rgba(5,150,105,0.3)' },
              { x: steps, y: pdeLoss.map((d, i) => d / totalLoss[i] * 100), type: 'scatter', mode: 'lines',
                name: 'PDE %', line: { color: '#d97706' }, stackgroup: 'one', fillcolor: 'rgba(217,119,6,0.3)' },
              { x: steps, y: bcLoss.map((d, i) => d / totalLoss[i] * 100), type: 'scatter', mode: 'lines',
                name: 'BC/IC %', line: { color: '#7c3aed' }, stackgroup: 'one', fillcolor: 'rgba(124,58,237,0.3)' },
            ]}
            layout={{
              xaxis: { title: 'Logging Step' }, yaxis: { title: 'Loss Component (%)' }, height: 450,
            }}
          />
        )}

        {tab === 'log' && (
          <Chart
            data={[
              { x: steps, y: totalLoss, type: 'scatter', mode: 'lines', name: 'Total',
                line: { color: '#2563eb', width: 2.5 } },
              { x: steps, y: dataLoss, type: 'scatter', mode: 'lines', name: 'Data',
                line: { color: '#059669', width: 2 } },
              { x: steps, y: pdeLoss, type: 'scatter', mode: 'lines', name: 'PDE',
                line: { color: '#d97706', width: 2 } },
            ]}
            layout={{
              xaxis: { title: 'Logging Step' }, yaxis: { title: 'Loss (log)', type: 'log' }, height: 480,
            }}
          />
        )}
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">📘 Training Strategy</h3>
        <div className="text-xs text-slate-600 space-y-2">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-blue-50 rounded-lg">
              <p className="font-semibold text-blue-700 mb-1">Phase 1: Adam Optimizer</p>
              <p>Stochastic gradient descent with adaptive learning rate. Explores the loss landscape broadly and handles noisy gradients well. Learning rate scheduling reduces lr over epochs.</p>
            </div>
            <div className="p-3 bg-purple-50 rounded-lg">
              <p className="font-semibold text-purple-700 mb-1">Phase 2: L-BFGS</p>
              <p>Quasi-Newton method for fine-tuning. Uses full-batch gradients with curvature information for faster local convergence. Typically runs for fewer iterations but each is more expensive.</p>
            </div>
          </div>
          <p className="text-slate-500 italic">Note: The physics loss (PDE residual) is weighted heavily (λ_physics = 1e6) to enforce Richards' Equation. All training data is currently in the unsaturated regime (ψ &lt; 0).</p>
        </div>
      </Card>
    </div>
  );
}
