import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function Validation() {
  const { geo, norm, defaults } = useApp();
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('metrics');

  useEffect(() => {
    api.getValidation().then(d => {
      const overall = d.overall || {};
      const perTime = d.per_time || [];
      setData({
        overall_r2: overall.R2,
        overall_rmse: overall.RMSE,
        overall_mae: overall.MAE,
        max_error: Math.max(...perTime.map(p => p.RMSE || 0)),
        per_time: perTime.map(p => ({ time: p.time, r2: p.R2, rmse: p.RMSE, mae: p.MAE, nse: p.NSE, kge: p.KGE })),
        temporal_errors: {
          times: perTime.map(p => p.time),
          mean_err: perTime.map(p => p.MAE || 0),
          max_err: perTime.map(p => p.RMSE || 0),
        },
      });
    });
  }, [geo, norm]);

  if (!data) return <Spinner />;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Model Validation" subtitle="Comprehensive validation against HYDRUS-1D reference solutions" icon="✅" />

      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Overall R²" value={data.overall_r2?.toFixed(4) || 'N/A'} color="blue" />
        <MetricCard label="Overall RMSE" value={`${data.overall_rmse?.toFixed(3) || 'N/A'} m`} color="amber" />
        <MetricCard label="Overall MAE" value={`${data.overall_mae?.toFixed(3) || 'N/A'} m`} color="green" />
        <MetricCard label="Max Error" value={`${data.max_error?.toFixed(3) || 'N/A'} m`} color="red" />
      </div>

      <Card>
        <Tabs tabs={[
          { key: 'metrics', label: '📊 Metrics' },
          { key: 'temporal', label: '⏳ Temporal' },
          { key: 'spatial', label: '📍 Spatial' },
        ]} active={tab} onChange={setTab} />

        {tab === 'metrics' && data.per_time && (
          <div className="space-y-4">
            <Chart
              data={[
                { x: data.per_time.map(p => p.time), y: data.per_time.map(p => p.r2),
                  type: 'scatter', mode: 'lines+markers', name: 'R²',
                  line: { color: '#2563eb', width: 2.5 }, marker: { size: 6 } },
              ]}
              layout={{
                xaxis: { title: 'Time (days)' }, yaxis: { title: 'R²', range: [0.9, 1.01] },
                height: 400, shapes: [{ type: 'line', y0: 0.99, y1: 0.99, x0: 0, x1: data.per_time.length,
                  xref: 'x', line: { color: '#059669', dash: 'dash', width: 1 } }],
              }}
            />
            <Chart
              data={[
                { x: data.per_time.map(p => p.time), y: data.per_time.map(p => p.rmse),
                  type: 'bar', name: 'RMSE', marker: { color: '#f59e0b' } },
              ]}
              layout={{
                xaxis: { title: 'Time (days)' }, yaxis: { title: 'RMSE (m)' }, height: 350,
              }}
            />
          </div>
        )}

        {tab === 'temporal' && data.temporal_errors && (
          <Chart
            data={[
              { x: data.temporal_errors.times, y: data.temporal_errors.mean_err,
                type: 'scatter', mode: 'lines+markers', name: 'Mean |Error|',
                line: { color: '#2563eb', width: 2 }, marker: { size: 5 } },
              { x: data.temporal_errors.times, y: data.temporal_errors.max_err,
                type: 'scatter', mode: 'lines+markers', name: 'Max |Error|',
                line: { color: '#dc2626', width: 2, dash: 'dash' }, marker: { size: 5 } },
            ]}
            layout={{
              xaxis: { title: 'Time (days)' }, yaxis: { title: 'Error (m)' }, height: 450,
            }}
          />
        )}

        {tab === 'spatial' && data.spatial_errors && (
          <Chart
            data={[
              { x: data.spatial_errors.mean_err, y: data.spatial_errors.depths,
                type: 'scatter', mode: 'lines', name: 'Mean |Error|',
                line: { color: '#2563eb', width: 2.5 }, fill: 'tozerox', fillcolor: 'rgba(37,99,235,0.08)' },
              { x: data.spatial_errors.max_err, y: data.spatial_errors.depths,
                type: 'scatter', mode: 'lines', name: 'Max |Error|',
                line: { color: '#dc2626', width: 2, dash: 'dash' } },
            ]}
            layout={{
              xaxis: { title: 'Error (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' }, height: 450,
            }}
          />
        )}
      </Card>

      {/* Validation summary */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">📋 Validation Summary</h3>
        <div className="grid grid-cols-3 gap-4 text-xs text-slate-600">
          <div className="p-3 bg-green-50 rounded-lg">
            <p className="font-semibold text-green-700 mb-1">✅ Strengths</p>
            <ul className="list-disc pl-4 space-y-0.5">
              <li>Good overall R² correlation</li>
              <li>Smooth pressure head profiles</li>
              <li>Physical boundary conditions maintained</li>
            </ul>
          </div>
          <div className="p-3 bg-amber-50 rounded-lg">
            <p className="font-semibold text-amber-700 mb-1">⚠️ Limitations</p>
            <ul className="list-disc pl-4 space-y-0.5">
              <li>All unsaturated regime (ψ &lt; 0)</li>
              <li>Loss plateaued at ~2.87</li>
              <li>L-BFGS converged after 1 step</li>
            </ul>
          </div>
          <div className="p-3 bg-blue-50 rounded-lg">
            <p className="font-semibold text-blue-700 mb-1">💡 Recommendations</p>
            <ul className="list-disc pl-4 space-y-0.5">
              <li>Add more collocation points near wetting front</li>
              <li>Include saturated scenarios in training</li>
              <li>Try adaptive loss weighting</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
