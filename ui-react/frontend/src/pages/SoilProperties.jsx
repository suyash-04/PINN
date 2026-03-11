import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, Tabs } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function SoilProperties() {
  const { geo, norm } = useApp();
  const [data, setData] = useState(null);
  const [tab, setTab] = useState('wrc');

  useEffect(() => {
    api.getSoilProps({
      alpha: geo.alpha, n: geo.n, theta_s: geo.theta_s, theta_r: geo.theta_r,
      Ks: geo.Ks, l: geo.l || 0.5, n_points: 200,
      psi_min: -500, psi_max: -0.01,
    }).then(setData);
  }, [geo, norm]);

  if (!data) return <Spinner />;

  const { psi, Se, theta, K, C } = data;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Soil Hydraulic Properties" subtitle="Van Genuchten model curves for current soil parameters" icon="🧱" />

      {/* Key parameters */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          ['θs', geo.theta_s, '-'], ['θr', geo.theta_r, '-'], ['α', geo.alpha, '1/m'],
          ['n', geo.n, '-'], ['Ks', geo.Ks, 'm/s'],
        ].map(([lbl, val, unit]) => (
          <Card key={lbl} className="text-center">
            <p className="text-[10px] text-slate-400 font-medium">{lbl}</p>
            <p className="text-lg font-bold text-slate-800">{typeof val === 'number' ? val.toExponential(3) : val}</p>
            <p className="text-[10px] text-slate-400">{unit}</p>
          </Card>
        ))}
      </div>

      <Card>
        <Tabs tabs={[
          { key: 'wrc', label: '💧 WRC' },
          { key: 'hcf', label: '🔄 K(ψ)' },
          { key: 'capacity', label: '📐 C(ψ)' },
          { key: 'all', label: '📊 Combined' },
        ]} active={tab} onChange={setTab} />

        {tab === 'wrc' && (
          <Chart
            data={[
              { x: psi, y: theta, type: 'scatter', mode: 'lines', name: 'θ(ψ)',
                line: { color: '#2563eb', width: 3 }, fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.08)' },
              { x: psi, y: Se, type: 'scatter', mode: 'lines', name: 'Se(ψ)',
                line: { color: '#7c3aed', width: 2.5, dash: 'dash' }, yaxis: 'y2' },
            ]}
            layout={{
              xaxis: { title: 'Suction head |ψ| (m)', type: 'log' },
              yaxis: { title: 'θ (m³/m³)', rangemode: 'tozero' },
              yaxis2: { title: 'Se (-)', overlaying: 'y', side: 'right', range: [0, 1.05] },
              height: 480,
            }}
          />
        )}

        {tab === 'hcf' && (
          <Chart
            data={[{ x: psi, y: K, type: 'scatter', mode: 'lines', name: 'K(ψ)',
              line: { color: '#059669', width: 3 } }]}
            layout={{
              xaxis: { title: '|ψ| (m)', type: 'log' },
              yaxis: { title: 'K (m/s)', type: 'log' }, height: 480,
            }}
          />
        )}

        {tab === 'capacity' && (
          <Chart
            data={[{ x: psi, y: C, type: 'scatter', mode: 'lines', name: 'C(ψ)',
              line: { color: '#d97706', width: 3 } }]}
            layout={{
              xaxis: { title: '|ψ| (m)', type: 'log' },
              yaxis: { title: 'C (1/m)', type: 'log' }, height: 480,
            }}
          />
        )}

        {tab === 'all' && (
          <div className="grid grid-cols-2 gap-4">
            <Chart data={[{ x: psi, y: theta, type: 'scatter', mode: 'lines',
              line: { color: '#2563eb', width: 2.5 } }]}
              layout={{ xaxis: { title: '|ψ| (m)', type: 'log' }, yaxis: { title: 'θ' }, height: 300,
                title: { text: 'Water Retention', font: { size: 13 } } }} />
            <Chart data={[{ x: psi, y: Se, type: 'scatter', mode: 'lines',
              line: { color: '#7c3aed', width: 2.5 } }]}
              layout={{ xaxis: { title: '|ψ| (m)', type: 'log' }, yaxis: { title: 'Se' }, height: 300,
                title: { text: 'Effective Saturation', font: { size: 13 } } }} />
            <Chart data={[{ x: psi, y: K, type: 'scatter', mode: 'lines',
              line: { color: '#059669', width: 2.5 } }]}
              layout={{ xaxis: { title: '|ψ| (m)', type: 'log' }, yaxis: { title: 'K (m/s)', type: 'log' }, height: 300,
                title: { text: 'Hydraulic Conductivity', font: { size: 13 } } }} />
            <Chart data={[{ x: psi, y: C, type: 'scatter', mode: 'lines',
              line: { color: '#d97706', width: 2.5 } }]}
              layout={{ xaxis: { title: '|ψ| (m)', type: 'log' }, yaxis: { title: 'C (1/m)', type: 'log' }, height: 300,
                title: { text: 'Specific Moisture Capacity', font: { size: 13 } } }} />
          </div>
        )}
      </Card>

      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-2">📘 Van Genuchten Model</h3>
        <div className="text-xs text-slate-600 space-y-1">
          <p><strong>Water Retention:</strong> θ(ψ) = θr + (θs − θr) · [1 + (α|ψ|)ⁿ]^(−m), where m = 1 − 1/n</p>
          <p><strong>Hydraulic Conductivity:</strong> K(Se) = Ks · Se^0.5 · [1 − (1 − Se^(1/m))^m]²</p>
          <p><strong>Specific Moisture Capacity:</strong> C(ψ) = dθ/dψ = α·m·n·(θs−θr)·(α|ψ|)^(n−1) · [1+(α|ψ|)ⁿ]^(−m−1)</p>
        </div>
      </Card>
    </div>
  );
}
