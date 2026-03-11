import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { MetricCard, PageHeader, Spinner } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function Overview() {
  const { geo, defaults } = useApp();
  const [profiles, setProfiles] = useState(null);

  useEffect(() => {
    const times = [0, 30, 60, 90, 96, 123];
    Promise.all([
      api.predict({ z: [5], t: [96] }),
      api.factorOfSafety({ z: [5], t: [96], geo }),
      ...times.map(t => api.predict({ z: Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.5), t: Array(100).fill(t) })),
      ...times.map(t => api.factorOfSafety({ z: Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.5), t: Array(100).fill(t), geo })),
    ]).then(([psiPoint, fsPoint, ...rest]) => {
      const psiProfiles = rest.slice(0, times.length);
      const fsProfiles = rest.slice(times.length);
      setProfiles({ psiPoint, fsPoint, psiProfiles, fsProfiles, times });
    });
  }, [geo]);

  if (!profiles) return <Spinner text="Computing overview…" />;

  const depths = Array.from({ length: 100 }, (_, i) => 0.5 + i * 0.5);
  const psiVal = profiles.psiPoint.psi[0].toFixed(1);
  const fsVal = profiles.fsPoint.fs[0];
  const fsLabel = fsVal >= 1.5 ? '🟢 Safe' : fsVal >= 1.0 ? '🟡 Marginal' : '🔴 Unstable';

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-700 to-blue-900 rounded-2xl p-6 text-white">
        <h1 className="text-2xl font-bold">🏔️ PINN Landslide Stability Dashboard</h1>
        <p className="text-blue-200 mt-1">Physics-Informed Neural Network for slope stability under rainfall infiltration</p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Data Points" value={defaults.n_data_points?.toLocaleString()} color="blue" />
        <MetricCard label="Time Span" value={`0 – ${defaults.norm.t_max} d`} color="purple" />
        <MetricCard label="Depth Range" value={`${defaults.depth_range[0]} – ${defaults.depth_range[1]} m`} color="blue" />
        <MetricCard label="ψ at 5m, Day 96" value={`${psiVal} m`} color="amber" />
        <MetricCard label="FS at 5m, Day 96" value={fsVal.toFixed(2)} sub={fsLabel} color={fsVal >= 1.5 ? 'green' : fsVal >= 1 ? 'amber' : 'red'} />
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">📊 Pressure Head Profiles (PINN)</h3>
          <Chart
            data={profiles.psiProfiles.map((p, i) => ({
              x: p.psi, y: depths, type: 'scatter', mode: 'lines',
              name: `Day ${profiles.times[i]}`, line: { color: COLORS[i], width: 2 },
            }))}
            layout={{
              xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
              height: 420, legend: { orientation: 'h', y: 1.1 },
            }}
          />
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">🛡️ Factor of Safety Profiles</h3>
          <Chart
            data={[
              ...profiles.fsProfiles.map((p, i) => ({
                x: p.fs, y: depths, type: 'scatter', mode: 'lines',
                name: `Day ${profiles.times[i]}`, line: { color: COLORS[i], width: 2 },
              })),
              { x: [1, 1], y: [0, 50], type: 'scatter', mode: 'lines', name: 'FS=1',
                line: { color: '#dc2626', dash: 'dash', width: 2 }, showlegend: true },
            ]}
            layout={{
              xaxis: { title: 'Factor of Safety', range: [0, 10] },
              yaxis: { title: 'Depth (m)', autorange: 'reversed' },
              height: 420, legend: { orientation: 'h', y: 1.1 },
            }}
          />
        </div>
      </div>

      {/* About */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-3">
          <h3 className="font-semibold text-slate-900">📖 About This Project</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            This dashboard visualises a <strong>Physics-Informed Neural Network (PINN)</strong> trained to predict
            subsurface pore-water pressure and slope stability during a rainfall event on a natural hillslope.
          </p>
          <ul className="text-sm text-slate-600 list-disc list-inside space-y-1">
            <li><strong>Richards' Equation</strong> — unsaturated water flow PDE</li>
            <li><strong>Van Genuchten</strong> — soil–water retention model</li>
            <li><strong>Mohr–Coulomb</strong> — shear-strength criterion</li>
            <li><strong>Infinite-slope</strong> — stability model with matric suction</li>
          </ul>
        </div>
        <div>
          <h3 className="font-semibold text-slate-900 mb-2">Current Parameters</h3>
          <div className="text-xs space-y-1">
            {[['β (slope)', `${geo.beta}°`], ['c′', `${geo.c_prime} kPa`], ['φ′', `${geo.phi_prime}°`],
              ['γ', `${geo.gamma} kN/m³`], ['Ks', `${geo.Ks.toExponential(2)} m/s`],
              ['θs', geo.theta_s], ['θr', geo.theta_r], ['α', `${geo.alpha} 1/m`], ['n', geo.n]
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-slate-100 pb-0.5">
                <span className="text-slate-500">{k}</span>
                <span className="font-semibold text-slate-900">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
