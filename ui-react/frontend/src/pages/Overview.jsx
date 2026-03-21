import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { MetricCard, Card, Spinner, StatusBadge, SectionTitle } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function Overview() {
  const { geo, defaults } = useApp();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const refDepth = 5;
    const refTime = defaults.norm.t_max;
    // Use actual available timesteps from the backend
    const times = defaults.available_times.slice(0, 6);
    const depths100 = Array.from({ length: 100 }, (_, i) => 0.5 + i * (defaults.norm.z_max / 100));

    Promise.all([
      api.predict({ z: [refDepth], t: [refTime] }),
      api.factorOfSafety({ z: [refDepth], t: [refTime], geo }),
      ...times.map(t => api.predict({ z: depths100, t: Array(100).fill(t) })),
      ...times.map(t => api.factorOfSafety({ z: depths100, t: Array(100).fill(t), geo })),
    ])
      .then(([psiPt, fsPt, ...rest]) => {
        setData({
          psiVal: psiPt.psi[0],
          fsVal: fsPt.fs[0],
          psiProfiles: rest.slice(0, times.length),
          fsProfiles: rest.slice(times.length),
          depths: depths100,
          times,
          refDepth,
          refTime,
        });
      })
      .catch(e => setError(e.message));
  }, [geo, defaults]);

  if (error) return <ErrorCard msg={error} />;
  if (!data) return <Spinner text="Computing overview…" />;

  const { psiVal, fsVal, psiProfiles, fsProfiles, depths, times } = data;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>

      {/* Header banner */}
      <div style={{ padding: '16px 20px', borderRadius: 8,
        background: 'linear-gradient(135deg, rgba(47,129,247,.12) 0%, rgba(163,113,247,.08) 100%)',
        border: '1px solid rgba(47,129,247,.2)' }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)',
          letterSpacing: '-0.02em' }}>
          PINN Landslide Stability Dashboard
        </div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4,
          fontFamily: 'var(--font-mono)' }}>
          Physics-Informed Neural Network · Richards Equation · Jure 2014 · 
          {' '}{defaults.n_data_points?.toLocaleString()} training points
        </div>
      </div>

      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12 }}>
        <MetricCard label="Training points" value={defaults.n_data_points?.toLocaleString()} color="blue" />
        <MetricCard label="Time span" value={`0 – ${defaults.norm.t_max} d`} color="muted" />
        <MetricCard label="Depth range"
          value={`${defaults.depth_range?.[0]} – ${defaults.depth_range?.[1]} m`} color="muted" />
        <MetricCard label={`ψ at 5 m, day ${defaults.norm.t_max}`}
          value={`${psiVal.toFixed(1)} m`} color="amber" />
        <MetricCard label={`FS at 5 m, day ${defaults.norm.t_max}`}
          value={fsVal.toFixed(3)}
          sub={<StatusBadge fs={fsVal} />}
          color={fsVal >= 1.5 ? 'green' : fsVal >= 1 ? 'amber' : 'red'} />
      </div>

      {/* Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card>
          <SectionTitle>Pressure Head Profiles — ψ(z)</SectionTitle>
          <Chart
            data={psiProfiles.map((p, i) => ({
              x: p.psi, y: depths, type: 'scatter', mode: 'lines',
              name: `Day ${times[i]}`,
              line: { color: COLORS[i], width: 1.8 },
            }))}
            layout={{
              xaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
              yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
              legend: { orientation: 'h', y: 1.12, xanchor: 'center', x: 0.5 },
            }}
            height={380}
          />
        </Card>
        <Card>
          <SectionTitle>Factor of Safety Profiles — FS(z)</SectionTitle>
          <Chart
            data={[
              ...fsProfiles.map((p, i) => ({
                x: p.fs, y: depths, type: 'scatter', mode: 'lines',
                name: `Day ${times[i]}`,
                line: { color: COLORS[i], width: 1.8 },
              })),
              { x: [1, 1], y: [0, defaults.norm.z_max], type: 'scatter', mode: 'lines',
                name: 'FS = 1', line: { color: '#f85149', dash: 'dash', width: 1.5 },
                showlegend: true },
            ]}
            layout={{
              xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } }, range: [0, 8] },
              yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
              legend: { orientation: 'h', y: 1.12, xanchor: 'center', x: 0.5 },
            }}
            height={380}
          />
        </Card>
      </div>

      {/* Model + params info row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
        <Card>
          <SectionTitle>Model Physics</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11,
            color: 'var(--muted)' }}>
            {[
              ['Governing PDE', "Richards' Equation (1D unsaturated)"],
              ['Hydraulic model', 'Van Genuchten–Mualem'],
              ['Stability model', 'Infinite slope (Mohr–Coulomb)'],
              ['Network', '7 × 64 tanh, 29,377 params'],
              ['Training', 'Adam (10k) + L-BFGS (5k)'],
              ['Loss terms', 'Data · PDE · IC · BC · Failure'],
            ].map(([k, v]) => (
              <div key={k} style={{ padding: '8px 12px', background: 'rgba(255,255,255,.02)',
                borderRadius: 6, border: '1px solid var(--border)' }}>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 3 }}>{k}</div>
                <div style={{ fontSize: 11, color: 'var(--text)', fontFamily:'var(--font-mono)' }}>{v}</div>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <SectionTitle>Active Parameters</SectionTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontFamily: 'var(--font-mono)',
            fontSize: 11 }}>
            {[
              ['β',  `${geo.beta}°`,               'slope angle'],
              ["c'", `${geo.c_prime} kPa`,         'effective cohesion'],
              ["φ'", `${geo.phi_prime}°`,           'friction angle'],
              ['γ',  `${geo.gamma} kN/m³`,         'unit weight'],
              ['Ks', geo.Ks.toExponential(2)+' m/s','hydraulic cond.'],
              ['θs', geo.theta_s,                  'sat. water content'],
              ['α',  `${geo.alpha} 1/m`,           'VG alpha'],
              ['n',  geo.n,                         'VG n'],
            ].map(([k, v, hint]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', padding: '4px 0',
                borderBottom: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--muted)', minWidth: 28 }}>{k}</span>
                <span style={{ color: 'var(--text)', fontWeight: 600 }}>{v}</span>
                <span style={{ color: 'var(--muted)', fontSize: 10, opacity: .6 }}>{hint}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function ErrorCard({ msg }) {
  return (
    <div style={{ padding: 20, color: 'var(--danger)', background: 'var(--surface)',
      border: '1px solid var(--border)', borderRadius: 8, fontFamily: 'var(--font-mono)',
      fontSize: 12 }}>
      Error: {msg}
    </div>
  );
}
