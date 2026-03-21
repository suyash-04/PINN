// SoilProperties.jsx
import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, Tabs, SectionTitle } from '../components/ui';
import Chart from '../components/Chart';

export default function SoilProperties() {
  const { geo } = useApp();
  const [data, setData] = useState(null);
  const [tab, setTab]   = useState('wrc');

  useEffect(() => {
    api.soilProperties({
      alpha: geo.alpha, n: geo.n, theta_s: geo.theta_s, theta_r: geo.theta_r,
      Ks: geo.Ks, l: geo.l ?? 0.5, n_points: 300,
      psi_min: -500, psi_max: -0.01,
    }).then(setData);
  }, [geo]);

  if (!data) return <Spinner text="Computing VG curves…" />;

  const { psi, Se, theta, K, C } = data;
  // Use absolute suction for log axis
  const absPsi = psi.map(v => Math.abs(v));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="Soil Hydraulic Properties"
        subtitle="Van Genuchten–Mualem constitutive curves — updates with sidebar parameters"
        badge="VG model"
      />

      {/* Key params banner */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {[
          ['θs', geo.theta_s],
          ['θr', geo.theta_r],
          ['α',  `${geo.alpha} 1/m`],
          ['n',  geo.n],
          ['Ks', geo.Ks.toExponential(2)+' m/s'],
          ['l',  geo.l ?? 0.5],
        ].map(([k, v]) => (
          <div key={k} style={{ padding: '6px 14px', background: 'var(--surface)',
            border: '1px solid var(--border)', borderRadius: 6, fontFamily: 'var(--font-mono)',
            fontSize: 12 }}>
            <span style={{ color: 'var(--muted)', marginRight: 8 }}>{k}</span>
            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{v}</span>
          </div>
        ))}
      </div>

      <Card>
        <Tabs
          tabs={[
            { key: 'wrc',      label: 'Water Retention' },
            { key: 'hcf',      label: 'K(ψ)' },
            { key: 'capacity', label: 'C(ψ)' },
            { key: 'all',      label: 'All curves' },
          ]}
          active={tab} onChange={setTab}
        />

        {tab === 'wrc' && (
          <Chart
            data={[
              { x: absPsi, y: theta, type: 'scatter', mode: 'lines', name: 'θ (vol. water content)',
                line: { color: '#2f81f7', width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(47,129,247,.06)' },
              { x: absPsi, y: Se, type: 'scatter', mode: 'lines', name: 'Se (eff. saturation)',
                line: { color: '#a371f7', width: 2, dash: 'dash' }, yaxis: 'y2' },
            ]}
            layout={{
              xaxis: { title: { text: '|ψ| — suction head (m)', font: { size: 11 } }, type: 'log' },
              yaxis: { title: { text: 'θ (m³/m³)', font: { size: 11 } } },
              yaxis2: { title: { text: 'Se (–)', font: { size: 11 } }, overlaying: 'y', side: 'right', range: [0, 1.05],
                gridcolor: '#21262d', tickfont: { size: 10 } },
            }}
            height={460}
          />
        )}

        {tab === 'hcf' && (
          <Chart
            data={[{ x: absPsi, y: K, type: 'scatter', mode: 'lines', name: 'K(ψ)',
              line: { color: '#3fb950', width: 2.5 } }]}
            layout={{
              xaxis: { title: { text: '|ψ| (m)', font: { size: 11 } }, type: 'log' },
              yaxis: { title: { text: 'K (m/s)', font: { size: 11 } }, type: 'log' },
            }}
            height={460}
          />
        )}

        {tab === 'capacity' && (
          <Chart
            data={[{ x: absPsi, y: C, type: 'scatter', mode: 'lines', name: 'C(ψ)',
              line: { color: '#d29922', width: 2.5 } }]}
            layout={{
              xaxis: { title: { text: '|ψ| (m)', font: { size: 11 } }, type: 'log' },
              yaxis: { title: { text: 'C = dθ/dψ (1/m)', font: { size: 11 } }, type: 'log' },
            }}
            height={460}
          />
        )}

        {tab === 'all' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {[
              { y: theta, color: '#2f81f7', label: 'θ',     yLabel: 'θ (m³/m³)',    logY: false },
              { y: Se,    color: '#a371f7', label: 'Se',    yLabel: 'Se',            logY: false },
              { y: K,     color: '#3fb950', label: 'K(ψ)',  yLabel: 'K (m/s)',       logY: true  },
              { y: C,     color: '#d29922', label: 'C(ψ)',  yLabel: 'C (1/m)',       logY: true  },
            ].map(({ y, color, label, yLabel, logY }) => (
              <Chart key={label}
                data={[{ x: absPsi, y, type: 'scatter', mode: 'lines',
                  line: { color, width: 2 }, name: label }]}
                layout={{
                  xaxis: { title: { text: '|ψ| (m)', font: { size: 10 } }, type: 'log' },
                  yaxis: { title: { text: yLabel, font: { size: 10 } }, type: logY ? 'log' : 'linear' },
                  margin: { l: 55, r: 10, t: 20, b: 45 },
                }}
                height={260}
              />
            ))}
          </div>
        )}
      </Card>

      <Card>
        <SectionTitle>Van Genuchten equations</SectionTitle>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontFamily: 'var(--font-mono)',
          fontSize: 11, color: 'var(--muted)' }}>
          {[
            ['Water Retention',      'θ(ψ) = θr + (θs − θr) · [1 + (α|ψ|)ⁿ]^(−m),   m = 1 − 1/n'],
            ['Hydraulic Cond.',      'K(Se) = Ks · Se^l · [1 − (1 − Se^(1/m))^m]²'],
            ['Moisture Capacity',    'C(ψ) = dθ/dψ = α·m·n·(θs−θr)·(α|ψ|)^(n−1) · [1+(α|ψ|)ⁿ]^(−m−1)'],
          ].map(([name, eq]) => (
            <div key={name} style={{ padding: '8px 12px', background: 'rgba(255,255,255,.02)',
              borderRadius: 6, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 3 }}>{name}</div>
              <div style={{ color: 'var(--text)' }}>{eq}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
