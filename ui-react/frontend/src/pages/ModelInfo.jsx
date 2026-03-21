import { useState, useEffect } from 'react';
import { api } from '../api';
import { PageHeader, Spinner, Card, SectionTitle } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  useEffect(() => { api.getModelInfo().then(setInfo); }, []);
  if (!info) return <Spinner text="Loading model metadata…" />;

  const arch = info.arch ?? {};
  const norm = info.norm ?? {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1100 }}>
      <PageHeader
        title="Model Information"
        subtitle="PINN architecture, normalisation, training strategy, and weight diagnostics"
        badge="Introspection"
      />

      {/* Summary strip */}
      <div style={{ padding: '12px 20px', borderRadius: 8,
        border: '1px solid rgba(47,129,247,.25)',
        background: 'rgba(47,129,247,.06)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
        <span style={{ color: 'var(--accent)' }}>
          {arch.n_hidden_layers} × {arch.neurons_per_layer}
        </span>
        <span style={{ color: 'var(--muted)' }}> hidden layers  ·  </span>
        <span style={{ color: 'var(--accent)' }}>{arch.activation}</span>
        <span style={{ color: 'var(--muted)' }}> activation  ·  </span>
        <span style={{ color: 'var(--accent)' }}>{info.total_params?.toLocaleString()}</span>
        <span style={{ color: 'var(--muted)' }}> total parameters  ·  </span>
        <span style={{ color: 'var(--accent)' }}>{info.size_kb} kB</span>
        <span style={{ color: 'var(--muted)' }}> model size</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Architecture */}
        <Card>
          <SectionTitle>Architecture</SectionTitle>
          <InfoTable rows={[
            ['Input neurons',  arch.input_dim],
            ['Hidden layers',  arch.n_hidden_layers],
            ['Neurons / layer',arch.neurons_per_layer],
            ['Activation',     arch.activation],
            ['Output neurons', arch.output_dim],
            ['Weight init',    'Xavier normal'],
            ['Bias init',      'zeros'],
            ['Total params',   info.total_params?.toLocaleString()],
            ['Model size',     `${info.size_kb} kB`],
          ]} />
        </Card>

        {/* Normalisation */}
        <Card>
          <SectionTitle>Input normalisation</SectionTitle>
          <InfoTable rows={[
            ['t_max',   `${norm.t_max} days`],
            ['z_max',   `${norm.z_max} m`],
            ['ψ_min',   `${norm.psi_min} m`],
            ['ψ_max',   `${norm.psi_max} m`],
            ['ψ range', `${(norm.psi_max - norm.psi_min).toFixed(2)} m`],
            ['All inputs → [0, 1]', 'min-max scaling'],
          ]} />

          <div style={{ marginTop: 16 }}>
            <SectionTitle>Training strategy</SectionTitle>
            <InfoTable rows={[
              ['Phase 1', 'Adam  lr=5×10⁻⁴  10 000 epochs'],
              ['Phase 2', 'L-BFGS  50 outer loops × 100 iter'],
              ['Early stop', 'Δloss < 10⁻⁶  for 5 steps'],
              ['LR decay',   'StepLR  ×0.5  every 2 000 epochs'],
            ]} />
          </div>
        </Card>
      </div>

      {/* Network topology visual */}
      <Card>
        <SectionTitle>Network topology</SectionTitle>
        <div style={{ overflowX: 'auto', paddingBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 'max-content',
            padding: '12px 4px' }}>
            <LayerBlock label="Input" neurons={arch.input_dim ?? 2} sub="(t̃, z̃)"
              color="var(--accent)" />
            {Array.from({ length: arch.n_hidden_layers ?? 7 }).map((_, i) => (
              <Arrow key={i} />
            )).reduce((acc, arrow, i) => [
              ...acc,
              arrow,
              <LayerBlock key={`h${i}`}
                label={`H${i + 1}`}
                neurons={arch.neurons_per_layer ?? 64}
                sub={arch.activation ?? 'tanh'}
                color="var(--purple)"
              />,
            ], [])}
            <Arrow />
            <LayerBlock label="Output" neurons={arch.output_dim ?? 1} sub="ψ̃(t̃,z̃)"
              color="var(--accent-2)" />
          </div>
        </div>
      </Card>

      {/* Weight histograms */}
      {info.weight_histograms && Object.keys(info.weight_histograms).length > 0 && (
        <Card>
          <SectionTitle>Weight distributions (first · mid · last layer)</SectionTitle>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12 }}>
            {Object.entries(info.weight_histograms).map(([name, vals]) => (
              <div key={name}>
                <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 6,
                  fontFamily: 'var(--font-mono)' }}>
                  {name}
                </div>
                <Chart
                  data={[{ x: vals, type: 'histogram', nbinsx: 40,
                    marker: { color: 'rgba(47,129,247,.5)' }, name: '' }]}
                  layout={{
                    xaxis: { title: { text: 'weight', font: { size: 9 } } },
                    yaxis: { title: { text: 'count',  font: { size: 9 } } },
                    bargap: 0.03,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    showlegend: false,
                  }}
                  height={180}
                />
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Layer table */}
      {info.layers && (
        <Card>
          <SectionTitle>Layer parameter table</SectionTitle>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse',
              fontSize: 11, fontFamily: 'var(--font-mono)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Layer name', 'Shape', 'Params', 'Trainable'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '5px 12px',
                      color: 'var(--muted)', fontSize: 10, fontWeight: 600,
                      letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {info.layers.map((l, i) => (
                  <tr key={l.name} style={{ borderBottom: '1px solid var(--border)',
                    background: i % 2 ? 'rgba(255,255,255,.015)' : 'transparent' }}>
                    <td style={{ padding: '5px 12px', color: 'var(--text)' }}>{l.name}</td>
                    <td style={{ padding: '5px 12px', color: 'var(--muted)' }}>
                      {l.shape.join(' × ')}
                    </td>
                    <td style={{ padding: '5px 12px', color: 'var(--text)' }}>
                      {l.n_params.toLocaleString()}
                    </td>
                    <td style={{ padding: '5px 12px',
                      color: l.requires_grad ? 'var(--accent-2)' : 'var(--muted)' }}>
                      {l.requires_grad ? '✓' : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function InfoTable({ rows }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {rows.map(([k, v], i) => (
        <div key={k} style={{ display: 'flex', justifyContent: 'space-between',
          padding: '5px 0', borderBottom: '1px solid var(--border)',
          fontSize: 11 }}>
          <span style={{ color: 'var(--muted)' }}>{k}</span>
          <span style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)',
            fontWeight: 500 }}>{String(v ?? '—')}</span>
        </div>
      ))}
    </div>
  );
}

function LayerBlock({ label, neurons, sub, color }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <div style={{ width: 52, height: 52, borderRadius: 8, border: `1.5px solid ${color}`,
        background: `${color}18`, display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexDirection: 'column', gap: 0 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700,
          color }}>{neurons}</span>
        <span style={{ fontSize: 8, color: 'var(--muted)' }}>{sub}</span>
      </div>
      <span style={{ fontSize: 9, color: 'var(--muted)' }}>{label}</span>
    </div>
  );
}

function Arrow() {
  return (
    <div style={{ color: 'var(--border2)', fontSize: 16, lineHeight: 1,
      marginBottom: 18, flexShrink: 0 }}>→</div>
  );
}
