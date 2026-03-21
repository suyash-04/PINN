import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SectionTitle, SliderField } from '../components/ui';
import Chart, { COLORS, DANGER } from '../components/Chart';

export default function FactorOfSafety() {
  const { geo, defaults } = useApp();
  const tMax = defaults.norm.t_max;
  const zMax = defaults.norm.z_max;

  const [zRes, setZRes]   = useState(50);
  const [tRes, setTRes]   = useState(50);
  const [fsClip, setFsClip] = useState(5);
  const [data, setData]   = useState(null);
  const [history, setHistory] = useState(null);
  const [selDepths, setSelDepths] = useState([1, 2, 5, 10, 20]);

  useEffect(() => {
    setData(null);
    api.fsGrid({ z_min: 0.5, z_max: zMax, z_res: zRes, t_res: tRes, geo }).then(setData);
  }, [geo, zRes, tRes, zMax]);

  useEffect(() => {
    if (!selDepths.length) return;
    setHistory(null);
    const times = Array.from({ length: 200 }, (_, i) => i * (tMax / 199));
    Promise.all(
      selDepths.map(d =>
        api.factorOfSafety({ z: Array(200).fill(d), t: times, geo })
          .then(r => ({ depth: d, times, fs: r.fs }))
      )
    ).then(setHistory);
  }, [selDepths, geo, tMax]);

  const depthBtns = [0.5, 1, 2, 5, 10, 15, 20, 30, 40];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="Factor of Safety  FS(z, t)"
        subtitle="2D stability map across depth and time — infinite slope model"
        badge="Stability"
      />

      {/* Controls */}
      <Card>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
          <SliderField label="Depth resolution"  value={zRes}   onChange={setZRes}   min={20} max={100} step={5} />
          <SliderField label="Time resolution"   value={tRes}   onChange={setTRes}   min={20} max={100} step={5} />
          <SliderField label="FS color-scale max" value={fsClip} onChange={setFsClip} min={1.5} max={10} step={0.5} />
        </div>
      </Card>

      {!data
        ? <Spinner text="Computing FS grid…" />
        : <>
            {/* Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
              <MetricCard label="Grid cells" value={data.stats.total.toLocaleString()} color="muted" />
              <MetricCard label="Unstable  FS < 1"
                value={`${data.stats.unstable.toLocaleString()} (${(100*data.stats.unstable/data.stats.total).toFixed(1)}%)`}
                color="red" />
              <MetricCard label="Marginal  1 ≤ FS < 1.5"
                value={`${data.stats.marginal.toLocaleString()} (${(100*data.stats.marginal/data.stats.total).toFixed(1)}%)`}
                color="amber" />
              <MetricCard label="Stable  FS ≥ 1.5"
                value={`${data.stats.safe.toLocaleString()} (${(100*data.stats.safe/data.stats.total).toFixed(1)}%)`}
                color="green" />
            </div>

            {/* Heatmap */}
            <Card>
              <SectionTitle>FS heatmap — depth × time</SectionTitle>
              <Chart
                data={[{
                  x: data.t, y: data.z, z: data.fs, type: 'heatmap',
                  colorscale: [
                    [0,   '#7f1d1d'],
                    [1/fsClip * 0.8,  '#f85149'],
                    [1/fsClip,        '#d29922'],
                    [1.5/fsClip,      '#3fb950'],
                    [1,               '#0ea5e9'],
                  ],
                  colorbar: { title: { text: 'FS', side: 'right' }, thickness: 14,
                    tickfont: { size: 10, color: '#7d8590' },
                    tickcolor: '#30363d', outlinecolor: '#21262d' },
                  hovertemplate: 'Day: %{x:.1f}<br>Depth: %{y:.1f} m<br>FS: %{z:.3f}<extra></extra>',
                  zmin: 0, zmax: fsClip,
                }]}
                layout={{
                  xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                }}
                height={500}
              />
            </Card>
          </>
      }

      {/* Time history */}
      <Card>
        <SectionTitle>FS time history at selected depths</SectionTitle>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
          {depthBtns.map(d => (
            <button key={d} onClick={() =>
              setSelDepths(prev =>
                prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d]
              )}
              style={{
                padding: '3px 10px', fontSize: 11, cursor: 'pointer',
                borderRadius: 4, border: selDepths.includes(d)
                  ? '1px solid var(--accent)' : '1px solid var(--border)',
                background: selDepths.includes(d) ? 'rgba(47,129,247,.15)' : 'transparent',
                color: selDepths.includes(d) ? 'var(--accent)' : 'var(--muted)',
                fontFamily: 'var(--font-mono)',
              }}>
              {d} m
            </button>
          ))}
        </div>

        {!history
          ? <Spinner text="Computing histories…" />
          : (
            <Chart
              data={[
                ...history.map((h, i) => ({
                  x: h.times, y: h.fs, type: 'scatter', mode: 'lines',
                  name: `z = ${h.depth} m`,
                  line: { color: COLORS[i % COLORS.length], width: 2 },
                })),
                { x: [0, tMax], y: [1, 1], type: 'scatter', mode: 'lines',
                  name: 'FS = 1', line: { color: DANGER, dash: 'dash', width: 1.5 }, showlegend: true },
              ]}
              layout={{
                xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                yaxis: { title: { text: 'Factor of Safety', font: { size: 11 } }, range: [0, Math.min(fsClip, 8)] },
                legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
              }}
              height={420}
            />
          )
        }
      </Card>
    </div>
  );
}
