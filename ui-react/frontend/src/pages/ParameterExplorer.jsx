import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SectionTitle, SliderField, StatusBadge, Button } from '../components/ui';
import Chart, { COLORS, DANGER } from '../components/Chart';

export default function ParameterExplorer() {
  const { geo, defaults } = useApp();
  const tMax = defaults.norm.t_max;
  const zMax = defaults.norm.z_max;

  const [time, setTime]     = useState(tMax);
  const [profiles, setProfiles] = useState(null);
  const [sensitivity, setSensitivity] = useState(null);

  // Point query — React state, no DOM access
  const [qDepth, setQDepth] = useState(5);
  const [qTime, setQTime]   = useState(tMax);
  const [qResult, setQResult] = useState(null);

  const depths = Array.from({ length: 200 }, (_, i) => 0.5 + i * ((zMax - 0.5) / 199));

  useEffect(() => {
    setProfiles(null);
    Promise.all([
      api.factorOfSafety({ z: depths, t: Array(200).fill(time), geo }),
      api.factorOfSafety({ z: depths, t: Array(200).fill(time), geo: defaults.geo }),
      api.predict({ z: depths, t: Array(200).fill(time) }),
    ]).then(([fsCur, fsDef, psi]) => {
      setProfiles({ depths, fsCurrent: fsCur.fs, fsDefault: fsDef.fs, psi: psi.psi });
    });
    api.sensitivity({ z: 10, t: time, geo }).then(setSensitivity);
  }, [geo, time]);

  const runQuery = useCallback(() => {
    Promise.all([
      api.predict({ z: [qDepth], t: [qTime] }),
      api.factorOfSafety({ z: [qDepth], t: [qTime], geo }),
    ]).then(([psiRes, fsRes]) => {
      setQResult({ depth: qDepth, time: qTime, psi: psiRes.psi[0], fs: fsRes.fs[0] });
    });
  }, [qDepth, qTime, geo]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="Parameter Explorer"
        subtitle="Live effect of geotechnical parameters on pressure head and stability"
        badge="Interactive"
      />

      {/* Time slider */}
      <Card style={{ maxWidth: 500 }}>
        <SliderField label="Analysis time" value={time} onChange={setTime}
          min={0} max={tMax} step={1} fmt={v => `Day ${v.toFixed(0)}`} />
      </Card>

      {/* Dual profile */}
      {!profiles
        ? <Spinner text="Computing profiles…" />
        : (
          <Card>
            <SectionTitle>ψ and FS depth profiles — current vs default parameters</SectionTitle>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Chart
                data={[{ x: profiles.psi, y: profiles.depths, type: 'scatter', mode: 'lines',
                  name: 'ψ (PINN)', line: { color: '#2f81f7', width: 2 } }]}
                layout={{
                  xaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                }}
                height={400}
              />
              <Chart
                data={[
                  { x: profiles.fsCurrent, y: profiles.depths, type: 'scatter', mode: 'lines',
                    name: 'Current', line: { color: '#3fb950', width: 2 } },
                  { x: profiles.fsDefault, y: profiles.depths, type: 'scatter', mode: 'lines',
                    name: 'Default', line: { color: '#7d8590', width: 1.5, dash: 'dot' } },
                  { x: [1, 1], y: [0, zMax], type: 'scatter', mode: 'lines', name: 'FS = 1',
                    line: { color: DANGER, dash: 'dash', width: 1.5 }, showlegend: true },
                ]}
                layout={{
                  xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } }, range: [0, 10] },
                  yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                  legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
                }}
                height={400}
              />
            </div>
          </Card>
        )
      }

      {/* Sensitivity tornado */}
      {sensitivity && (
        <Card>
          <SectionTitle>Sensitivity — ΔFS at z = 10 m, FS₀ = {sensitivity.fs_base.toFixed(3)}</SectionTitle>
          <Chart
            data={[
              { y: sensitivity.sensitivity.map(s => s.param),
                x: sensitivity.sensitivity.map(s => s.fs_low),
                type: 'bar', orientation: 'h', name: '↓ param',
                marker: { color: '#f85149' } },
              { y: sensitivity.sensitivity.map(s => s.param),
                x: sensitivity.sensitivity.map(s => s.fs_high),
                type: 'bar', orientation: 'h', name: '↑ param',
                marker: { color: '#3fb950' } },
            ]}
            layout={{
              barmode: 'relative',
              xaxis: { title: { text: 'ΔFS', font: { size: 11 } },
                zeroline: true, zerolinecolor: '#7d8590', zerolinewidth: 1 },
              margin: { l: 110, r: 20, t: 20, b: 40 },
              shapes: [{ type: 'line', x0: 0, x1: 0, y0: -0.5, y1: 5.5,
                line: { color: '#7d8590', width: 1 } }],
            }}
            height={280}
          />
        </Card>
      )}

      {/* Point query — pure React state */}
      <Card>
        <SectionTitle>Point query</SectionTitle>
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 16 }}>
          <div style={{ flex: 1, minWidth: 180 }}>
            <SliderField label="Depth z" value={qDepth} onChange={setQDepth}
              min={0.5} max={zMax} step={0.5} fmt={v => `${v.toFixed(1)} m`} />
          </div>
          <div style={{ flex: 1, minWidth: 180 }}>
            <SliderField label="Time t" value={qTime} onChange={setQTime}
              min={0} max={tMax} step={1} fmt={v => `Day ${v.toFixed(0)}`} />
          </div>
          <Button onClick={runQuery}>Query →</Button>
        </div>
        {qResult && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            <MetricCard label="Depth" value={`${qResult.depth.toFixed(1)} m`} color="muted" />
            <MetricCard label="Time" value={`Day ${qResult.time}`} color="muted" />
            <MetricCard label="ψ" value={`${qResult.psi.toFixed(2)} m`} color="amber" />
            <MetricCard label="FS"
              value={qResult.fs.toFixed(4)}
              sub={<StatusBadge fs={qResult.fs} />}
              color={qResult.fs >= 1.5 ? 'green' : qResult.fs >= 1 ? 'amber' : 'red'} />
          </div>
        )}
      </Card>
    </div>
  );
}
