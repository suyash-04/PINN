import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SectionTitle, SliderField, StatusBadge } from '../components/ui';
import Chart, { DANGER } from '../components/Chart';

export default function CriticalSlip() {
  const { geo, norm, defaults } = useApp();
  const tMax = defaults.norm.t_max;
  const zMax = norm.z_max;

  const [time, setTime] = useState(Math.round(tMax * 0.78));
  const [frame, setFrame] = useState(null);
  const [evolution, setEvolution] = useState(null);

  const depths100 = Array.from({ length: 100 }, (_, i) => 0.5 + (i / 99) * (zMax - 0.5));

  // Per-time frame
  useEffect(() => {
    setFrame(null);
    Promise.all([
      api.getCriticalSlip({ times: [time], geo, norm, z_res: 150 }),
      api.factorOfSafety({ z: depths100, t: Array(100).fill(time), geo }),
      api.predict({ z: depths100, t: Array(100).fill(time) }),
    ]).then(([crit, fsRes, psiRes]) => {
      const critDepth = crit.critical_depths[0];
      const minFs     = crit.min_fs[0];
      const critIdx   = depths100.reduce((best, d, i) =>
        Math.abs(d - critDepth) < Math.abs(depths100[best] - critDepth) ? i : best, 0);
      setFrame({
        critDepth, minFs,
        psiAtCrit: psiRes.psi[critIdx],
        fsProfile:  fsRes.fs,
        psiProfile: psiRes.psi,
      });
    });
  }, [time, geo, norm]);

  // Full evolution across all available timesteps
  useEffect(() => {
    setEvolution(null);
    api.getCriticalSlip({ times: defaults.available_times, geo, norm, z_res: 80 })
      .then(r => setEvolution({ times: r.times, depths: r.critical_depths, fs: r.min_fs }));
  }, [geo, norm, defaults.available_times]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1200 }}>
      <PageHeader
        title="Critical Slip Surface"
        subtitle="Most failure-prone depth and minimum FS over simulation time"
        badge="Slip"
      />

      <Card style={{ maxWidth: 460 }}>
        <SliderField label="Analysis time" value={time} onChange={setTime}
          min={0} max={tMax} step={1} fmt={v => `Day ${v.toFixed(0)}`} />
      </Card>

      {!frame
        ? <Spinner text="Locating critical slip surface…" />
        : <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
              <MetricCard label="Critical depth" value={`${frame.critDepth.toFixed(2)} m`} color="red" />
              <MetricCard label="Min FS"
                value={frame.minFs.toFixed(4)}
                sub={<StatusBadge fs={frame.minFs} />}
                color={frame.minFs >= 1.5 ? 'green' : frame.minFs >= 1 ? 'amber' : 'red'} />
              <MetricCard label="ψ at critical depth" value={`${frame.psiAtCrit.toFixed(2)} m`} color="blue" />
              <MetricCard label="Day" value={time} color="muted" />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <Card>
                <SectionTitle>FS profile — critical depth marked</SectionTitle>
                <Chart
                  data={[
                    { x: frame.fsProfile, y: depths100, type: 'scatter', mode: 'lines',
                      name: 'FS', line: { color: '#2f81f7', width: 2.5 } },
                    { x: [1, 1], y: [0, zMax], type: 'scatter', mode: 'lines',
                      name: 'FS = 1', line: { color: DANGER, dash: 'dash', width: 1.5 } },
                    { x: [frame.minFs], y: [frame.critDepth], type: 'scatter', mode: 'markers',
                      name: 'Critical', marker: { color: DANGER, size: 12, symbol: 'star' } },
                  ]}
                  layout={{
                    xaxis: { title: { text: 'Factor of Safety', font: { size: 11 } } },
                    yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                    shapes: [{
                      type: 'line', y0: frame.critDepth, y1: frame.critDepth,
                      x0: 0, x1: 1, xref: 'paper',
                      line: { color: DANGER, dash: 'dot', width: 1 },
                    }],
                    legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
                  }}
                  height={400}
                />
              </Card>

              <Card>
                <SectionTitle>ψ profile — critical depth marked</SectionTitle>
                <Chart
                  data={[
                    { x: frame.psiProfile, y: depths100, type: 'scatter', mode: 'lines',
                      name: 'ψ', line: { color: '#3fb950', width: 2.5 },
                      fill: 'tozerox', fillcolor: 'rgba(63,185,80,.06)' },
                    { x: [frame.psiAtCrit], y: [frame.critDepth], type: 'scatter', mode: 'markers',
                      name: 'Critical', marker: { color: DANGER, size: 12, symbol: 'star' } },
                  ]}
                  layout={{
                    xaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
                    yaxis: { title: { text: 'Depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                    shapes: [{
                      type: 'line', y0: frame.critDepth, y1: frame.critDepth,
                      x0: 0, x1: 1, xref: 'paper',
                      line: { color: DANGER, dash: 'dot', width: 1 },
                    }],
                  }}
                  height={400}
                />
              </Card>
            </div>
          </>
      }

      {/* Temporal evolution */}
      <Card>
        <SectionTitle>Critical depth & min FS — full temporal evolution</SectionTitle>
        {!evolution
          ? <Spinner text="Loading evolution…" />
          : (
            <Chart
              data={[
                { x: evolution.times, y: evolution.depths, type: 'scatter', mode: 'lines+markers',
                  name: 'Critical depth',
                  line: { color: DANGER, width: 2 }, marker: { size: 5 } },
                { x: evolution.times, y: evolution.fs, type: 'scatter', mode: 'lines+markers',
                  name: 'Min FS',
                  line: { color: '#2f81f7', width: 2, dash: 'dash' },
                  marker: { size: 5 }, yaxis: 'y2' },
              ]}
              layout={{
                xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                yaxis: { title: { text: 'Critical depth (m)', font: { size: 11 } }, autorange: 'reversed' },
                yaxis2: { title: { text: 'Min FS', font: { size: 11 } },
                  overlaying: 'y', side: 'right',
                  gridcolor: '#21262d', tickfont: { size: 10 } },
                legend: { orientation: 'h', y: 1.1, xanchor: 'center', x: 0.5 },
              }}
              height={380}
            />
          )
        }
      </Card>
    </div>
  );
}
