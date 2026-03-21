import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, MetricCard, SliderField } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function CriticalSlip() {
  const { geo, norm, defaults } = useApp();
  const [time, setTime] = useState(96);
  const [data, setData] = useState(null);

  useEffect(() => {
    // Get critical slip for current time + full FS profile
    Promise.all([
      api.getCriticalSlip({ times: [time], geo, norm, z_res: 100 }),
      api.factorOfSafety({
        z: Array.from({ length: 100 }, (_, i) => 0.5 + (i / 99) * 39.5),
        t: Array(100).fill(time), geo,
      }),
      api.predict({
        z: Array.from({ length: 100 }, (_, i) => 0.5 + (i / 99) * 39.5),
        t: Array(100).fill(time),
      }),
    ]).then(([crit, fsRes, psiRes]) => {
      const depths = Array.from({ length: 100 }, (_, i) => 0.5 + (i / 99) * 39.5);
      const critDepth = crit.critical_depths[0];
      const minFs = crit.min_fs[0];
      // Find psi at critical depth
      const critIdx = depths.reduce((best, d, i) => Math.abs(d - critDepth) < Math.abs(depths[best] - critDepth) ? i : best, 0);
      setData({
        critical_depth: critDepth,
        min_fs: minFs,
        psi_at_critical: psiRes.psi[critIdx],
        fs_profile: fsRes.fs,
        psi_profile: psiRes.psi,
        depths,
      });
    });
  }, [time, geo, norm]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Critical Slip Surface" subtitle="Identification of the most critical failure depth over time" icon="⚠️" />

      <Card>
        <SliderField label={`Time: Hour ${time}`} value={time} min={0} max={defaults.norm.t_max} step={1} onChange={setTime} />
      </Card>

      {!data ? <Spinner /> : (
        <>
          <div className="grid grid-cols-4 gap-3">
            <MetricCard label="Critical Depth" value={`${data.critical_depth?.toFixed(2)} m`} color="red" />
            <MetricCard label="Min FS" value={data.min_fs?.toFixed(4)} color="amber" />
            <MetricCard label="ψ at Critical" value={`${data.psi_at_critical?.toFixed(2)} m`} color="blue" />
            <MetricCard label="Stability" value={data.min_fs > 1.5 ? '🟢 Safe' : data.min_fs > 1 ? '🟡 Marginal' : '🔴 Unstable'} color="green" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* FS profile with critical depth highlighted */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">FS Profile</h3>
              <Chart
                data={[
                  { x: data.fs_profile, y: data.depths, type: 'scatter', mode: 'lines',
                    name: 'FS', line: { color: '#2563eb', width: 2.5 } },
                  { x: [1, 1], y: [0, Math.max(...data.depths)], type: 'scatter', mode: 'lines',
                    name: 'FS = 1', line: { color: '#dc2626', dash: 'dash', width: 1.5 } },
                  { x: [data.min_fs], y: [data.critical_depth], type: 'scatter', mode: 'markers',
                    name: 'Critical', marker: { color: '#dc2626', size: 14, symbol: 'star' } },
                ]}
                layout={{
                  xaxis: { title: 'Factor of Safety' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 420, shapes: [{
                    type: 'line', y0: data.critical_depth, y1: data.critical_depth, x0: 0, x1: 1, xref: 'paper',
                    line: { color: '#dc2626', dash: 'dot', width: 1 },
                  }],
                }}
              />
            </Card>

            {/* ψ profile with critical depth */}
            <Card>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">ψ Profile</h3>
              <Chart
                data={[
                  { x: data.psi_profile, y: data.depths, type: 'scatter', mode: 'lines',
                    name: 'ψ', line: { color: '#059669', width: 2.5 }, fill: 'tozerox',
                    fillcolor: 'rgba(5,150,105,0.08)' },
                  { x: [data.psi_at_critical], y: [data.critical_depth], type: 'scatter', mode: 'markers',
                    name: 'Critical', marker: { color: '#dc2626', size: 14, symbol: 'star' } },
                ]}
                layout={{
                  xaxis: { title: 'ψ (m)' }, yaxis: { title: 'Depth (m)', autorange: 'reversed' },
                  height: 420,
                }}
              />
            </Card>
          </div>

          {/* Critical depth evolution over time */}
          <Card>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Critical Depth & Min FS Evolution</h3>
            <CriticalEvolution geo={geo} norm={norm} times={defaults.available_times} />
          </Card>
        </>
      )}
    </div>
  );
}

function CriticalEvolution({ geo, norm, times }) {
  const [evo, setEvo] = useState(null);

  useEffect(() => {
    api.getCriticalSlip({ times, geo, norm, z_res: 50 }).then(results => {
      setEvo({
        times: results.times,
        depths: results.critical_depths,
        fs: results.min_fs,
      });
    });
  }, [geo, norm, times]);

  if (!evo) return <div className="py-8 text-center text-xs text-slate-400">Loading evolution…</div>;

  return (
    <Chart
      data={[
        { x: evo.times, y: evo.depths, type: 'scatter', mode: 'lines+markers',
          name: 'Critical Depth', line: { color: '#dc2626', width: 2.5 }, marker: { size: 6 } },
        { x: evo.times, y: evo.fs, type: 'scatter', mode: 'lines+markers',
          name: 'Min FS', line: { color: '#2563eb', width: 2.5, dash: 'dash' },
          marker: { size: 6 }, yaxis: 'y2' },
      ]}
      layout={{
        xaxis: { title: 'Time (days)' },
        yaxis: { title: 'Critical Depth (m)', autorange: 'reversed' },
        yaxis2: { title: 'Min FS', overlaying: 'y', side: 'right' },
        height: 400,
      }}
    />
  );
}
