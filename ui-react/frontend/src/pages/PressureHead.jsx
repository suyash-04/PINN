import { useState, useEffect } from 'react';
import { api } from '../api';
import { useApp } from '../context';
import { PageHeader, Spinner, Card, TimeToggle, SectionTitle, SliderField } from '../components/ui';
import Chart, { COLORS } from '../components/Chart';

export default function PressureHead() {
  const { defaults } = useApp();
  const tMax = defaults.norm.t_max;
  const zMax = defaults.norm.z_max;

  const defaultTimes = defaults.available_times.slice(0, 4);
  const [selectedTimes, setSelectedTimes] = useState(defaultTimes);
  const [depth, setDepth] = useState(5);
  const [profiles, setProfiles] = useState(null);
  const [timeSeries, setTimeSeries] = useState(null);

  const depths200 = Array.from({ length: 200 }, (_, i) => 0.5 + i * (zMax / 200));

  useEffect(() => {
    if (!selectedTimes.length) { setProfiles(null); return; }
    Promise.all(
      selectedTimes.map(t => api.predict({ z: depths200, t: Array(200).fill(t) }))
    ).then(res => setProfiles(res));
  }, [selectedTimes]);

  useEffect(() => {
    const times = Array.from({ length: 200 }, (_, i) => i * (tMax / 199));
    api.predict({ z: Array(200).fill(depth), t: times }).then(d =>
      setTimeSeries({ times, psi: d.psi })
    );
  }, [depth, tMax]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1400 }}>
      <PageHeader
        title="Pressure Head  ψ(z, t)"
        subtitle="PINN-predicted pore-water suction profiles across depth and time"
        badge="Hydraulics"
      />

      <Card>
        <SectionTitle>Select timesteps</SectionTitle>
        <TimeToggle
          times={defaults.available_times}
          selected={selectedTimes}
          onChange={setSelectedTimes}
          label="Day"
        />
      </Card>

      {!profiles
        ? <Spinner text="Fetching profiles…" />
        : (
          <Card>
            <SectionTitle>ψ vs Depth — selected timesteps</SectionTitle>
            <Chart
              data={profiles.map((p, i) => ({
                x: p.psi, y: depths200, type: 'scatter', mode: 'lines',
                name: `Day ${selectedTimes[i]}`,
                line: { color: COLORS[i % COLORS.length], width: 2 },
              }))}
              layout={{
                xaxis: { title: { text: 'Pressure Head ψ (m)', font: { size: 11 } } },
                yaxis: { title: { text: 'Depth z (m)', font: { size: 11 } }, autorange: 'reversed' },
                legend: { orientation: 'h', y: 1.12, xanchor: 'center', x: 0.5 },
              }}
              height={480}
            />
          </Card>
        )
      }

      <Card>
        <SectionTitle>ψ vs Time — at fixed depth</SectionTitle>
        <div style={{ marginBottom: 16, maxWidth: 380 }}>
          <SliderField
            label="Depth z"
            value={depth}
            onChange={setDepth}
            min={0.5} max={zMax} step={0.5}
            fmt={v => `${v.toFixed(1)} m`}
          />
        </div>
        {!timeSeries
          ? <Spinner text="Loading time series…" />
          : (
            <Chart
              data={[{
                x: timeSeries.times, y: timeSeries.psi, type: 'scatter', mode: 'lines',
                name: `ψ at z = ${depth} m`,
                line: { color: '#2f81f7', width: 2 },
                fill: 'tozeroy', fillcolor: 'rgba(47,129,247,.06)',
              }]}
              layout={{
                xaxis: { title: { text: 'Time (days)', font: { size: 11 } } },
                yaxis: { title: { text: 'ψ (m)', font: { size: 11 } } },
              }}
              height={340}
            />
          )
        }
      </Card>
    </div>
  );
}
