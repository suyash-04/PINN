import Plot from 'react-plotly.js';

// Publication-quality dark theme
const LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: "'IBM Plex Mono', monospace", size: 11, color: '#7d8590' },
  margin: { l: 60, r: 20, t: 30, b: 50 },
  xaxis: {
    gridcolor: '#21262d', linecolor: '#30363d', zerolinecolor: '#30363d',
    tickcolor: '#30363d', tickfont: { size: 10 },
  },
  yaxis: {
    gridcolor: '#21262d', linecolor: '#30363d', zerolinecolor: '#30363d',
    tickcolor: '#30363d', tickfont: { size: 10 },
  },
  legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#21262d', borderwidth: 1,
    font: { size: 10, color: '#7d8590' } },
  hoverlabel: { bgcolor: '#161b22', bordercolor: '#30363d',
    font: { color: '#e6edf3', size: 11, family: "'IBM Plex Mono', monospace" } },
};

function mergeLayouts(base, override) {
  const merged = { ...base, ...override };
  if (override.xaxis) merged.xaxis = { ...base.xaxis, ...override.xaxis };
  if (override.yaxis) merged.yaxis = { ...base.yaxis, ...override.yaxis };
  if (override.yaxis2) merged.yaxis2 = {
    gridcolor: '#21262d', linecolor: '#30363d', tickfont: { size: 10 },
    ...override.yaxis2,
  };
  return merged;
}

export default function Chart({ data, layout = {}, config, height = 420 }) {
  const finalLayout = mergeLayouts(LAYOUT_BASE, {
    height,
    ...layout,
  });
  return (
    <Plot
      data={data}
      layout={finalLayout}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        ...config,
      }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
}

// Scientific publication color palette — colorblind-safe
export const COLORS = [
  '#2f81f7', // blue
  '#3fb950', // green
  '#d29922', // amber
  '#f85149', // red
  '#a371f7', // purple
  '#79c0ff', // light blue
  '#56d364', // light green
  '#ffa657', // orange
  '#ff7b72', // light red
  '#c9a0ff', // light purple
];

export const ACCENT = '#2f81f7';
export const DANGER = '#f85149';
export const SUCCESS = '#3fb950';
export const WARN = '#d29922';
