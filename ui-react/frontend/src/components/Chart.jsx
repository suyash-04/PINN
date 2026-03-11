import Plot from 'react-plotly.js';

const LAYOUT_DEFAULTS = {
  template: 'plotly_white',
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#f8fafc',
  font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#334155' },
  margin: { l: 55, r: 20, t: 35, b: 50 },
  hoverlabel: { bgcolor: '#1e293b', font: { color: '#fff', size: 12 } },
};

export default function Chart({ data, layout = {}, config, style, className = '' }) {
  return (
    <Plot
      data={data}
      layout={{ ...LAYOUT_DEFAULTS, ...layout }}
      config={{ responsive: true, displayModeBar: true, displaylogo: false, ...config }}
      useResizeHandler
      style={{ width: '100%', ...style }}
      className={className}
    />
  );
}

export const COLORS = [
  '#2563eb', '#7c3aed', '#db2777', '#f97316', '#16a34a',
  '#dc2626', '#ca8a04', '#0d9488', '#6366f1', '#e11d48',
];
