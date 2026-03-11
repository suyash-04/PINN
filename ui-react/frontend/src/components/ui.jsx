/* Shared lightweight components */

export function Card({ children, className = '' }) {
  return (
    <div className={`bg-white rounded-xl border border-slate-200 shadow-sm p-5 ${className}`}>
      {children}
    </div>
  );
}

export function MetricCard({ label, value, sub, color = 'blue' }) {
  const ring = { blue: 'border-l-blue-500', green: 'border-l-green-500', red: 'border-l-red-500', amber: 'border-l-amber-500', purple: 'border-l-purple-500' };
  return (
    <div className={`bg-white rounded-xl border border-slate-200 shadow-sm p-4 border-l-4 ${ring[color] || ring.blue}`}>
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">{label}</p>
      <p className="text-2xl font-bold text-slate-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export function PageHeader({ title, subtitle, icon }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
        {icon && <span className="text-2xl">{icon}</span>}
        {title}
      </h1>
      {subtitle && <p className="text-slate-500 mt-1">{subtitle}</p>}
    </div>
  );
}

export function Spinner({ text = 'Loading…' }) {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="text-center space-y-3">
        <div className="animate-spin h-8 w-8 border-3 border-blue-600 border-t-transparent rounded-full mx-auto" />
        <p className="text-sm text-slate-500">{text}</p>
      </div>
    </div>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="flex border-b border-slate-200 mb-4 gap-1 overflow-x-auto">
      {tabs.map((t) => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className={`px-4 py-2 text-sm font-medium whitespace-nowrap rounded-t-lg transition-colors
            ${active === t.key
              ? 'bg-blue-600 text-white'
              : 'text-slate-600 hover:bg-slate-100'}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function SliderField({ label, value, onChange, min, max, step = 1, unit = '' }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="font-medium text-slate-600">{label}</span>
        <span className="text-slate-900 font-semibold">{typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full bg-slate-200 accent-blue-600 cursor-pointer" />
    </div>
  );
}
