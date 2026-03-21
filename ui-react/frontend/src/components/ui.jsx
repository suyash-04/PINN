/* ──────────────────────────────────────────────────────────
   Design system — dark scientific theme
   ────────────────────────────────────────────────────────── */

// ── Card ──────────────────────────────────────────────────
export function Card({ children, className = '', style = {} }) {
  return (
    <div className={className} style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '16px 20px',
      ...style,
    }}>
      {children}
    </div>
  );
}

// ── MetricCard ────────────────────────────────────────────
const ACCENT_MAP = {
  blue:   'var(--accent)',
  green:  'var(--accent-2)',
  red:    'var(--danger)',
  amber:  'var(--warn)',
  purple: 'var(--purple)',
  muted:  'var(--muted)',
};

export function MetricCard({ label, value, sub, color = 'blue' }) {
  const accent = ACCENT_MAP[color] ?? ACCENT_MAP.blue;
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderTop: `2px solid ${accent}`,
      borderRadius: 8,
      padding: '12px 16px',
    }}>
      <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.08em',
        textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)',
        fontFamily: 'var(--font-mono)', lineHeight: 1.1 }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 3 }}>{sub}</div>
      )}
    </div>
  );
}

// ── PageHeader ────────────────────────────────────────────
export function PageHeader({ title, subtitle, badge }) {
  return (
    <div style={{ marginBottom: 24, borderBottom: '1px solid var(--border)', paddingBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: 'var(--text)',
          fontFamily: 'var(--font-sans)', letterSpacing: '-0.02em' }}>
          {title}
        </h1>
        {badge && (
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.06em',
            padding: '2px 8px', borderRadius: 4, background: 'rgba(47,129,247,.15)',
            color: 'var(--accent)', border: '1px solid rgba(47,129,247,.3)',
            textTransform: 'uppercase' }}>
            {badge}
          </span>
        )}
      </div>
      {subtitle && (
        <p style={{ margin: '4px 0 0', fontSize: 12, color: 'var(--muted)' }}>{subtitle}</p>
      )}
    </div>
  );
}

// ── Spinner ───────────────────────────────────────────────
export function Spinner({ size = 24, text = '' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', padding: '40px 20px', gap: 12 }}>
      <div style={{
        width: size, height: size,
        border: '2px solid var(--border2)',
        borderTopColor: 'var(--accent)',
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }} />
      {text && <span style={{ fontSize: 11, color: 'var(--muted)' }}>{text}</span>}
    </div>
  );
}

// ── Tabs ──────────────────────────────────────────────────
export function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 2, marginBottom: 16,
      borderBottom: '1px solid var(--border)', paddingBottom: 0 }}>
      {tabs.map(t => (
        <button key={t.key} onClick={() => onChange(t.key)}
          style={{
            padding: '6px 14px', fontSize: 12, fontWeight: 500, cursor: 'pointer',
            border: 'none', borderBottom: active === t.key
              ? '2px solid var(--accent)' : '2px solid transparent',
            background: 'transparent',
            color: active === t.key ? 'var(--text)' : 'var(--muted)',
            transition: 'color .15s',
            fontFamily: 'var(--font-sans)',
            marginBottom: -1,
          }}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ── SliderField ───────────────────────────────────────────
export function SliderField({ label, value, onChange, min, max, step = 1, fmt }) {
  const display = fmt ? fmt(value)
    : (step < 1 ? value.toFixed(2) : typeof value === 'number' ? value.toFixed(0) : value);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)',
          fontFamily: 'var(--font-mono)' }}>{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer',
          height: 4, appearance: 'none' }} />
    </div>
  );
}

// ── TimeToggle (chip selector) ────────────────────────────
export function TimeToggle({ times, selected, onChange, label = 'Day' }) {
  const toggle = t => onChange(
    selected.includes(t) ? selected.filter(x => x !== t) : [...selected, t]
  );
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {times.map(t => (
        <button key={t} onClick={() => toggle(t)} style={{
          padding: '3px 10px', fontSize: 11, fontWeight: 500, cursor: 'pointer',
          borderRadius: 4, border: selected.includes(t)
            ? '1px solid var(--accent)' : '1px solid var(--border)',
          background: selected.includes(t)
            ? 'rgba(47,129,247,.15)' : 'transparent',
          color: selected.includes(t) ? 'var(--accent)' : 'var(--muted)',
          transition: 'all .12s',
          fontFamily: 'var(--font-mono)',
        }}>
          {label} {t}
        </button>
      ))}
    </div>
  );
}

// ── SectionTitle ─────────────────────────────────────────
export function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--muted)',
      letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12 }}>
      {children}
    </div>
  );
}

// ── Badge ─────────────────────────────────────────────────
export function StatusBadge({ fs }) {
  const [label, color] = fs >= 1.5 ? ['STABLE', 'var(--accent-2)']
    : fs >= 1.0 ? ['MARGINAL', 'var(--warn)']
    : ['UNSTABLE', 'var(--danger)'];
  return (
    <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
      padding: '2px 8px', borderRadius: 3,
      background: `${color}22`, color, border: `1px solid ${color}44` }}>
      {label}
    </span>
  );
}

// ── Button ────────────────────────────────────────────────
export function Button({ children, onClick, disabled, variant = 'primary', style: s = {} }) {
  const base = {
    padding: '7px 18px', fontSize: 12, fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
    border: 'none', borderRadius: 6, transition: 'opacity .12s, background .12s',
    opacity: disabled ? 0.5 : 1, fontFamily: 'var(--font-sans)',
    letterSpacing: '0.02em',
  };
  const variants = {
    primary:  { background: 'var(--accent)',    color: '#fff' },
    success:  { background: 'var(--accent-2)',  color: '#0d1117' },
    ghost:    { background: 'var(--border)',     color: 'var(--text)' },
    danger:   { background: 'var(--danger)',     color: '#fff' },
  };
  return (
    <button onClick={disabled ? undefined : onClick} style={{ ...base, ...variants[variant], ...s }}>
      {children}
    </button>
  );
}

// ── Table ─────────────────────────────────────────────────
export function DataTable({ columns, rows, keyFn }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12,
        fontFamily: 'var(--font-mono)' }}>
        <thead>
          <tr>
            {columns.map(c => (
              <th key={c.key} style={{ textAlign: 'left', padding: '6px 12px',
                color: 'var(--muted)', fontSize: 10, fontWeight: 600,
                letterSpacing: '0.08em', textTransform: 'uppercase',
                borderBottom: '1px solid var(--border)' }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={keyFn ? keyFn(row) : i} style={{
              borderBottom: '1px solid var(--border)',
              background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,.02)',
            }}>
              {columns.map(c => (
                <td key={c.key} style={{ padding: '7px 12px', color: 'var(--text)' }}>
                  {c.render ? c.render(row[c.key], row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── ErrorBoundary ────────────────────────────────────────
import { Component } from 'react';
export class ErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(e) { return { error: e }; }
  render() {
    if (this.state.error) return (
      <div style={{ padding: 24, color: 'var(--danger)', fontFamily: 'var(--font-mono)',
        fontSize: 12, background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 8, margin: 16 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>Component error</div>
        <div style={{ opacity: .7 }}>{this.state.error.message}</div>
      </div>
    );
    return this.props.children;
  }
}
