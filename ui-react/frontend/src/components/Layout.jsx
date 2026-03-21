import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useApp } from '../context';
import { SliderField } from './ui';
import {
  LayoutDashboard, TrendingUp, ShieldAlert, Sliders, GitCompare,
  Layers, Cpu, Play, Activity, BrainCircuit, BarChart2,
  CheckSquare, Sigma, Target, CloudRain, ArrowLeftRight,
  FileDown, ChevronLeft, ChevronRight, RotateCcw, PanelLeftClose,
} from 'lucide-react';

const NAV = [
  { section: 'Core', items: [
    { to: '/',                 icon: LayoutDashboard, label: 'Overview' },
    { to: '/pressure-head',   icon: TrendingUp,      label: 'Pressure Head' },
    { to: '/factor-of-safety',icon: ShieldAlert,     label: 'Factor of Safety' },
    { to: '/parameters',      icon: Sliders,         label: 'Parameter Explorer' },
    { to: '/hydrus',          icon: GitCompare,      label: 'HYDRUS vs PINN' },
    { to: '/soil',            icon: Layers,          label: 'Soil Properties' },
    { to: '/model',           icon: Cpu,             label: 'Model Info' },
  ]},
  { section: 'Advanced', items: [
    { to: '/animation',       icon: Play,            label: 'Animation' },
    { to: '/training',        icon: Activity,        label: 'Training Loss' },
    { to: '/pde-residual',    icon: BrainCircuit,    label: 'PDE Residual' },
    { to: '/error',           icon: BarChart2,       label: 'Error Analysis' },
    { to: '/validation',      icon: CheckSquare,     label: 'Validation' },
  ]},
  { section: 'Research', items: [
    { to: '/uncertainty',     icon: Sigma,           label: 'Uncertainty (MC)' },
    { to: '/critical-slip',   icon: Target,          label: 'Critical Slip' },
    { to: '/rainfall',        icon: CloudRain,       label: 'Rainfall Sim' },
    { to: '/scenarios',       icon: ArrowLeftRight,  label: 'Scenario Compare' },
    { to: '/export',          icon: FileDown,        label: 'Export Data' },
  ]},
];

const GEO_SLIDERS = [
  { key: 'beta',      label: 'β  slope',       min: 10, max: 65,  step: 0.5, unit: '°' },
  { key: 'c_prime',   label: "c′  cohesion",   min: 0,  max: 50,  step: 0.5, unit: ' kPa' },
  { key: 'phi_prime', label: "φ′  friction",   min: 10, max: 45,  step: 0.5, unit: '°' },
  { key: 'gamma',     label: 'γ  unit weight', min: 14, max: 26,  step: 0.1, unit: ' kN/m³' },
  { key: 'alpha',     label: 'α  VG alpha',    min: 0.1,max: 5,   step: 0.1, unit: ' 1/m' },
  { key: 'n',         label: 'n  VG pore idx', min: 1.05,max: 3,  step: 0.05 },
];

function Sidebar({ collapsed }) {
  const { geo, updateGeo, resetGeo } = useApp();
  const [geoOpen, setGeoOpen] = useState(false);

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100%',
      fontFamily:'var(--font-sans)' }}>

      {/* Brand */}
      <div style={{ padding: collapsed ? '18px 0' : '14px 16px',
        borderBottom: '1px solid var(--border)',
        textAlign: collapsed ? 'center' : 'left' }}>
        {collapsed
          ? <span style={{ fontSize:18 }}>⛰</span>
          : <>
              <div style={{ fontSize:13, fontWeight:700, color:'var(--text)',
                letterSpacing:'-0.01em' }}>PINN Dashboard</div>
              <div style={{ fontSize:10, color:'var(--muted)', marginTop:2,
                fontFamily:'var(--font-mono)' }}>Jure Slope · Richards Eq.</div>
            </>
        }
      </div>

      {/* Nav */}
      <nav style={{ flex:1, overflowY:'auto', padding:'8px 6px' }}>
        {NAV.map(({ section, items }) => (
          <div key={section} style={{ marginBottom:16 }}>
            {!collapsed && (
              <div style={{ fontSize:9, fontWeight:700, letterSpacing:'0.12em',
                textTransform:'uppercase', color:'var(--muted)', padding:'4px 10px 4px',
                opacity:.7 }}>
                {section}
              </div>
            )}
            {items.map(({ to, icon: Icon, label }) => (
              <NavLink key={to} to={to} end={to === '/'}
                title={collapsed ? label : undefined}
                style={({ isActive }) => ({
                  display:'flex', alignItems:'center',
                  gap: collapsed ? 0 : 9,
                  padding: collapsed ? '9px 0' : '7px 10px',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  borderRadius:6, textDecoration:'none', marginBottom:1,
                  fontSize:12, fontWeight:500,
                  background: isActive ? 'rgba(47,129,247,.15)' : 'transparent',
                  color: isActive ? 'var(--accent)' : 'var(--muted)',
                  transition:'all .1s',
                })}>
                <Icon size={14} strokeWidth={isActive => isActive ? 2.5 : 1.8} />
                {!collapsed && label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* Geo params panel */}
      {!collapsed && (
        <div style={{ borderTop:'1px solid var(--border)', padding:'10px 12px' }}>
          <button onClick={() => setGeoOpen(!geoOpen)}
            style={{ display:'flex', alignItems:'center', justifyContent:'space-between',
              width:'100%', background:'none', border:'none', cursor:'pointer',
              color:'var(--muted)', fontSize:10, fontWeight:700, letterSpacing:'0.1em',
              textTransform:'uppercase', padding:'4px 0' }}>
            <span style={{ display:'flex', alignItems:'center', gap:6 }}>
              <Sliders size={11} /> Geo Parameters
            </span>
            {geoOpen ? <ChevronLeft size={11} /> : <ChevronRight size={11} />}
          </button>
          {geoOpen && (
            <div style={{ marginTop:12, display:'flex', flexDirection:'column', gap:12 }}>
              {GEO_SLIDERS.map(({ key, label, min, max, step, unit }) => (
                <SliderField key={key} label={label} value={geo[key] ?? 0}
                  onChange={v => updateGeo(key, v)} min={min} max={max} step={step}
                  fmt={v => `${step < 1 ? v.toFixed(step < 0.1 ? 3 : 2) : v.toFixed(0)}${unit ?? ''}`}
                />
              ))}
              <button onClick={resetGeo}
                style={{ display:'flex', alignItems:'center', gap:5, fontSize:10,
                  color:'var(--muted)', background:'none', border:'1px solid var(--border)',
                  borderRadius:4, padding:'4px 10px', cursor:'pointer', width:'fit-content',
                  marginTop:4 }}>
                <RotateCcw size={10} /> Reset defaults
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const W = collapsed ? 44 : 220;

  return (
    <div style={{ display:'flex', height:'100vh', background:'var(--bg)',
      overflow:'hidden', fontFamily:'var(--font-sans)' }}>

      {/* Sidebar */}
      <aside style={{ width:W, minWidth:W, background:'var(--surface)',
        borderRight:'1px solid var(--border)', display:'flex', flexDirection:'column',
        transition:'width .2s', overflow:'hidden' }}>
        <Sidebar collapsed={collapsed} />
        <button onClick={() => setCollapsed(!collapsed)}
          style={{ padding:'8px 0', borderTop:'1px solid var(--border)',
            background:'none', border:'none', cursor:'pointer', color:'var(--muted)',
            display:'flex', justifyContent:'center' }}>
          {collapsed ? <ChevronRight size={13} /> : <PanelLeftClose size={13} />}
        </button>
      </aside>

      {/* Main */}
      <main style={{ flex:1, overflow:'hidden', display:'flex', flexDirection:'column' }}>
        <div style={{ flex:1, overflowY:'auto', padding:'20px 24px' }}>
          {children}
        </div>
      </main>
    </div>
  );
}
