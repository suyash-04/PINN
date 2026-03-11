import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useApp } from '../context';
import { SliderField } from './ui';
import {
  Home, LineChart, Map, SlidersHorizontal, GitCompare, Droplets, Brain,
  Play, TrendingDown, Waves, BarChart3, CheckCircle, Target, MapPin,
  CloudRain, Shuffle, Download, ChevronLeft, ChevronRight, Settings,
  Menu, X
} from 'lucide-react';

const NAV = [
  { section: 'Core', items: [
    { to: '/',               icon: Home,           label: 'Overview' },
    { to: '/pressure-head',  icon: LineChart,       label: 'Pressure Head' },
    { to: '/factor-of-safety', icon: Map,           label: 'Factor of Safety' },
    { to: '/parameters',     icon: SlidersHorizontal, label: 'Parameter Explorer' },
    { to: '/hydrus',         icon: GitCompare,      label: 'HYDRUS vs PINN' },
    { to: '/soil',           icon: Droplets,        label: 'Soil Properties' },
    { to: '/model',          icon: Brain,           label: 'Model Info' },
  ]},
  { section: 'Advanced', items: [
    { to: '/animation',      icon: Play,            label: 'Animation' },
    { to: '/training',       icon: TrendingDown,    label: 'Training Loss' },
    { to: '/pde-residual',   icon: Waves,           label: 'PDE Residual' },
    { to: '/error',          icon: BarChart3,        label: 'Error Analysis' },
    { to: '/validation',     icon: CheckCircle,     label: 'Validation' },
  ]},
  { section: 'Research', items: [
    { to: '/uncertainty',    icon: Target,          label: 'Uncertainty (MC)' },
    { to: '/critical-slip',  icon: MapPin,          label: 'Critical Slip' },
    { to: '/rainfall',       icon: CloudRain,       label: 'Rainfall Sim' },
    { to: '/scenarios',      icon: Shuffle,         label: 'Scenario Compare' },
    { to: '/export',         icon: Download,        label: 'Export Data' },
  ]},
];

function SidebarContent({ collapsed }) {
  const { geo, updateGeo, defaults } = useApp();
  const [paramsOpen, setParamsOpen] = useState(false);

  return (
    <div className="flex flex-col h-full">
      {/* Brand */}
      <div className="px-4 py-5 border-b border-slate-200">
        {!collapsed ? (
          <>
            <h2 className="text-lg font-bold text-blue-700">🏔️ PINN Dashboard</h2>
            <p className="text-[11px] text-slate-500 mt-0.5">Physics-Informed Neural Network</p>
          </>
        ) : (
          <span className="text-xl block text-center">🏔️</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2 px-2 space-y-4">
        {NAV.map(({ section, items }) => (
          <div key={section}>
            {!collapsed && (
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 px-2 mb-1">{section}</p>
            )}
            <div className="space-y-0.5">
              {items.map(({ to, icon: Icon, label }) => (
                <NavLink key={to} to={to} end={to === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors
                    ${isActive
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`
                  }
                  title={collapsed ? label : undefined}
                >
                  <Icon size={16} />
                  {!collapsed && label}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Geo params panel */}
      {!collapsed && (
        <div className="border-t border-slate-200 px-3 py-3">
          <button onClick={() => setParamsOpen(!paramsOpen)}
            className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider w-full">
            <Settings size={14} />
            Geotechnical Params
            <ChevronRight size={14} className={`ml-auto transition-transform ${paramsOpen ? 'rotate-90' : ''}`} />
          </button>
          {paramsOpen && (
            <div className="mt-3 space-y-3">
              <SliderField label="β (slope °)" value={geo.beta} onChange={(v) => updateGeo('beta', v)} min={10} max={60} step={0.5} />
              <SliderField label="c′ (kPa)" value={geo.c_prime} onChange={(v) => updateGeo('c_prime', v)} min={0} max={50} step={0.5} />
              <SliderField label="φ′ (°)" value={geo.phi_prime} onChange={(v) => updateGeo('phi_prime', v)} min={10} max={45} step={0.5} />
              <SliderField label="γ (kN/m³)" value={geo.gamma} onChange={(v) => updateGeo('gamma', v)} min={14} max={26} step={0.1} />
              <SliderField label="α VG (1/m)" value={geo.alpha} onChange={(v) => updateGeo('alpha', v)} min={0.1} max={5} step={0.1} />
              <SliderField label="n VG" value={geo.n} onChange={(v) => updateGeo('n', v)} min={1.05} max={3} step={0.05} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      {/* Desktop sidebar */}
      <aside className={`hidden lg:flex flex-col border-r border-slate-200 bg-white transition-all duration-200
        ${collapsed ? 'w-16' : 'w-60'}`}>
        <SidebarContent collapsed={collapsed} />
        <button onClick={() => setCollapsed(!collapsed)}
          className="p-2 border-t border-slate-200 text-slate-400 hover:text-slate-600 flex justify-center">
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </aside>

      {/* Mobile sidebar overlay */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setMobileOpen(false)} />
          <aside className="relative w-64 h-full bg-white shadow-xl">
            <SidebarContent collapsed={false} />
          </aside>
        </div>
      )}

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar (mobile) */}
        <div className="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-slate-200 bg-white">
          <button onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
          <span className="font-bold text-blue-700">🏔️ PINN Dashboard</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4 lg:p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
