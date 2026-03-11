import { useState, useEffect } from 'react';
import { api } from '../api';
import { PageHeader, Spinner, Card } from '../components/ui';

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  useEffect(() => { api.getModelInfo().then(setInfo); }, []);

  if (!info) return <Spinner />;

  const arch = info.arch || {};
  const normalization = info.norm || {};

  const sections = [
    { title: '🏗️ Architecture', items: [
      ['Hidden Layers', arch.n_hidden_layers],
      ['Neurons per Layer', arch.neurons_per_layer],
      ['Activation', arch.activation],
      ['Input / Output', `${arch.input_dim} → ${arch.output_dim}`],
      ['Total Parameters', info.total_params?.toLocaleString()],
    ]},
    { title: '📊 Normalization', items: [
      ['t_max', `${normalization.t_max} days`],
      ['z_max', `${normalization.z_max} m`],
      ['ψ_min', `${normalization.psi_min} m`],
      ['ψ_max', `${normalization.psi_max} m`],
    ]},
    { title: '⚙️ Training', items: [
      ['Phase 1', 'Adam (lr scheduling)'],
      ['Phase 2', 'L-BFGS (full batch)'],
      ['Data Points', '24,000 (24 × 1000)'],
      ['Loss', 'Data MSE + PDE Residual + IC + BC'],
    ]},
    { title: '🌍 Geomechanical Defaults', items: info.layers ? info.layers.slice(0, 8).map(l =>
      [l.name, `${l.shape.join('×')} (${l.n_params})`]
    ) : [] },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <PageHeader title="Model Information" subtitle="PINN architecture, training details, and configuration" icon="🧠" />

      {/* Summary banner */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <div className="flex items-center gap-4">
          <div className="text-4xl">🔬</div>
          <div>
            <h3 className="text-base font-bold text-slate-800">Physics-Informed Neural Network for Richards' Equation</h3>
            <p className="text-xs text-slate-600 mt-1">
              Solving variably-saturated flow in hillslopes coupled with infinite-slope stability analysis.
              Uses Van Genuchten soil-water retention model with {arch.n_hidden_layers} hidden layers
              × {arch.neurons_per_layer} neurons ({info.total_params?.toLocaleString()} parameters).
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sections.map(s => (
          <Card key={s.title}>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">{s.title}</h3>
            <dl className="space-y-2">
              {s.items.map(([k, v]) => (
                <div key={k} className="flex justify-between items-center text-xs border-b border-slate-50 pb-1">
                  <dt className="text-slate-500 font-medium">{k}</dt>
                  <dd className="text-slate-800 font-mono">{String(v)}</dd>
                </div>
              ))}
            </dl>
          </Card>
        ))}
      </div>

      {/* Network diagram */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">🔗 Network Topology</h3>
        <div className="flex items-center justify-center gap-1 py-4 overflow-x-auto">
          {/* Input */}
          <div className="flex flex-col items-center min-w-[60px]">
            <div className="w-12 h-12 rounded-lg bg-blue-100 border-2 border-blue-400 flex items-center justify-center text-xs font-bold text-blue-700">2</div>
            <span className="text-[10px] text-slate-500 mt-1">Input</span>
            <span className="text-[9px] text-slate-400">(t, z)</span>
          </div>
          {Array.from({ length: arch.n_hidden_layers || 7 }).map((_, i) => (
            <div key={i} className="flex flex-col items-center min-w-[48px]">
              <div className="text-slate-300 text-lg mb-1">→</div>
              <div className="w-10 h-10 rounded-lg bg-indigo-100 border-2 border-indigo-400 flex items-center justify-center text-xs font-bold text-indigo-700">
                {arch.neurons_per_layer || 64}
              </div>
              <span className="text-[9px] text-slate-400 mt-1">H{i + 1}</span>
            </div>
          ))}
          <div className="flex flex-col items-center min-w-[60px]">
            <div className="text-slate-300 text-lg mb-1">→</div>
            <div className="w-12 h-12 rounded-lg bg-emerald-100 border-2 border-emerald-400 flex items-center justify-center text-xs font-bold text-emerald-700">1</div>
            <span className="text-[10px] text-slate-500 mt-1">Output</span>
            <span className="text-[9px] text-slate-400">ψ(t,z)</span>
          </div>
        </div>
      </Card>

      {/* Physics equations */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">📐 Governing Equations</h3>
        <div className="space-y-3 text-xs text-slate-600">
          <div className="p-3 bg-slate-50 rounded-lg font-mono">
            <p className="font-semibold text-slate-700 mb-1">Richards' Equation (1D vertical):</p>
            <p>C(ψ) · ∂ψ/∂t = ∂/∂z [K(ψ) · (∂ψ/∂z + 1)]</p>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg font-mono">
            <p className="font-semibold text-slate-700 mb-1">Factor of Safety (infinite slope):</p>
            <p>FS = c' / (γ·z·sin β·cos β) + tan φ' / tan β − u·tan φ' / (γ·z·sin β·cos β)</p>
          </div>
          <div className="p-3 bg-slate-50 rounded-lg font-mono">
            <p className="font-semibold text-slate-700 mb-1">Van Genuchten Retention:</p>
            <p>Se = [1 + (α|ψ|)ⁿ]^(−m), θ = θr + (θs − θr)·Se, K = Ks·Se^0.5·[1−(1−Se^(1/m))^m]²</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
