import React from 'react'
import { ShieldAlert, ShieldCheck, Zap, Activity, Cpu, AlertTriangle, Layers, BarChart2 } from 'lucide-react'

export default function VerdictCard({ prediction, loading }) {
  if (loading) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col items-center justify-center h-full min-h-[380px]">
        <div className="relative flex items-center justify-center">
          <div className="w-16 h-16 rounded-full border-4 border-cyan-500/20 border-t-cyan-500 animate-spin"></div>
          <Activity className="w-6 h-6 text-cyan-400 absolute animate-pulse" />
        </div>
        <p className="mt-4 text-sm font-medium text-slate-300">Evaluating flow vector through tree splits...</p>
        <p className="text-xs text-slate-500 font-mono mt-1">Traversing edge classifier nodes</p>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col items-center justify-center h-full min-h-[380px] text-center">
        <div className="w-14 h-14 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center mb-3">
          <ShieldAlert className="w-7 h-7 text-slate-500" />
        </div>
        <h3 className="text-base font-semibold text-slate-200">Awaiting Flow Vector</h3>
        <p className="text-xs text-slate-400 max-w-xs mt-1.5 leading-relaxed">
          Select a curated preset from the left or tweak feature values in the center, then click <strong className="text-cyan-400">Inspect Network Flow</strong>.
        </p>
      </div>
    )
  }

  const isAttack = prediction.is_attack
  const confidencePercent = Math.round((prediction.confidence || 1.0) * 100)

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
            Inference Telemetry
          </h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
          {prediction.model_name}
        </span>
      </div>

      {/* Main Verdict Card */}
      <div
        className={`p-4 rounded-xl border relative overflow-hidden transition-all mb-4 ${
          isAttack
            ? 'bg-gradient-to-br from-red-950/80 via-slate-900/90 to-red-950/40 border-red-500/60 shadow-lg shadow-red-500/10'
            : 'bg-gradient-to-br from-emerald-950/80 via-slate-900/90 to-emerald-950/40 border-emerald-500/60 shadow-lg shadow-emerald-500/10'
        }`}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                isAttack ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
              }`}
            >
              {isAttack ? <AlertTriangle className="w-6 h-6 animate-pulse" /> : <ShieldCheck className="w-6 h-6" />}
            </div>
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                Classification Verdict
              </div>
              <div className={`text-lg font-extrabold tracking-tight ${isAttack ? 'text-red-400' : 'text-emerald-400'}`}>
                {prediction.verdict}
              </div>
              <div className="text-xs font-mono text-slate-300">
                Category: <strong className="text-white uppercase">{prediction.predicted_category}</strong>
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-[10px] font-mono uppercase text-slate-400">Confidence</div>
            <div className="text-xl font-mono font-bold text-cyan-300">
              {confidencePercent}%
            </div>
          </div>
        </div>
      </div>

      {/* Edge Metrics Pill Grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="bg-slate-950/70 border border-slate-800/80 p-2.5 rounded-xl">
          <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1 mb-0.5">
            <Zap className="w-3 h-3 text-amber-400" />
            <span>Measured Latency</span>
          </div>
          <div className="text-sm font-mono font-bold text-white">
            {prediction.latency_us} <span className="text-xs text-slate-400">µs</span>
          </div>
          <div className="text-[9px] text-emerald-400 font-mono mt-0.5">
            Ultra-low edge overhead
          </div>
        </div>

        <div className="bg-slate-950/70 border border-slate-800/80 p-2.5 rounded-xl">
          <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1 mb-0.5">
            <Layers className="w-3 h-3 text-cyan-400" />
            <span>Features Checked</span>
          </div>
          <div className="text-sm font-mono font-bold text-white">
            {prediction.features_evaluated} <span className="text-xs text-slate-400">features</span>
          </div>
          <div className="text-[9px] text-cyan-400 font-mono mt-0.5">
            Decision Tree traversal
          </div>
        </div>
      </div>

      {/* Class Probabilities Distribution */}
      {prediction.probabilities && Object.keys(prediction.probabilities).length > 0 && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-[11px] font-medium text-slate-300 mb-1.5">
            <span className="flex items-center gap-1 text-slate-400 font-mono text-[10px] uppercase">
              <BarChart2 className="w-3 h-3 text-cyan-400" /> Class Probabilities
            </span>
            <span className="text-[10px] text-slate-500 font-mono">8 Categories</span>
          </div>
          <div className="space-y-1.5">
            {Object.entries(prediction.probabilities).map(([cat, prob]) => {
              const pPercent = Math.round(prob * 100)
              const isSelected = cat.toLowerCase() === prediction.predicted_category.toLowerCase()
              return (
                <div key={cat} className="text-xs font-mono">
                  <div className="flex justify-between text-[11px] mb-0.5">
                    <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-slate-400'}>{cat}</span>
                    <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-slate-500'}>{pPercent}%</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isSelected
                          ? (isAttack ? 'bg-red-500' : 'bg-emerald-500')
                          : 'bg-slate-700'
                      }`}
                      style={{ width: `${Math.max(pPercent, prob > 0 ? 3 : 0)}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Anomalous Feature Indicators */}
      {prediction.anomalous_indicators && prediction.anomalous_indicators.length > 0 && (
        <div className="mt-auto pt-3 border-t border-slate-800">
          <div className="text-[10px] font-mono uppercase text-slate-400 mb-1.5 flex items-center justify-between">
            <span>Significant Deviations (vs Normal Mean)</span>
            <span className="text-amber-400">Z-Score Alert</span>
          </div>
          <div className="space-y-1">
            {prediction.anomalous_indicators.map((ind, i) => (
              <div key={i} className="flex items-center justify-between text-[11px] font-mono bg-slate-950/60 px-2 py-1 rounded border border-slate-800/80">
                <span className="text-slate-300 truncate max-w-[120px]">{ind.feature}</span>
                <span className="text-cyan-300 font-semibold">{ind.value}</span>
                <span className="text-amber-400 font-bold">+{ind.deviation_sigma}σ</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

