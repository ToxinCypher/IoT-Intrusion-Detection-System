import React, { useState } from 'react'
import { Sliders, RotateCcw, Play, CheckCircle2, AlertCircle } from 'lucide-react'

const TOP15_FEATURES = new Set([
  'IAT', 'flow_duration', 'Header_Length', 'Tot size', 'rst_count',
  'Protocol Type', 'Weight', 'Magnitue', 'Tot sum', 'Min', 'urg_count',
  'Srate', 'Max', 'Duration', 'AVG'
])

export default function FeatureForm({
  featuresMeta,
  featureGroups,
  formValues,
  onChangeFeature,
  onResetDefaults,
  onSubmit,
  loading,
  isTop15Model,
}) {
  const groupNames = Object.keys(featureGroups)
  const [activeGroup, setActiveGroup] = useState(groupNames[0] || 'Flow & Timing')

  const currentFeatures = (featureGroups[activeGroup] || []).map((name) => {
    const meta = featuresMeta.find((m) => m.name === name) || {
      name,
      min: 0,
      max: 100,
      mean: 0,
      default: 0,
    }
    return meta
  })

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 sm:p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
            Network Flow Vector
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {isTop15Model && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Top-15 Edge Pruning Active
            </span>
          )}
          <button
            onClick={onResetDefaults}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700/60 transition"
            title="Reset features to benign baseline"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 mb-4 overflow-x-auto gap-2">
        {groupNames.map((grp) => {
          const isActive = activeGroup === grp
          return (
            <button
              key={grp}
              onClick={() => setActiveGroup(grp)}
              className={`pb-2 text-xs font-medium whitespace-nowrap transition border-b-2 px-1 ${
                isActive
                  ? 'border-cyan-400 text-cyan-300 font-semibold'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {grp}
            </button>
          )
        })}
      </div>

      {/* Feature Inputs Grid */}
      <div className="flex-1 overflow-y-auto max-h-[340px] pr-1 grid grid-cols-1 md:grid-cols-2 gap-3">
        {currentFeatures.map((feat) => {
          const val = formValues[feat.name] !== undefined ? formValues[feat.name] : feat.default
          const isPrunedInTop15 = isTop15Model && !TOP15_FEATURES.has(feat.name)

          return (
            <div
              key={feat.name}
              className={`p-2.5 rounded-xl border transition ${
                isPrunedInTop15
                  ? 'bg-slate-950/30 border-slate-800/40 opacity-40'
                  : 'bg-slate-950/70 border-slate-800/80 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-xs font-mono font-medium text-slate-200 truncate" title={feat.name}>
                    {feat.name}
                  </span>
                  {TOP15_FEATURES.has(feat.name) && (
                    <span className="text-[9px] font-mono text-emerald-400 bg-emerald-500/10 px-1 py-0.2 rounded border border-emerald-500/20">
                      Top15
                    </span>
                  )}
                </div>
                <span className="text-[10px] font-mono text-slate-400">
                  μ: {feat.mean}
                </span>
              </div>

              <div className="flex items-center gap-2 mt-1.5">
                <input
                  type="number"
                  step="any"
                  value={val}
                  disabled={isPrunedInTop15}
                  onChange={(e) => onChangeFeature(feat.name, parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 rounded-lg px-2.5 py-1 text-xs font-mono text-cyan-200 outline-none transition disabled:cursor-not-allowed"
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* CTA Run Inspection Button */}
      <div className="mt-4 pt-3 border-t border-slate-800">
        <button
          onClick={onSubmit}
          disabled={loading}
          className="w-full py-3 px-4 rounded-xl font-bold text-sm bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 hover:from-cyan-400 hover:via-blue-500 hover:to-indigo-500 text-slate-950 tracking-wide flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/25 transition disabled:opacity-50"
        >
          {loading ? (
            <div className="flex items-center gap-2 text-white">
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></span>
              <span>Evaluating Network Flow...</span>
            </div>
          ) : (
            <>
              <Play className="w-4 h-4 fill-slate-950" />
              <span>Inspect Network Flow</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}

