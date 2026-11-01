import React from 'react'
import { Cpu, Zap, Award, Layers, Sparkles } from 'lucide-react'

export default function ModelSelector({ models, selectedModelId, onSelectModel }) {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
              Selected Classifier Model
            </h2>
          </div>
          <p className="text-xs text-slate-400">
            Compare lightweight edge models against baseline ensembles
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
        {models.map((m) => {
          const isSelected = m.id === selectedModelId
          return (
            <button
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              className={`text-left p-3 rounded-xl border transition-all relative flex flex-col justify-between ${
                isSelected
                  ? 'bg-cyan-950/40 border-cyan-500/80 shadow-md shadow-cyan-500/10 ring-1 ring-cyan-500/50'
                  : 'bg-slate-900/40 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40 text-slate-300'
              }`}
            >
              {m.recommended && (
                <span className="absolute -top-2 right-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase tracking-wider flex items-center gap-0.5 shadow-sm">
                  <Sparkles className="w-2.5 h-2.5" /> Best
                </span>
              )}

              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded ${
                    m.features_count <= 15 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-800 text-slate-400'
                  }`}>
                    {m.features_count} Feats
                  </span>
                  <span className="text-[11px] font-mono text-cyan-300 font-bold">
                    {m.accuracy}% Acc
                  </span>
                </div>
                <div className={`font-medium text-xs line-clamp-1 ${isSelected ? 'text-cyan-200 font-semibold' : 'text-slate-200'}`}>
                  {m.name}
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span className="flex items-center gap-1">
                  <Zap className="w-3 h-3 text-amber-400" /> {m.latency_us} µs
                </span>
                <span>{m.size_kb} KB</span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

