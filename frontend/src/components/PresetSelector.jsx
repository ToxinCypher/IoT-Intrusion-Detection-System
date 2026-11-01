import React, { useState } from 'react'
import { Sparkles, Dna, Radio, Shield, AlertTriangle, RefreshCw } from 'lucide-react'

const CATEGORY_COLORS = {
  Benign: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20',
  DDoS: 'bg-red-500/10 text-red-400 border-red-500/30 hover:bg-red-500/20',
  DoS: 'bg-orange-500/10 text-orange-400 border-orange-500/30 hover:bg-orange-500/20',
  Mirai: 'bg-purple-500/10 text-purple-400 border-purple-500/30 hover:bg-purple-500/20',
  Recon: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/20',
  Web: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/20',
  Spoofing: 'bg-pink-500/10 text-pink-400 border-pink-500/30 hover:bg-pink-500/20',
  BruteForce: 'bg-amber-500/10 text-amber-400 border-amber-500/30 hover:bg-amber-500/20',
}

export default function PresetSelector({
  samples,
  activeSample,
  onSelectSample,
  onRandomSample,
}) {
  const [selectedCategory, setSelectedCategory] = useState('All')

  const categories = ['All', 'Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Web', 'Spoofing', 'BruteForce']

  const filteredSamples = selectedCategory === 'All'
    ? samples
    : samples.filter((s) => s.label_category?.toLowerCase() === selectedCategory.toLowerCase())

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 sm:p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Dna className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider">
            Curated Traffic Presets
          </h2>
        </div>
        <button
          onClick={onRandomSample}
          className="flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 bg-cyan-950/40 border border-cyan-800/60 px-2.5 py-1 rounded-lg transition"
          title="Pick random test flow"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Random</span>
        </button>
      </div>

      <p className="text-xs text-slate-400 mb-3">
        Inject validated flows sampled directly from the CICIoT2023 test partition:
      </p>

      {/* Category Filter Pills */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {categories.map((cat) => {
          const isActive = selectedCategory === cat
          return (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`text-[11px] px-2.5 py-1 rounded-lg font-medium transition ${
                isActive
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                  : 'bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 border border-slate-700/60'
              }`}
            >
              {cat}
            </button>
          )
        })}
      </div>

      {/* Preset List */}
      <div className="flex-1 overflow-y-auto max-h-[360px] space-y-1.5 pr-1">
        {filteredSamples.map((sample, idx) => {
          const isCurrent = activeSample && activeSample.label === sample.label && activeSample.flow_duration === sample.flow_duration
          const cat = sample.label_category || 'Unknown'
          const badgeClass = CATEGORY_COLORS[cat] || 'bg-slate-800 text-slate-400 border-slate-700'

          return (
            <button
              key={idx}
              onClick={() => onSelectSample(sample)}
              className={`w-full text-left p-2.5 rounded-xl border transition flex items-center justify-between group ${
                isCurrent
                  ? 'bg-cyan-950/40 border-cyan-500/80 ring-1 ring-cyan-500/40'
                  : 'bg-slate-900/40 border-slate-800/80 hover:bg-slate-800/40 hover:border-slate-700'
              }`}
            >
              <div className="min-w-0 flex-1 pr-2">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md border ${badgeClass}`}>
                    {cat}
                  </span>
                  <span className="text-xs font-medium text-slate-200 truncate group-hover:text-cyan-300 transition">
                    {sample.label}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-mono flex items-center gap-3">
                  <span>Rate: {Number(sample.Rate || 0).toFixed(1)}/s</span>
                  <span>IAT: {(Number(sample.IAT || 0) / 1e6).toFixed(1)}ms</span>
                </div>
              </div>
              <Radio className={`w-3.5 h-3.5 flex-shrink-0 ${isCurrent ? 'text-cyan-400' : 'text-slate-600 group-hover:text-slate-400'}`} />
            </button>
          )
        })}
      </div>

      {/* Active Preset Ground Truth Banner */}
      {activeSample && (
        <div className="mt-3 pt-3 border-t border-slate-800 flex items-center justify-between text-xs bg-slate-950/40 p-2.5 rounded-xl border border-slate-800">
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Ground Truth</div>
            <div className="font-semibold text-slate-200 flex items-center gap-1.5">
              {activeSample.label_category === 'Benign' ? (
                <Shield className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              )}
              <span>{activeSample.label}</span>
            </div>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
            {activeSample.label_binary || (activeSample.label_category === 'Benign' ? 'Benign' : 'Attack')}
          </span>
        </div>
      )}
    </div>
  )
}

