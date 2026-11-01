import React from 'react'
import { ShieldAlert, ShieldCheck, Cpu, Activity, BarChart3, UploadCloud } from 'lucide-react'

export default function Navbar({ activeTab, setActiveTab, onOpenBatch, onOpenBenchmarks }) {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-indigo-600 to-emerald-500 p-0.5 shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <ShieldAlert className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">IoT Flow Inspector</span>
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                CICIoT2023 IDS
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Lightweight Multi-Class ML for Edge IoT Intrusion Detection
            </p>
          </div>
        </div>

        {/* Action Buttons & Status */}
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenBatch}
            className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <UploadCloud className="w-4 h-4 text-cyan-400" />
            <span className="hidden md:inline">Batch CSV Audit</span>
          </button>

          <button
            onClick={onOpenBenchmarks}
            className="flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <BarChart3 className="w-4 h-4 text-emerald-400" />
            <span className="hidden md:inline">Benchmarks</span>
          </button>

          <div className="h-6 w-px bg-slate-800 hidden sm:block" />

          <div className="flex items-center gap-2 bg-slate-950/80 border border-slate-800 px-3 py-1.5 rounded-full">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-[11px] font-mono text-emerald-400 font-medium">Inference Ready</span>
          </div>
        </div>
      </div>
    </header>
  )
}

