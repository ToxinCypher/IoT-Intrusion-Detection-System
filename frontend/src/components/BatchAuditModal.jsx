import React, { useState } from 'react'
import { X, UploadCloud, FileText, CheckCircle, AlertTriangle, ShieldCheck, Activity, Zap } from 'lucide-react'

export default function BatchAuditModal({ isOpen, onClose, selectedModelId, models }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  if (!isOpen) return null

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a CSV file first.')
      return
    }

    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`/api/predict/batch?model_id=${selectedModelId}`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Failed to process batch CSV')
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <UploadCloud className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Batch Traffic Audit & Anomaly Scan</h3>
              <p className="text-xs text-slate-400">Upload network flow CSV logs to audit bulk traffic</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* File input card */}
          <div className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 rounded-xl p-6 text-center bg-slate-950/40 transition">
            <input
              type="file"
              accept=".csv"
              id="csvUpload"
              className="hidden"
              onChange={handleFileChange}
            />
            <label htmlFor="csvUpload" className="cursor-pointer flex flex-col items-center">
              <FileText className="w-10 h-10 text-cyan-400 mb-2 opacity-80" />
              <span className="text-sm font-semibold text-slate-200">
                {file ? file.name : 'Click to select CSV traffic file'}
              </span>
              <span className="text-xs text-slate-500 mt-1">
                CSV should contain CICIoT2023 flow features (IAT, flow_duration, Rate, etc.)
              </span>
            </label>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-950/50 border border-red-500/50 text-red-300 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">
              Model: <strong className="text-cyan-300">{selectedModelId}</strong>
            </span>
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="px-5 py-2.5 rounded-xl font-bold text-xs bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Processing flows...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Run Batch Scan</span>
                </>
              )}
            </button>
          </div>

          {/* Results Summary */}
          {result && (
            <div className="space-y-4 pt-4 border-t border-slate-800">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-[10px] font-mono uppercase text-slate-400">Total Flows</div>
                  <div className="text-lg font-mono font-bold text-white">{result.total_flows}</div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-[10px] font-mono uppercase text-red-400">Attacks Detected</div>
                  <div className="text-lg font-mono font-bold text-red-400">{result.attack_count} ({result.attack_percentage}%)</div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-[10px] font-mono uppercase text-emerald-400">Benign Traffic</div>
                  <div className="text-lg font-mono font-bold text-emerald-400">{result.benign_count}</div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <div className="text-[10px] font-mono uppercase text-cyan-400">Throughput</div>
                  <div className="text-lg font-mono font-bold text-cyan-400">{result.throughput_flows_sec} <span className="text-xs">flows/s</span></div>
                </div>
              </div>

              {/* Category Breakdown */}
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                <div className="text-xs font-semibold text-slate-300 mb-2 uppercase font-mono tracking-wider">
                  Attack Category Breakdown
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.category_breakdown).map(([cat, cnt]) => (
                    <span
                      key={cat}
                      className={`text-xs font-mono px-2.5 py-1 rounded-lg border ${
                        cat === 'Benign'
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                          : 'bg-red-500/10 text-red-400 border-red-500/20'
                      }`}
                    >
                      {cat}: <strong>{cnt}</strong>
                    </span>
                  ))}
                </div>
              </div>

              {/* Sample Table */}
              {result.preview && result.preview.length > 0 && (
                <div className="border border-slate-800 rounded-xl overflow-hidden max-h-56 overflow-y-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800 sticky top-0">
                      <tr>
                        <th className="p-2">#</th>
                        <th className="p-2">Verdict</th>
                        <th className="p-2">Category</th>
                        <th className="p-2">Rate (/s)</th>
                        <th className="p-2">Duration</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                      {result.preview.map((row) => (
                        <tr key={row.index} className="hover:bg-slate-800/40">
                          <td className="p-2 text-slate-500">{row.index}</td>
                          <td className="p-2 font-bold">
                            {row.is_attack ? (
                              <span className="text-red-400">ATTACK</span>
                            ) : (
                              <span className="text-emerald-400">BENIGN</span>
                            )}
                          </td>
                          <td className="p-2 text-slate-200">{row.predicted_category}</td>
                          <td className="p-2 text-slate-400">{row.Rate.toFixed(1)}</td>
                          <td className="p-2 text-slate-400">{row.flow_duration.toFixed(3)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

