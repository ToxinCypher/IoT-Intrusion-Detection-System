import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import ModelSelector from './components/ModelSelector'
import PresetSelector from './components/PresetSelector'
import FeatureForm from './components/FeatureForm'
import VerdictCard from './components/VerdictCard'
import BatchAuditModal from './components/BatchAuditModal'
import BenchmarksModal from './components/BenchmarksModal'

export default function App() {
  const [models, setModels] = useState([])
  const [selectedModelId, setSelectedModelId] = useState('decision_tree_best')
  const [featuresMeta, setFeaturesMeta] = useState([])
  const [featureGroups, setFeatureGroups] = useState({})
  const [samples, setSamples] = useState([])
  const [activeSample, setActiveSample] = useState(null)
  const [formValues, setFormValues] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [batchModalOpen, setBatchModalOpen] = useState(false)
  const [benchmarksModalOpen, setBenchmarksModalOpen] = useState(false)

  // Initialize data from API
  useEffect(() => {
    async function initData() {
      try {
        const [modelsRes, featsRes, samplesRes] = await Promise.all([
          fetch('/api/models'),
          fetch('/api/features'),
          fetch('/api/samples'),
        ])

        const modelsData = await modelsRes.json()
        const featsData = await featsRes.json()
        const samplesData = await samplesRes.json()

        setModels(modelsData)
        setFeaturesMeta(featsData.features || [])
        setFeatureGroups(featsData.groups || {})
        setSamples(samplesData || [])

        // Initialize default form values from feature metadata
        const initialVals = {}
        ;(featsData.features || []).forEach((f) => {
          initialVals[f.name] = f.default
        })
        setFormValues(initialVals)

        // Select first sample by default
        if (samplesData && samplesData.length > 0) {
          const firstSample = samplesData[0]
          setActiveSample(firstSample)
          const sampleVals = { ...initialVals }
          Object.keys(firstSample).forEach((k) => {
            if (!k.startsWith('label')) {
              sampleVals[k] = Number(firstSample[k]) || 0
            }
          })
          setFormValues(sampleVals)
          runInference('decision_tree_best', sampleVals)
        }
      } catch (err) {
        console.error('Failed to initialize API data:', err)
      }
    }

    initData()
  }, [])

  // Run live inference
  const runInference = async (modelId, valuesToPredict) => {
    setLoading(true)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          features: valuesToPredict || formValues,
        }),
      })
      if (!res.ok) {
        throw new Error('Inference request failed')
      }
      const data = await res.json()
      setPrediction(data)
    } catch (err) {
      console.error('Inference error:', err)
    } finally {
      setLoading(false)
    }
  }

  // Model switch: re-run prediction on existing vector
  const handleSelectModel = (modelId) => {
    setSelectedModelId(modelId)
    runInference(modelId, formValues)
  }

  // Preset selected
  const handleSelectSample = (sample) => {
    setActiveSample(sample)
    const newVals = { ...formValues }
    Object.keys(sample).forEach((k) => {
      if (!k.startsWith('label')) {
        newVals[k] = Number(sample[k]) || 0
      }
    })
    setFormValues(newVals)
    runInference(selectedModelId, newVals)
  }

  // Random preset
  const handleRandomSample = () => {
    if (!samples.length) return
    const randomIdx = Math.floor(Math.random() * samples.length)
    handleSelectSample(samples[randomIdx])
  }

  // Feature value changed
  const handleChangeFeature = (name, value) => {
    setFormValues((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  // Reset to defaults
  const handleResetDefaults = () => {
    const defaults = {}
    featuresMeta.forEach((f) => {
      defaults[f.name] = f.default
    })
    setFormValues(defaults)
    setActiveSample(null)
    runInference(selectedModelId, defaults)
  }

  const isTop15 = selectedModelId === 'decision_tree_top15'

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation */}
      <Navbar
        onOpenBatch={() => setBatchModalOpen(true)}
        onOpenBenchmarks={() => setBenchmarksModalOpen(true)}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-5">
        {/* Model Selector Bar */}
        <ModelSelector
          models={models}
          selectedModelId={selectedModelId}
          onSelectModel={handleSelectModel}
        />

        {/* 3-Column Interactive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
          {/* Left: Preset Selector (3 cols) */}
          <div className="lg:col-span-3 h-full">
            <PresetSelector
              samples={samples}
              activeSample={activeSample}
              onSelectSample={handleSelectSample}
              onRandomSample={handleRandomSample}
            />
          </div>

          {/* Center: Feature Vector Inspector & Editor (5 cols) */}
          <div className="lg:col-span-5 h-full">
            <FeatureForm
              featuresMeta={featuresMeta}
              featureGroups={featureGroups}
              formValues={formValues}
              onChangeFeature={handleChangeFeature}
              onResetDefaults={handleResetDefaults}
              onSubmit={() => runInference(selectedModelId, formValues)}
              loading={loading}
              isTop15Model={isTop15}
            />
          </div>

          {/* Right: Live Telemetry & Verdict (4 cols) */}
          <div className="lg:col-span-4 h-full">
            <VerdictCard
              prediction={prediction}
              loading={loading}
            />
          </div>
        </div>
      </main>

      {/* Modals */}
      <BatchAuditModal
        isOpen={batchModalOpen}
        onClose={() => setBatchModalOpen(false)}
        selectedModelId={selectedModelId}
        models={models}
      />

      <BenchmarksModal
        isOpen={benchmarksModalOpen}
        onClose={() => setBenchmarksModalOpen(false)}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/60 py-4 px-6 text-center text-xs text-slate-500 font-mono">
        IoT IDS Live Inspector • CICIoT2023 Reproducibility & Edge Optimization • DeepMind Advanced Agentic Coding
      </footer>
    </div>
  )
}
