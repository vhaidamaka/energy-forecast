import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useStore } from '../store'
import { datasetsApi, runsApi } from '../api/client'
import { StatusBadge } from '../components/StatusBadge'
import toast from 'react-hot-toast'
import { Layers, Play, CheckSquare, Square, ChevronRight, Info } from 'lucide-react'

const MODEL_COLORS = {
  arima:'#eab308', lstm:'#00e5ff', bilstm:'#f97316', gru:'#a855f7',
  bigru:'#ec4899', wavelet_lstm:'#22c55e', transformer:'#38bdf8',
  tft:'#fb7185', nbeats:'#06b6d4',
}

const BLEND_OPTIONS = [
  { value: 'ridge',          label: 'Ridge regression', desc: 'Learns optimal weights from test set data — best accuracy' },
  { value: 'weighted_mape',  label: 'Weighted by MAPE',  desc: 'Weight = 1/MAPE — good models get more say, no fitting needed' },
  { value: 'mean',           label: 'Simple average',    desc: 'Equal weights — robust baseline' },
]

export default function Ensemble() {
  const navigate = useNavigate()
  const { runs, setRuns, addRun } = useStore()

  const [selectedIds, setSelectedIds]   = useState([])
  const [blendMethod, setBlendMethod]   = useState('ridge')
  const [alpha, setAlpha]               = useState(1.0)
  const [horizon, setHorizon]           = useState(7)
  const [runName, setRunName]           = useState('')
  const [submitting, setSubmitting]     = useState(false)

  useEffect(() => { runsApi.list().then(r => setRuns(r.data)) }, [])

  const doneRuns = runs.filter(r => r.status === 'done' && r.result?.test_predicted_json)
  const toggle   = (id) => setSelectedIds(p => p.includes(id) ? p.filter(i=>i!==id) : [...p, id])

  const preview = selectedIds.length >= 2
    ? doneRuns.filter(r => selectedIds.includes(r.id))
    : []

  const handleSubmit = async () => {
    if (selectedIds.length < 2) return toast.error('Select at least 2 runs to ensemble')
    setSubmitting(true)
    try {
      const payload = {
        dataset_id:  doneRuns.find(r => r.id === selectedIds[0])?.dataset_id,
        model:       'ensemble',
        horizon_days: horizon,
        name:        runName || `Ensemble (${selectedIds.join('+')})`,
        hyperparams: {
          base_run_ids: selectedIds,
          alpha,
          blend_method: blendMethod,
        }
      }
      const res  = await runsApi.create(payload)
      const run  = res.data
      addRun(run)
      await runsApi.start(run.id)
      toast.success(`Ensemble run #${run.id} started!`)
      navigate(`/training/${run.id}`)
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Failed to start ensemble')
    } finally { setSubmitting(false) }
  }

  return (
    <div className="p-8 animate-fade-in max-w-3xl">
      <div className="mb-8">
        <h1 className="section-title mb-1">Stacking Ensemble</h1>
        <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
          Blend predictions from multiple trained runs using a meta-learner
        </p>
      </div>

      {/* Info banner */}
      <div className="card p-4 mb-6 flex gap-3"
           style={{ borderColor: 'var(--accent-dim)', backgroundColor: 'var(--accent-bg)' }}>
        <Info size={16} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }} />
        <div className="text-sm" style={{ color: 'var(--fg-muted)' }}>
          The ensemble does <strong style={{ color: 'var(--fg)' }}>not re-train</strong> the base models.
          It loads their saved test-set predictions, fits a Ridge meta-learner in seconds,
          and blends their forecasts. Only runs with <em>Predicted vs Actual</em> data are eligible.
        </div>
      </div>

      {/* 1. Select runs */}
      <div className="card p-6 mb-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium" style={{ color: 'var(--fg)' }}>
            1. Select base runs <span className="text-xs font-normal ml-1" style={{ color: 'var(--fg-muted)' }}>(min 2)</span>
          </h2>
          {selectedIds.length > 0 && (
            <span className="text-xs px-2 py-1 rounded-lg"
                  style={{ backgroundColor: 'var(--accent-bg)', color: 'var(--accent)' }}>
              {selectedIds.length} selected
            </span>
          )}
        </div>

        {doneRuns.length === 0 ? (
          <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
            No completed runs with predictions yet.{' '}
            <Link to="/configure" style={{ color: 'var(--accent)' }}>Train some models first →</Link>
          </p>
        ) : (
          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {doneRuns.map(run => {
              const sel   = selectedIds.includes(run.id)
              const color = MODEL_COLORS[run.model] || 'var(--fg-muted)'
              return (
                <button key={run.id} onClick={() => toggle(run.id)}
                  className="w-full flex items-center gap-3 p-3 rounded-xl border transition-all duration-150 text-left"
                  style={{
                    borderColor: sel ? 'var(--accent)' : 'var(--border-mid)',
                    backgroundColor: sel ? 'var(--accent-bg)' : 'var(--bg-2)',
                  }}>
                  {sel
                    ? <CheckSquare size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                    : <Square     size={15} style={{ color: 'var(--fg-subtle)', flexShrink: 0 }} />
                  }
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                  <span className="text-sm flex-1 truncate" style={{ color: 'var(--fg)' }}>
                    {run.name || `Run #${run.id}`}
                  </span>
                  <span className="text-xs font-mono" style={{ color }}>
                    {run.model?.toUpperCase()}
                  </span>
                  <span className="text-xs font-mono" style={{ color: 'var(--fg-muted)' }}>
                    MAPE {run.result?.mape?.toFixed(2)}%
                  </span>
                  <span className="text-xs" style={{ color: 'var(--fg-subtle)' }}>
                    {run.horizon_days}d
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {/* Preview: expected blended MAPE range */}
        {preview.length >= 2 && (
          <div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-0)', border: '1px solid var(--border)' }}>
            <div className="text-xs mb-2" style={{ color: 'var(--fg-muted)' }}>Selected models:</div>
            <div className="flex flex-wrap gap-2">
              {preview.map(r => {
                const color = MODEL_COLORS[r.model] || 'var(--fg-muted)'
                return (
                  <span key={r.id} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-lg"
                        style={{ border: `1px solid ${color}30`, backgroundColor: `${color}12` }}>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                    <span style={{ color }}>{r.model?.toUpperCase()}</span>
                    <span style={{ color: 'var(--fg-muted)' }}>{r.result?.mape?.toFixed(1)}%</span>
                  </span>
                )
              })}
            </div>
            <div className="text-xs mt-2" style={{ color: 'var(--fg-subtle)' }}>
              Best base MAPE: <span style={{ color: 'var(--accent)' }}>
                {Math.min(...preview.map(r => r.result?.mape || 99)).toFixed(2)}%
              </span>
              {' '}· Ensemble typically improves by 1–4%
            </div>
          </div>
        )}
      </div>

      {/* 2. Blend method */}
      <div className="card p-6 mb-4">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>2. Blend method</h2>
        <div className="space-y-2">
          {BLEND_OPTIONS.map(opt => {
            const sel = blendMethod === opt.value
            return (
              <button key={opt.value} onClick={() => setBlendMethod(opt.value)}
                className="w-full text-left p-3 rounded-xl border transition-all"
                style={{
                  borderColor: sel ? 'var(--accent)' : 'var(--border-mid)',
                  backgroundColor: sel ? 'var(--accent-bg)' : 'var(--bg-2)',
                }}>
                <div className="text-sm font-medium" style={{ color: sel ? 'var(--accent)' : 'var(--fg)' }}>
                  {opt.label}
                </div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--fg-muted)' }}>{opt.desc}</div>
              </button>
            )
          })}
        </div>

        {blendMethod === 'ridge' && (
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div>
              <label className="label">Ridge alpha (regularisation)</label>
              <input type="number" className="input" value={alpha} min={0.001} max={1000} step={0.1}
                     onChange={e => setAlpha(parseFloat(e.target.value) || 1)} />
              <p className="text-xs mt-1" style={{ color: 'var(--fg-subtle)' }}>
                Lower = tighter fit to test set. Higher = more uniform weights.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 3. Settings */}
      <div className="card p-6 mb-6">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>3. Settings</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Forecast horizon (days)</label>
            <input type="number" className="input" value={horizon} min={1} max={90}
                   onChange={e => setHorizon(parseInt(e.target.value) || 7)} />
          </div>
          <div>
            <label className="label">Run name (optional)</label>
            <input className="input" value={runName}
                   onChange={e => setRunName(e.target.value)}
                   placeholder={`Ensemble (${selectedIds.slice(0,3).join('+')})`} />
          </div>
        </div>
      </div>

      <button className="btn-primary w-full py-3 flex items-center justify-center gap-2 text-base"
              onClick={handleSubmit}
              disabled={submitting || selectedIds.length < 2}>
        {submitting
          ? 'Starting...'
          : <><Layers size={16} /> Create Ensemble</>
        }
      </button>
    </div>
  )
}
