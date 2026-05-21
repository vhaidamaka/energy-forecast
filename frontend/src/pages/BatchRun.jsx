import { useEffect, useState, useRef, useMemo } from 'react'
import { useStore } from '../store'
import { datasetsApi, batchApi, runsApi } from '../api/client'
import { StatusBadge } from '../components/StatusBadge'
import toast from 'react-hot-toast'
import {
  Play, CheckSquare, Square, RefreshCw,
  BarChart3, ChevronRight, History, Plus, Trash2
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { formatDistanceToNow } from 'date-fns'
import { formatDuration } from '../utils'

const ALL_MODELS = [
  { key: 'nbeats',       label: 'N-BEATS',      color: '#06b6d4', desc: 'Doubly residual, trend+seasonality stacks' },
  { key: 'lstm',         label: 'LSTM',          color: '#00e5ff', desc: '2-layer Long Short-Term Memory' },
  { key: 'bilstm',       label: 'BiLSTM',        color: '#f97316', desc: 'Bidirectional LSTM' },
  { key: 'gru',          label: 'GRU',           color: '#a855f7', desc: 'Gated Recurrent Unit' },
  { key: 'bigru',        label: 'BiGRU',         color: '#ec4899', desc: 'Bidirectional GRU' },
  { key: 'wavelet_lstm', label: 'Wavelet+LSTM',  color: '#22c55e', desc: 'DWT features + LSTM' },
  { key: 'transformer',  label: 'Transformer',   color: '#38bdf8', desc: 'Multi-head self-attention encoder' },
  { key: 'tft',          label: 'TFT',           color: '#fb7185', desc: 'Temporal Fusion Transformer' },
  { key: 'arima',        label: 'ARIMA',         color: '#eab308', desc: 'Classical statistical baseline' },

]

// Group runs into "batches" by name prefix (text before " — ")
function groupIntoBatches(runs) {
  const groups = {}
  runs.forEach(run => {
    const prefix = run.name?.includes(' — ')
      ? run.name.split(' — ')[0].trim()
      : run.name?.startsWith('Batch')
        ? 'Batch'
        : null
    if (!prefix) return
    if (!groups[prefix]) groups[prefix] = []
    groups[prefix].push(run)
  })
  // Sort each group by id asc (creation order)
  Object.values(groups).forEach(g => g.sort((a, b) => a.id - b.id))
  // Return sorted by newest first (max id in group)
  return Object.entries(groups)
    .sort((a, b) => Math.max(...b[1].map(r => r.id)) - Math.max(...a[1].map(r => r.id)))
}

export default function BatchRun() {
  const { datasets, runs, setDatasets, setRuns, addRun, updateRun, removeRun } = useStore()

  const [tab, setTab] = useState('history')   // 'new' | 'history'
  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedModels, setSelectedModels] = useState(ALL_MODELS.map(m => m.key))
  const [horizon, setHorizon] = useState(7)
  const [namePrefix, setNamePrefix] = useState('')
  const [submitting, setSubmitting] = useState(false)

  // Active batch session (in-memory, for progress tracking)
  const [activeBatchIds, setActiveBatchIds] = useState([])
  const [activeBatchRuns, setActiveBatchRuns] = useState([])
  const pollRef = useRef(null)

  // Expanded batch in history
  const [expandedBatch, setExpandedBatch] = useState(null)

  useEffect(() => {
    datasetsApi.list().then(r => setDatasets(r.data))
    runsApi.list().then(r => setRuns(r.data))
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  // Keep activeBatchRuns in sync with store
  useEffect(() => {
    if (!activeBatchIds.length) return
    const current = runs.filter(r => activeBatchIds.includes(r.id))
    setActiveBatchRuns(current)
    const allDone = current.length > 0 && current.every(r => ['done','failed'].includes(r.status))
    if (allDone && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
      const done   = current.filter(r => r.status === 'done').length
      const failed = current.filter(r => r.status === 'failed').length
      toast.success(`Batch complete — ${done} done${failed ? `, ${failed} failed` : ''}`)
    }
  }, [runs, activeBatchIds])

  const startPolling = (ids) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      const updates = await Promise.all(ids.map(id => runsApi.get(id).then(r => r.data)))
      updates.forEach(r => updateRun(r.id, r))
    }, 3000)
  }

  const toggleModel = (key) =>
    setSelectedModels(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])

  const handleSubmit = async () => {
    if (!selectedDataset) return toast.error('Select a dataset first')
    if (!selectedModels.length) return toast.error('Select at least one model')
    setSubmitting(true)
    try {
      const res = await batchApi.start(
        parseInt(selectedDataset), selectedModels, horizon, namePrefix || 'Batch'
      )
      const newRuns = res.data
      newRuns.forEach(r => addRun(r))
      setActiveBatchIds(newRuns.map(r => r.id))
      setActiveBatchRuns(newRuns)
      startPolling(newRuns.map(r => r.id))
      toast.success(`${newRuns.length} runs queued`)
      setTab('history')
      setExpandedBatch(newRuns[0]?.name?.split(' — ')[0]?.trim() || 'Batch')
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Batch start failed')
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeleteRun = async (e, id) => {
    e.preventDefault(); e.stopPropagation()
    if (!confirm(`Delete Run #${id}?`)) return
    try { await runsApi.delete(id); removeRun(id); toast.success('Run deleted') }
    catch { toast.error('Delete failed') }
  }

  // Compute batches from all runs in store
  const batches = useMemo(() => groupIntoBatches(runs), [runs])
  const running = activeBatchRuns.some(r => ['pending','training'].includes(r.status))

  return (
    <div className="p-8 animate-fade-in max-w-4xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="section-title mb-1">Batch Run</h1>
          <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
            Run multiple models at once and compare results
          </p>
        </div>
        {running && (
          <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg"
               style={{ backgroundColor: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-dim)' }}>
            <RefreshCw size={12} className="animate-spin" />
            {activeBatchRuns.filter(r => ['done','failed'].includes(r.status)).length} / {activeBatchRuns.length} complete
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex mb-6" style={{ borderBottom: '1px solid var(--border)' }}>
        {[
          { id: 'history', label: 'Batch History', icon: History },
          { id: 'new',     label: 'New Batch',     icon: Plus    },
        ].map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            className="flex items-center gap-2 px-5 py-3 text-sm transition-colors"
            style={{
              color: tab === id ? 'var(--accent)' : 'var(--fg-muted)',
              borderBottom: tab === id ? '2px solid var(--accent)' : '2px solid transparent',
              fontWeight: tab === id ? 600 : 400,
            }}>
            <Icon size={14} />{label}
          </button>
        ))}
      </div>

      {/* ── History tab ── */}
      {tab === 'history' && (
        <div className="space-y-4">
          {batches.length === 0 ? (
            <div className="card p-16 text-center">
              <BarChart3 size={40} className="mx-auto mb-4" style={{ color: 'var(--border-mid)' }} />
              <p className="text-sm mb-4" style={{ color: 'var(--fg-muted)' }}>
                No batch runs yet.
              </p>
              <button className="btn-primary text-sm flex items-center gap-2 mx-auto"
                      onClick={() => setTab('new')}>
                <Plus size={14} /> Start your first batch
              </button>
            </div>
          ) : batches.map(([prefix, bRuns]) => {
            const isActive   = bRuns.some(r => activeBatchIds.includes(r.id))
            const isExpanded = expandedBatch === prefix
            const doneRuns   = bRuns.filter(r => r.status === 'done' && r.result)
            const ranked     = [...doneRuns].sort((a,b) => (a.result?.mae||999) - (b.result?.mae||999))
            const allDone    = bRuns.every(r => ['done','failed'].includes(r.status))
            const totalDone  = bRuns.filter(r => ['done','failed'].includes(r.status)).length
            const newestRun  = bRuns[bRuns.length - 1]
            const bestModel  = ALL_MODELS.find(m => m.key === ranked[0]?.model)
            const ds         = newestRun?.dataset?.name || newestRun?.dataset_id

            return (
              <div key={prefix} className="card overflow-hidden">
                {/* Batch header — clickable to expand */}
                <button
                  className="w-full flex items-center gap-3 px-5 py-4 text-left transition-colors"
                  style={{ backgroundColor: isExpanded ? 'var(--bg-2)' : undefined }}
                  onClick={() => setExpandedBatch(isExpanded ? null : prefix)}
                >
                  {/* Prefix + meta */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold" style={{ color: 'var(--fg)' }}>{prefix}</span>
                      {isActive && !allDone && (
                        <span className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
                          style={{ backgroundColor: 'var(--accent-bg)', color: 'var(--accent)' }}>
                          <RefreshCw size={10} className="animate-spin" /> running
                        </span>
                      )}
                      {allDone && (
                        <span className="text-xs px-2 py-0.5 rounded-full"
                          style={{ backgroundColor: 'rgba(34,197,94,0.1)', color: '#22c55e' }}>
                          done
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-1 text-xs" style={{ color: 'var(--fg-muted)' }}>
                      <span>{bRuns.length} models</span>
                      {ds && <><span>·</span><span>{ds}</span></>}
                      {newestRun?.horizon_days && (
                        <><span>·</span><span>{newestRun.horizon_days}d horizon</span></>
                      )}
                      {newestRun?.started_at && (
                        <><span>·</span>
                          <span>{formatDistanceToNow(new Date(newestRun.started_at), { addSuffix: true })}</span>
                        </>
                      )}
                      {!allDone && (
                        <><span>·</span><span>{totalDone}/{bRuns.length} complete</span></>
                      )}
                    </div>
                  </div>

                  {/* Progress bar (inline for running) */}
                  {!allDone && (
                    <div className="w-24 h-1.5 rounded-full overflow-hidden flex-shrink-0"
                         style={{ backgroundColor: 'var(--bg-3)' }}>
                      <div className="h-full rounded-full transition-all duration-500"
                           style={{ width: `${(totalDone / bRuns.length) * 100}%`, backgroundColor: 'var(--accent)' }} />
                    </div>
                  )}

                  {/* Best model badge */}
                  {allDone && ranked.length > 0 && (
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      <span className="text-xs" style={{ color: 'var(--fg-subtle)' }}>Best:</span>
                      <span className="text-xs font-semibold" style={{ color: bestModel?.color }}>
                        {bestModel?.label}
                      </span>
                      <span className="text-xs font-mono" style={{ color: 'var(--fg-muted)' }}>
                        {ranked[0]?.result?.mape?.toFixed(1)}%
                      </span>
                    </div>
                  )}

                  <ChevronRight size={14} style={{
                    color: 'var(--fg-subtle)',
                    transform: isExpanded ? 'rotate(90deg)' : 'none',
                    transition: 'transform 0.2s',
                    flexShrink: 0,
                  }} />
                </button>

                {/* Expanded content */}
                {isExpanded && (
                  <div style={{ borderTop: '1px solid var(--border)' }}>
                    {/* Run list */}
                    {bRuns.map((run, idx) => {
                      const meta    = ALL_MODELS.find(m => m.key === run.model)
                      const isLive  = ['pending','training'].includes(run.status)
                      return (
                        <div key={run.id}
                          className="flex items-center gap-3 px-5 py-3"
                          style={{ borderBottom: idx < bRuns.length - 1 ? '1px solid var(--border)' : 'none' }}>

                          <span className="w-2 h-2 rounded-full flex-shrink-0"
                                style={{ backgroundColor: meta?.color || 'var(--fg-muted)' }} />

                          <span className="text-sm font-medium w-28 flex-shrink-0" style={{ color: 'var(--fg)' }}>
                            {meta?.label || run.model.toUpperCase()}
                          </span>

                          <StatusBadge status={run.status} />

                          {run.result && (
                            <div className="flex gap-4 ml-2 text-xs font-mono flex-1">
                              <span style={{ color: 'var(--fg-muted)' }}>
                                MAE <span style={{ color: meta?.color }}>{run.result.mae?.toFixed(4)}</span>
                              </span>
                              <span style={{ color: 'var(--fg-muted)' }}>
                                RMSE <span style={{ color: 'var(--fg)' }}>{run.result.rmse?.toFixed(4)}</span>
                              </span>
                              <span style={{ color: 'var(--fg-muted)' }}>
                                MAPE <span style={{ color: meta?.color }}>{run.result.mape?.toFixed(2)}%</span>
                              </span>
                              {run.duration_seconds && (
                                <span style={{ color: 'var(--fg-subtle)' }}>
                                  {formatDuration(run.duration_seconds)}
                                </span>
                              )}
                            </div>
                          )}

                          {!run.result && !isLive && (
                            <span className="flex-1 text-xs" style={{ color: 'var(--fg-subtle)' }}>
                              {run.error_message?.slice(0, 60) || '—'}
                            </span>
                          )}

                          <div className="flex items-center gap-2 flex-shrink-0 ml-auto">
                            {isLive && (
                              <Link to={`/training/${run.id}`}
                                className="text-xs flex items-center gap-1"
                                style={{ color: 'var(--fg-muted)' }}>
                                View log <ChevronRight size={11} />
                              </Link>
                            )}
                            {run.status === 'done' && (
                              <Link to={`/results/${run.id}`}
                                className="text-xs flex items-center gap-1"
                                style={{ color: 'var(--accent)' }}>
                                Results <ChevronRight size={11} />
                              </Link>
                            )}
                            <button onClick={e => handleDeleteRun(e, run.id)}
                              className="p-1 rounded opacity-40 hover:opacity-100 transition-opacity"
                              style={{ color: 'var(--fg-subtle)' }}
                              onMouseEnter={e => e.currentTarget.style.color='#ef4444'}
                              onMouseLeave={e => e.currentTarget.style.color='var(--fg-subtle)'}>
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      )
                    })}

                    {/* Ranking + compare — only when all done */}
                    {allDone && ranked.length > 1 && (
                      <div style={{ borderTop: '1px solid var(--border)', backgroundColor: 'var(--bg-2)' }}>
                        <div className="px-5 py-3">
                          <div className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--fg-subtle)' }}>
                            Ranking by MAE
                          </div>
                          <div className="space-y-1.5">
                            {ranked.map((run, idx) => {
                              const m = ALL_MODELS.find(m => m.key === run.model)
                              const maxMae = ranked[ranked.length - 1]?.result?.mae || 1
                              const pct = Math.max(5, (1 - (run.result.mae / maxMae)) * 100)
                              return (
                                <div key={run.id} className="flex items-center gap-3">
                                  <span className="text-xs w-5 text-right" style={{ color: 'var(--fg-subtle)' }}>
                                    {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : idx + 1}
                                  </span>
                                  <span className="text-xs font-medium w-24" style={{ color: m?.color }}>
                                    {m?.label}
                                  </span>
                                  <div className="flex-1 h-1.5 rounded-full overflow-hidden"
                                       style={{ backgroundColor: 'var(--bg-3)' }}>
                                    <div className="h-full rounded-full"
                                         style={{ width: `${pct}%`, backgroundColor: m?.color, opacity: 0.7 }} />
                                  </div>
                                  <span className="text-xs font-mono w-16 text-right" style={{ color: m?.color }}>
                                    {run.result.mape?.toFixed(2)}%
                                  </span>
                                  <span className="text-xs font-mono w-20 text-right" style={{ color: 'var(--fg-muted)' }}>
                                    {run.result.mae?.toFixed(4)}
                                  </span>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                        <div className="px-5 py-3 flex justify-end" style={{ borderTop: '1px solid var(--border)' }}>
                          <Link to={`/compare?ids=${ranked.map(r => r.id).join(',')}`}>
                            <button className="btn-primary text-xs flex items-center gap-1.5">
                              <BarChart3 size={12} /> Compare all in Compare
                            </button>
                          </Link>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* ── New Batch tab ── */}
      {tab === 'new' && (
        <div className="card p-6">
          {/* Dataset */}
          <div className="mb-5">
            <label className="label">Dataset</label>
            {datasets.length === 0 ? (
              <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
                No datasets yet.{' '}
                <Link to="/datasets" style={{ color: 'var(--accent)' }}>Upload one first →</Link>
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {datasets.map(ds => {
                  const sel = selectedDataset === String(ds.id)
                  return (
                    <button key={ds.id} onClick={() => setSelectedDataset(String(ds.id))}
                      className="text-left p-3 rounded-xl border transition-all duration-150"
                      style={{
                        borderColor: sel ? 'var(--accent)' : 'var(--border-mid)',
                        backgroundColor: sel ? 'var(--accent-bg)' : 'var(--bg-2)',
                      }}>
                      <div className="text-sm font-medium" style={{ color: 'var(--fg)' }}>{ds.name}</div>
                      <div className="text-xs mt-0.5" style={{ color: 'var(--fg-muted)' }}>
                        {ds.rows?.toLocaleString()} rows · {ds.granularity}
                      </div>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Models */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-2">
              <label className="label" style={{ marginBottom: 0 }}>Models</label>
              <div className="flex gap-2">
                <button className="text-xs" style={{ color: 'var(--accent)' }}
                  onClick={() => setSelectedModels(ALL_MODELS.map(m => m.key))}>All</button>
                <span style={{ color: 'var(--fg-subtle)' }}>·</span>
                <button className="text-xs" style={{ color: 'var(--fg-muted)' }}
                  onClick={() => setSelectedModels([])}>None</button>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {ALL_MODELS.map(m => {
                const sel = selectedModels.includes(m.key)
                return (
                  <button key={m.key} onClick={() => toggleModel(m.key)}
                    className="flex items-start gap-2.5 p-3 rounded-xl border transition-all duration-150 text-left"
                    style={{
                      borderColor: sel ? m.color : 'var(--border-mid)',
                      backgroundColor: sel ? `${m.color}14` : 'var(--bg-2)',
                    }}>
                    <div className="mt-0.5 flex-shrink-0">
                      {sel
                        ? <CheckSquare size={14} style={{ color: m.color }} />
                        : <Square     size={14} style={{ color: 'var(--fg-subtle)' }} />}
                    </div>
                    <div>
                      <div className="text-xs font-semibold" style={{ color: sel ? m.color : 'var(--fg)' }}>
                        {m.label}
                      </div>
                      <div className="text-xs mt-0.5 leading-snug" style={{ color: 'var(--fg-muted)' }}>
                        {m.desc}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Horizon + name */}
          <div className="grid grid-cols-2 gap-4 mb-5">
            <div>
              <label className="label">Forecast horizon (days)</label>
              <input type="number" className="input" value={horizon} min={1} max={90}
                onChange={e => setHorizon(parseInt(e.target.value) || 7)} />
            </div>
            <div>
              <label className="label">Batch name</label>
              <input className="input" value={namePrefix}
                onChange={e => setNamePrefix(e.target.value)}
                placeholder="Zorya 7d comparison" />
            </div>
          </div>

          <button className="btn-primary w-full py-3 flex items-center justify-center gap-2 text-base"
            onClick={handleSubmit}
            disabled={submitting || running || !selectedDataset || !selectedModels.length}>
            {submitting
              ? <><RefreshCw size={16} className="animate-spin" /> Starting...</>
              : <><Play size={16} /> Run {selectedModels.length} model{selectedModels.length !== 1 ? 's' : ''}</>
            }
          </button>
        </div>
      )}
    </div>
  )
}
