import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { runsApi } from '../api/client'
import toast from 'react-hot-toast'
import { useStore } from '../store'
import { StatusBadge } from '../components/StatusBadge'
import { useThemeColors } from '../hooks/useThemeColors'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { GitCompare, X, Trophy, Timer, Trash2 } from 'lucide-react'
import { formatDuration } from '../utils'

const COLORS = ['#00e5ff', '#f97316', '#22c55e', '#a855f7', '#eab308']
const MODEL_COLORS = {
  nbeats: '#06b6d4',
  arima: '#eab308', lstm: '#00e5ff', bilstm: '#f97316',
  gru: '#a855f7', bigru: '#ec4899', wavelet_lstm: '#22c55e',
  transformer: '#38bdf8', tft: '#fb7185',
}

export default function Compare() {
  const { runs, setRuns, compareIds, toggleCompare, clearCompare, removeRun } = useStore()
  const [compareData, setCompareData] = useState([])
  const [allRuns, setAllRuns] = useState([])
  const c = useThemeColors()

  useEffect(() => {
    runsApi.list().then((r) => {
      setRuns(r.data)
      setAllRuns(r.data.filter((r) => r.status === 'done' && r.result))
    })
  }, [])

  useEffect(() => {
    if (compareIds.length < 1) return
    runsApi.compare(compareIds).then((r) => setCompareData(r.data))
  }, [compareIds])

  const handleDeleteRun = async (e, id) => {
    e.stopPropagation()
    if (!confirm(`Delete Run #${id}?`)) return
    try {
      await runsApi.delete(id); removeRun(id); toggleCompare(id)
      setAllRuns(prev => prev.filter(r => r.id !== id))
      toast.success('Run deleted')
    } catch { toast.error('Delete failed') }
  }

  const selected = compareData.filter((d) => d.mae != null)
  const barData = ['mae', 'rmse', 'mape'].map((metric) => ({
    metric: metric.toUpperCase(),
    ...Object.fromEntries(selected.map((d) => [d.name || `#${d.run_id}`, d[metric]])),
  }))
  const best = selected.reduce((b, d) => (!b || d.mae < b.mae ? d : b), null)

  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="section-title mb-1">Compare Models</h1>
        <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>Side-by-side comparison of prediction runs</p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* ── Run selector ── */}
        <div className="col-span-1">
          <div className="card p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--fg-muted)' }}>Select Runs</span>
              {compareIds.length > 0 && (
                <button onClick={clearCompare}
                  className="text-xs flex items-center gap-1 transition-colors"
                  style={{ color: 'var(--fg-subtle)' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--fg)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--fg-subtle)'}
                >
                  <X size={11} /> Clear
                </button>
              )}
            </div>

            <div className="space-y-2 max-h-[480px] overflow-y-auto">
              {allRuns.length === 0 ? (
                <p className="text-xs py-4 text-center" style={{ color: 'var(--fg-muted)' }}>
                  No completed runs yet.{' '}
                  <Link to="/configure" style={{ color: 'var(--accent)' }}>Start one →</Link>
                </p>
              ) : (
                allRuns.map((run) => {
                  const inSel = compareIds.includes(run.id)
                  const color = MODEL_COLORS[run.model] || c.fg
                  return (
                    <button key={run.id} onClick={() => toggleCompare(run.id)}
                      className="w-full text-left p-3 rounded-xl border transition-all duration-150"
                      style={{
                        borderColor: inSel ? 'var(--accent)' : 'var(--border-mid)',
                        backgroundColor: inSel ? 'var(--accent-bg)' : 'var(--bg-2)',
                      }}>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                        <span className="text-xs font-medium truncate flex-1" style={{ color: 'var(--fg)' }}>
                          {run.name || `Run #${run.id}`}
                        </span>
                        {inSel && <span className="text-xs" style={{ color: 'var(--accent)' }}>✓</span>}
                        <button onClick={(e) => handleDeleteRun(e, run.id)}
                          className="p-0.5 rounded transition-opacity opacity-40 hover:opacity-100"
                          style={{ color: 'var(--fg-subtle)' }}
                          onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                          onMouseLeave={e => e.currentTarget.style.color = 'var(--fg-subtle)'}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                      <div className="text-xs mt-1 ml-4" style={{ color: 'var(--fg-muted)' }}>
                        {run.model?.toUpperCase()} · MAE {run.result?.mae?.toFixed(4)}
                      </div>
                    </button>
                  )
                })
              )}
            </div>
          </div>
        </div>

        {/* ── Comparison panels ── */}
        <div className="col-span-2">
          {selected.length < 2 ? (
            <div className="card p-16 text-center">
              <GitCompare size={40} className="mx-auto mb-4" style={{ color: 'var(--border-mid)' }} />
              <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>Select at least 2 completed runs to compare</p>
            </div>
          ) : (
            <div className="space-y-6">

              {/* Winner banner */}
              {best && (
                <div className="card p-4 flex items-center gap-3"
                     style={{ borderColor: 'rgba(234,179,8,0.3)', backgroundColor: 'rgba(234,179,8,0.06)' }}>
                  <Trophy size={18} style={{ color: '#eab308' }} className="flex-shrink-0" />
                  <div>
                    <span className="text-sm font-medium" style={{ color: 'var(--fg)' }}>Best model: </span>
                    <span className="text-sm font-semibold" style={{ color: '#eab308' }}>
                      {best.name || best.model?.toUpperCase()}
                    </span>
                    <span className="text-xs ml-2" style={{ color: 'var(--fg-muted)' }}>
                      MAE={best.mae?.toFixed(4)} RMSE={best.rmse?.toFixed(4)} MAPE={best.mape?.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Metrics table */}
              <div className="card overflow-hidden">
                <div className="px-5 py-3" style={{ borderBottom: `1px solid ${c.border}` }}>
                  <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--fg-muted)' }}>
                    Metrics Comparison
                  </span>
                </div>
                <table className="w-full">
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${c.border}` }}>
                      {['Run', 'Model', 'MAE', 'RMSE', 'MAPE%', 'Duration'].map((h, i) => (
                        <th key={h} className={`px-5 py-3 text-xs ${i > 1 ? 'text-right' : 'text-left'}`}
                            style={{ color: 'var(--fg-subtle)' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selected.map((d, i) => {
                      const isBest = d.run_id === best?.run_id
                      const rowColor = COLORS[i % COLORS.length]
                      return (
                        <tr key={d.run_id} className="border-b last:border-0"
                            style={{
                              borderColor: c.border,
                              backgroundColor: isBest ? 'rgba(234,179,8,0.05)' : undefined,
                            }}>
                          <td className="px-5 py-3">
                            <div className="flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: rowColor }} />
                              <Link to={`/results/${d.run_id}`} className="text-sm transition-colors"
                                    style={{ color: 'var(--fg)' }}
                                    onMouseEnter={e => e.currentTarget.style.color = 'var(--accent)'}
                                    onMouseLeave={e => e.currentTarget.style.color = 'var(--fg)'}>
                                {d.name || `Run #${d.run_id}`}
                              </Link>
                              {isBest && <Trophy size={12} style={{ color: '#eab308' }} />}
                            </div>
                          </td>
                          <td className="px-5 py-3">
                            <span className="text-xs font-mono" style={{ color: 'var(--fg-muted)' }}>{d.model}</span>
                          </td>
                          {[d.mae?.toFixed(4), d.rmse?.toFixed(4), d.mape?.toFixed(2)].map((val, vi) => (
                            <td key={vi} className="px-5 py-3 text-right font-mono text-sm" style={{ color: rowColor }}>
                              {val}
                            </td>
                          ))}
                          <td className="px-5 py-3 text-right text-xs" style={{ color: 'var(--fg-muted)' }}>
                            {d.duration_seconds != null
                              ? <span className="flex items-center gap-1 justify-end"><Timer size={10}/>{formatDuration(d.duration_seconds)}</span>
                              : '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* Bar chart */}
              <div className="card p-5">
                <div className="text-xs uppercase tracking-wider mb-4" style={{ color: 'var(--fg-muted)' }}>
                  Metric Bar Chart
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={barData} barGap={4}>
                    <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />
                    <XAxis dataKey="metric" tick={{ fill: c.axisText, fontSize: 11 }} />
                    <YAxis tick={{ fill: c.axisText, fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.tooltipText }}
                             labelStyle={{ color: c.tooltipLabel }} />
                    <Legend wrapperStyle={{ fontSize: 11, color: c.legendText }} />
                    {selected.map((d, i) => (
                      <Bar key={d.run_id} dataKey={d.name || `#${d.run_id}`}
                           fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Hyperparams diff */}
              <div className="card p-5">
                <div className="text-xs uppercase tracking-wider mb-4" style={{ color: 'var(--fg-muted)' }}>
                  Hyperparameters
                </div>
                <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${selected.length}, 1fr)` }}>
                  {selected.map((d, i) => (
                    <div key={d.run_id}>
                      <div className="text-xs font-semibold mb-2" style={{ color: COLORS[i % COLORS.length] }}>
                        {d.name || `Run #${d.run_id}`}
                      </div>
                      {Object.entries(d.hyperparams || {}).map(([k, v]) => (
                        <div key={k} className="flex justify-between py-1"
                             style={{ borderBottom: `1px solid ${c.border}` }}>
                          <span className="text-xs font-mono" style={{ color: 'var(--fg-subtle)' }}>{k}</span>
                          <span className="text-xs font-mono" style={{ color: 'var(--fg-muted)' }}>{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  )
}
