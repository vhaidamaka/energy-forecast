import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useStore } from '../store'
import { datasetsApi, runsApi } from '../api/client'
import toast from 'react-hot-toast'
import { StatusBadge } from '../components/StatusBadge'
import { Database, Play, BarChart3, GitCompare, Plus, Zap, TrendingUp, Clock, Timer, Trash2 } from 'lucide-react'
import { formatDuration } from '../utils'
import { formatDistanceToNow } from 'date-fns'

const STAT_COLORS = {
  Datasets:  { text: 'var(--accent)',   icon: Database   },
  'Total Runs': { text: '#f97316',      icon: Play       },
  Completed: { text: '#22c55e',         icon: BarChart3  },
  'Avg MAE': { text: '#eab308',         icon: TrendingUp },
}

export default function Dashboard() {
  const { datasets, runs, setDatasets, setRuns, removeRun } = useStore()

  useEffect(() => {
    datasetsApi.list().then((r) => setDatasets(r.data))
    runsApi.list().then((r) => setRuns(r.data))
  }, [])

  const handleDeleteRun = async (e, id) => {
    e.preventDefault(); e.stopPropagation()
    if (!confirm(`Delete Run #${id}?`)) return
    try { await runsApi.delete(id); removeRun(id); toast.success('Run deleted') }
    catch { toast.error('Delete failed') }
  }

  const doneRuns = runs.filter((r) => r.status === 'done')
  const avgMae = doneRuns.length
    ? (doneRuns.reduce((s, r) => s + (r.result?.mae || 0), 0) / doneRuns.length).toFixed(4)
    : '-'
  const modelCounts = runs.reduce((acc, r) => { acc[r.model] = (acc[r.model] || 0) + 1; return acc }, {})

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold mb-1" style={{ color: 'var(--fg)' }}>
          <span style={{ color: 'var(--accent)' }}>⚡</span> Dashboard
        </h1>
        <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>Energy consumption prediction platform</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Datasets',    value: datasets.length,  },
          { label: 'Total Runs',  value: runs.length,      },
          { label: 'Completed',   value: doneRuns.length,  },
          { label: 'Avg MAE',     value: avgMae,           },
        ].map(({ label, value }) => {
          const { text, icon: Icon } = STAT_COLORS[label]
          return (
            <div key={label} className="card p-5">
              <div className="flex items-start justify-between mb-3">
                <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--fg-muted)' }}>{label}</span>
                <Icon size={16} style={{ color: text }} />
              </div>
              <div className="font-display text-3xl font-bold" style={{ color: text }}>{value}</div>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Quick actions */}
        <div className="col-span-1 space-y-3">
          <h2 className="text-xs uppercase tracking-wider mb-4" style={{ color: 'var(--fg-muted)' }}>Quick Actions</h2>
          {[
            { to: '/datasets', icon: Database,  label: 'Manage Datasets',   desc: `${datasets.length} datasets` },
            { to: '/configure', icon: Plus,     label: 'New Prediction Run', desc: 'Train a model' },
            { to: '/compare',   icon: GitCompare, label: 'Compare Models',  desc: `${doneRuns.length} runs available` },
          ].map(({ to, icon: Icon, label, desc }) => (
            <Link key={to} to={to}>
              <div className="card-hover p-4 flex items-center gap-4 cursor-pointer">
                <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                     style={{ backgroundColor: 'var(--accent-bg)', border: '1px solid var(--accent-dim)' }}>
                  <Icon size={16} style={{ color: 'var(--accent)' }} />
                </div>
                <div>
                  <div className="text-sm font-medium" style={{ color: 'var(--fg)' }}>{label}</div>
                  <div className="text-xs" style={{ color: 'var(--fg-muted)' }}>{desc}</div>
                </div>
              </div>
            </Link>
          ))}

          {/* Model breakdown */}
          {Object.keys(modelCounts).length > 0 && (
            <div className="card p-4 mt-6">
              <div className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--fg-muted)' }}>Runs by Model</div>
              {Object.entries(modelCounts).map(([model, count]) => (
                <div key={model} className="flex items-center justify-between py-1.5">
                  <span className="text-xs font-mono" style={{ color: 'var(--fg-muted)' }}>{model.toUpperCase()}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--border-mid)' }}>
                      <div className="h-full rounded-full" style={{
                        width: `${Math.min(100, (count / runs.length) * 100)}%`,
                        backgroundColor: 'var(--accent)'
                      }} />
                    </div>
                    <span className="text-xs w-4 text-right font-mono" style={{ color: 'var(--accent)' }}>{count}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent runs */}
        <div className="col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs uppercase tracking-wider" style={{ color: 'var(--fg-muted)' }}>Recent Runs</h2>
            <Link to="/compare" className="text-xs transition-colors" style={{ color: 'var(--accent)' }}>Compare all →</Link>
          </div>

          {runs.length === 0 ? (
            <div className="card p-12 text-center">
              <Zap size={32} className="mx-auto mb-3" style={{ color: 'var(--border-mid)' }} />
              <p className="text-sm" style={{ color: 'var(--fg-subtle)' }}>No runs yet.</p>
              <Link to="/configure"><button className="btn-primary mt-4 text-sm">Start your first prediction</button></Link>
            </div>
          ) : (
            <div className="space-y-2">
              {runs.slice(0, 8).map((run) => (
                <Link key={run.id} to={`/results/${run.id}`}>
                  <div className="card-hover p-4 flex items-center gap-4 cursor-pointer group">
                    {/* Model badge */}
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 font-mono text-xs font-bold"
                         style={{ backgroundColor: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--border-mid)' }}>
                      {run.model?.slice(0, 2).toUpperCase()}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium truncate" style={{ color: 'var(--fg)' }}>
                          {run.name || `Run #${run.id}`}
                        </span>
                        <StatusBadge status={run.status} />
                      </div>
                      <div className="text-xs mt-0.5" style={{ color: 'var(--fg-muted)' }}>
                        {run.model?.toUpperCase()} · {run.horizon_days}d horizon
                      </div>
                    </div>

                    <div className="text-right flex-shrink-0 flex items-center gap-2">
                      <div>
                        {run.result && (
                          <div className="text-sm font-mono" style={{ color: 'var(--accent)' }}>
                            MAE {run.result.mae?.toFixed(4)}
                          </div>
                        )}
                        <div className="flex items-center gap-2 mt-0.5 justify-end">
                          <div className="flex items-center gap-1 text-xs" style={{ color: 'var(--fg-subtle)' }}>
                            <Clock size={10} />
                            {run.started_at ? formatDistanceToNow(new Date(run.started_at), { addSuffix: true }) : 'pending'}
                          </div>
                          {run.duration_seconds != null && (
                            <div className="flex items-center gap-1 text-xs" style={{ color: 'var(--fg-subtle)' }}>
                              <Timer size={10} />{formatDuration(run.duration_seconds)}
                            </div>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => handleDeleteRun(e, run.id)}
                        className="opacity-0 group-hover:opacity-100 p-1.5 rounded transition-all"
                        style={{ color: 'var(--fg-subtle)' }}
                        title="Delete run"
                        onMouseEnter={e => e.currentTarget.style.color='#ef4444'}
                        onMouseLeave={e => e.currentTarget.style.color='var(--fg-subtle)'}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
