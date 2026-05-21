import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { runsApi } from '../api/client'
import { useStore } from '../store'
import { StatusBadge } from '../components/StatusBadge'
import { Terminal, ChevronRight, BarChart3, Timer } from 'lucide-react'
import { useThemeColors } from '../hooks/useThemeColors'
import { formatDuration } from '../utils'
import clsx from 'clsx'

export default function Training() {
  const { runId } = useParams()
  const c = useThemeColors()
  const navigate = useNavigate()
  const { updateRun } = useStore()

  const [run, setRun] = useState(null)
  const [logs, setLogs] = useState([])
  const [done, setDone] = useState(false)
  const [connecting, setConnecting] = useState(true)
  const [epochData, setEpochData] = useState([])
  const logsEndRef = useRef(null)
  const esRef = useRef(null)

  // Fetch run info
  useEffect(() => {
    runsApi.get(runId).then((r) => setRun(r.data))
  }, [runId])

  // Connect SSE log stream
  useEffect(() => {
    const es = new EventSource(`/api/runs/${runId}/logs`)
    esRef.current = es

    // Mark as connected when first message arrives
    es.onopen = () => { setConnecting(false) }

    es.onmessage = (e) => {
      setConnecting(false)
      const msg = e.data
      setLogs((prev) => [...prev, msg])

      // Parse epoch lines to build mini loss chart
      const epochMatch = msg.match(/Epoch (\d+): loss=([\d.]+).*val_loss=([\d.]+)/)
      if (epochMatch) {
        setEpochData((prev) => [
          ...prev,
          { epoch: parseInt(epochMatch[1]), loss: parseFloat(epochMatch[2]), val_loss: parseFloat(epochMatch[3]) },
        ])
      }
    }

    es.addEventListener('done', () => {
      setDone(true)
      es.close()
      // Refresh run to get results
      runsApi.get(runId).then((r) => {
        setRun(r.data)
        updateRun(r.data.id, r.data)
      })
    })

    es.addEventListener('timeout', () => {
      setLogs((p) => [...p, '⚠ Stream timed out'])
      setDone(true)
      es.close()
    })

    es.onerror = () => {
      // SSE connection failed (404 race, network error, nginx timeout)
      // Fall back to polling so the UI still updates when training finishes
      es.close()
      const poll = setInterval(() => {
        runsApi.get(runId).then((r) => {
          setRun(r.data)
          updateRun(r.data.id, r.data)
          if (['done', 'failed'].includes(r.data.status)) {
            setDone(true)
            clearInterval(poll)
          }
        })
      }, 2000)
    }

    return () => { es.close() }
  }, [runId])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const maxLoss = epochData.length ? Math.max(...epochData.map((d) => d.loss)) : 1
  const lastEpoch = epochData[epochData.length - 1]

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="section-title mb-1">
            Training Run #{runId}
          </h1>
          {run && (
            <div className="flex items-center gap-3 text-sm">
              <span className="font-mono text-accent">{run.model?.toUpperCase()}</span>
              <span>·</span>
              <span>{run.horizon_days}d horizon</span>
              <span>·</span>
              <StatusBadge status={run?.status || 'training'} />
            </div>
          )}
        </div>
        {done && run?.status === 'done' && (
          <button
            className="btn-primary flex items-center gap-2"
            onClick={() => navigate(`/results/${runId}`)}
          >
            View Results <ChevronRight size={16} />
          </button>
        )}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Log terminal */}
        <div className="col-span-2">
          <div className="card overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: '1px solid var(--border)', backgroundColor: 'var(--bg-0)' }}>
              <Terminal size={14} className="text-accent" />
              <span className="text-xs font-mono">training.log</span>
              {!done && (
                <span className="ml-auto flex items-center gap-1.5 text-xs text-accent">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                  live
                </span>
              )}
            </div>
            <div className="h-[480px] overflow-y-auto p-4 font-mono text-xs space-y-0.5" style={{ backgroundColor: "var(--bg-0)" }}>
              {logs.length === 0 && !done && !connecting && (
                <span style={{ color: 'var(--fg-subtle)' }}>No log output yet...</span>
              )}
            {logs.length === 0 && !done && connecting && (
                <div className=" flex items-center gap-2">
                  <span className="animate-pulse">▋</span> Connecting to training process...
                </div>
              )}
              {logs.map((line, i) => (
                <div
                  key={i}
                  className={clsx(
                    'leading-5',
                    line.startsWith('✓') || line.startsWith('[') ? '' : '',
                    line.includes('MAE') || line.includes('RMSE') ? 'text-accent' : '',
                    line.includes('✗') || line.includes('failed') ? 'text-danger' : '',
                    line.includes('✓') ? 'text-success' : '',
                    line.startsWith('Epoch') ? '' : '',
                  )}
                >
                  <span className=" select-none mr-2">{String(i + 1).padStart(3, ' ')}</span>
                  {line}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          {/* Run config */}
          {run && (
            <div className="card p-4">
              <div className="text-xs uppercase tracking-wider mb-3">Configuration</div>
              {Object.entries(run.hyperparams || {}).map(([k, v]) => (
                <div key={k} className="flex justify-between py-1 last:border-0" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-xs font-mono">{k}</span>
                  <span className="text-xs font-mono">{String(v)}</span>
                </div>
              ))}
            </div>
          )}

          {/* Live loss chart */}
          {epochData.length > 1 && (
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 size={13} className="text-accent" />
                <span className="text-xs uppercase tracking-wider">Training Loss</span>
              </div>
              <MiniLossChart data={epochData} maxLoss={maxLoss} />
              {lastEpoch && (
                <div className="flex gap-4 mt-3">
                  <div>
                    <div className="text-xs">Loss</div>
                    <div className="text-sm font-mono text-accent">{lastEpoch.loss.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-xs">Val Loss</div>
                    <div className="text-sm font-mono text-warn">{lastEpoch.val_loss.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-xs">Epoch</div>
                    <div className="text-sm font-mono">{lastEpoch.epoch}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Final metrics */}
          {run?.result && (
            <div className="card p-4" style={{ borderColor: "rgba(34,197,94,0.2)" }}>
              <div className="text-xs text-success uppercase tracking-wider mb-3">✓ Results</div>
              {[
                { label: 'MAE', val: run.result.mae?.toFixed(4) },
                { label: 'RMSE', val: run.result.rmse?.toFixed(4) },
                { label: 'MAPE', val: `${run.result.mape?.toFixed(2)}%` },
                ...(run.duration_seconds != null
                  ? [{ label: 'Duration', val: formatDuration(run.duration_seconds), icon: true }]
                  : []),
              ].map(({ label, val, icon }) => (
                <div key={label} className="flex justify-between py-2 last:border-0" style={{ borderBottom: "1px solid var(--border)" }}>
                  <span className="text-xs flex items-center gap-1">
                    {icon && <Timer size={10} />}{label}
                  </span>
                  <span className="text-sm font-mono text-accent">{val}</span>
                </div>
              ))}
            </div>
          )}

          {run?.status === 'failed' && (
            <div className="card p-4 border-danger/20 bg-danger/5">
              <div className="text-xs text-danger mb-2">✗ Training Failed</div>
              <p className="text-xs font-mono break-all">{run.error_message}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function MiniLossChart({ data, maxLoss }) {
  const { accentLine, warnLine } = useThemeColors()
  const W = 220
  const H = 60
  const pad = 4

  const toX = (i) => pad + (i / Math.max(1, data.length - 1)) * (W - pad * 2)
  const toY = (v) => H - pad - (v / maxLoss) * (H - pad * 2)

  const linePath = (key) =>
    data.map((d, i) => `${i === 0 ? 'M' : 'L'}${toX(i)},${toY(d[key])}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <path d={linePath('loss')} fill="none" stroke={accentLine} strokeWidth="1.5" strokeLinecap="round" />
      <path d={linePath('val_loss')} fill="none" stroke={warnLine} strokeWidth="1.5" strokeLinecap="round" strokeDasharray="4 2" />
    </svg>
  )
}
