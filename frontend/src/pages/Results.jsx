import { useEffect, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { runsApi, exportApi } from '../api/client'
import { useStore } from '../store'
import { StatusBadge } from '../components/StatusBadge'
import { useThemeColors } from '../hooks/useThemeColors'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area, ReferenceLine, ComposedChart,
  Brush, ReferenceArea,
} from 'recharts'
import { Download, GitCompare, ArrowLeft, TrendingUp, TrendingDown, Minus, Timer, Trash2, Info } from 'lucide-react'
import { formatDuration } from '../utils'
import toast from 'react-hot-toast'
import clsx from 'clsx'

export default function Results() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const { compareIds, toggleCompare, removeRun } = useStore()
  const c = useThemeColors()

  const [run, setRun] = useState(null)
  const [tab, setTab] = useState('validation')

  useEffect(() => { runsApi.get(runId).then((r) => setRun(r.data)) }, [runId])

  const handleDelete = async () => {
    if (!confirm(`Delete Run #${runId} and all its results?`)) return
    try { await runsApi.delete(runId); removeRun(parseInt(runId)); navigate('/') }
    catch { toast.error('Delete failed') }
  }

  if (!run) return <div className="p-8" style={{ color: 'var(--fg-muted)' }}>Loading...</div>
  if (!run.result) return (
    <div className="p-8">
      <p style={{ color: 'var(--fg-muted)' }}>No results yet — status: <StatusBadge status={run.status} /></p>
      {run.status === 'training' && (
        <Link to={`/training/${run.id}`} className="text-sm mt-2 inline-block" style={{ color: 'var(--accent)' }}>
          → Watch training live
        </Link>
      )}
    </div>
  )

  // ── Data preparation ──────────────────────────────────────────────────────

  // Test set: align actual ground truth with model predictions by timestamp
  const actualMap = {}
  ;(run.result.actual_json || []).forEach(d => {
    actualMap[d.timestamp] = d.actual
  })
  const predMap = {}
  ;(run.result.test_predicted_json || []).forEach(d => {
    predMap[d.timestamp] = d.predicted
  })
  // Build aligned validation set (both actual AND predicted, same timestamps)
  const validationData = (run.result.actual_json || []).map(d => ({
    ts: d.timestamp?.slice(0, 16).replace('T', ' '),
    actual: d.actual,
    predicted: predMap[d.timestamp] ?? null,
  })).filter(d => d.actual != null)

  // Future forecast (beyond training data)
  const forecastData = (run.result.forecast_json || []).map(d => ({
    ts: d.timestamp?.slice(0, 16).replace('T', ' '),
    predicted: d.predicted,
  }))

  // History + forecast overlay: validation actuals followed by future forecast
  const overlayData = [
    ...validationData.map(d => ({ ts: d.ts, actual: d.actual, predicted: undefined })),
    ...forecastData.map(d => ({ ts: d.ts, actual: undefined, predicted: d.predicted })),
  ]
  const lastActualTs = validationData[validationData.length - 1]?.ts

  const historyData = run.result.training_history || []
  const inCompare = compareIds.includes(run.id)

  const mapeQuality = run.result.mape < 5  ? { label: 'Excellent', color: '#22c55e' }
    : run.result.mape < 10 ? { label: 'Good',      color: c.accentLine }
    : run.result.mape < 20 ? { label: 'Fair',      color: '#eab308' }
    :                        { label: 'Poor',       color: '#ef4444' }

  // Shared chart props
  const tooltipStyle = {
    contentStyle: { background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.tooltipText },
    labelStyle: { color: c.tooltipLabel, fontSize: 11 },
  }
  const legendStyle = { wrapperStyle: { fontSize: 12, color: c.legendText } }

  // Tabs — show validation tab only if test_predicted_json exists
  const hasPredVsActual = (run.result.test_predicted_json || []).length > 0
  const tabs = [
    { id: 'validation', label: 'Predicted vs Actual', desc: 'Test set: model predictions vs ground truth' },
    { id: 'forecast',   label: 'Future Forecast',     desc: `${forecastData.length} steps beyond training data` },
    { id: 'overlay',    label: 'History + Forecast',  desc: 'Validation window then future predictions' },
    ...(historyData.length ? [{ id: 'history', label: 'Training Loss', desc: `${historyData.length} epochs` }] : []),
  ]

  return (
    <div className="p-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <button onClick={() => navigate(-1)}
            className="flex items-center gap-1.5 text-sm mb-3 transition-colors"
            style={{ color: 'var(--fg-muted)' }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--fg)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--fg-muted)'}>
            <ArrowLeft size={14} /> Back
          </button>
          <h1 className="section-title mb-1">{run.name || `Run #${run.id}`}</h1>
          <div className="flex items-center gap-3 text-sm flex-wrap" style={{ color: 'var(--fg-muted)' }}>
            <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>{run.model?.toUpperCase()}</span>
            <span>·</span><span>{run.horizon_days}d horizon</span>
            <span>·</span><span>{run.dataset?.name}</span>
            {run.duration_seconds != null && (
              <><span>·</span>
                <span className="flex items-center gap-1"><Timer size={12} />{formatDuration(run.duration_seconds)}</span>
              </>
            )}
            <StatusBadge status={run.status} />
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => toggleCompare(run.id)} className="btn-ghost text-xs flex items-center gap-1.5"
            style={inCompare ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : {}}>
            <GitCompare size={13} />{inCompare ? 'In Compare' : 'Add to Compare'}
          </button>
          <a href={exportApi.csvUrl(run.id)} download>
            <button className="btn-ghost text-xs flex items-center gap-1.5"><Download size={13}/> CSV</button>
          </a>
          <a href={exportApi.excelUrl(run.id)} download>
            <button className="btn-ghost text-xs flex items-center gap-1.5"><Download size={13}/> Excel</button>
          </a>
          <button onClick={handleDelete} className="btn-danger text-xs flex items-center gap-1.5">
            <Trash2 size={13}/> Delete
          </button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {[
          { label: 'MAE',  value: run.result.mae?.toFixed(4),          desc: 'Mean Absolute Error — test set',       icon: Minus,        color: c.accentLine },
          { label: 'RMSE', value: run.result.rmse?.toFixed(4),         desc: 'Root Mean Squared Error — test set',   icon: TrendingUp,   color: c.accentLine },
          { label: 'MAPE', value: `${run.result.mape?.toFixed(2)}%`,   desc: `Abs % Error · ${mapeQuality.label}`,   icon: TrendingDown, color: mapeQuality.color },
        ].map(({ label, value, desc, icon: Icon, color }) => (
          <div key={label} className="card p-5 flex items-center gap-5">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                 style={{ backgroundColor: c.accentBg, border: `1px solid ${c.iconBorder}` }}>
              <Icon size={20} style={{ color: c.iconColor }} />
            </div>
            <div>
              <div className="text-xs mb-0.5" style={{ color: 'var(--fg-muted)' }}>{desc}</div>
              <div className="font-display text-2xl font-bold" style={{ color }}>{value}</div>
              <div className="text-xs" style={{ color: 'var(--fg-subtle)' }}>{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="card overflow-hidden">
        {/* Tab bar */}
        <div className="flex" style={{ borderBottom: `1px solid ${c.border}` }}>
          {tabs.map(({ id, label, desc }) => (
            <button key={id} onClick={() => setTab(id)}
              className="px-5 py-3 text-sm transition-colors whitespace-nowrap"
              style={{
                color: tab === id ? 'var(--accent)' : 'var(--fg-muted)',
                borderBottom: tab === id ? `2px solid var(--accent)` : '2px solid transparent',
                fontWeight: tab === id ? 600 : 400,
              }}>
              {label}
            </button>
          ))}
        </div>

        <div className="p-6">

          {/* ── Predicted vs Actual (test set) ── */}
          {tab === 'validation' && (
            <>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm font-medium" style={{ color: 'var(--fg)' }}>
                    Test Set: Predicted vs Actual
                  </p>
                  <p className="text-xs mt-0.5" style={{ color: 'var(--fg-muted)' }}>
                    {validationData.filter(d => d.predicted != null).length} paired points ·
                    20% holdout — model never saw this data during training
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {/* Legend */}
                  <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--fg-muted)' }}>
                    <span className="flex items-center gap-1.5">
                      <span className="inline-block w-6 h-0.5 rounded" style={{ backgroundColor: c.energyLine }} />
                      Actual
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="inline-block w-6 h-0.5 rounded" style={{ backgroundColor: c.accentLine }} />
                      Predicted
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="inline-block w-6 h-2 rounded opacity-30" style={{ backgroundColor: c.accentLine }} />
                      ±MAE band
                    </span>
                  </div>
                  {!hasPredVsActual && (
                    <div className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg"
                         style={{ backgroundColor: 'rgba(234,179,8,0.08)', color: '#eab308', border: '1px solid rgba(234,179,8,0.25)' }}>
                      <Info size={12} />
                      Re-run to see predictions
                    </div>
                  )}
                </div>
              </div>
              <ValidationChart data={validationData} mae={run.result.mae} c={c} tooltipStyle={tooltipStyle} />
              <p className="text-xs mt-3" style={{ color: 'var(--fg-subtle)' }}>
                Shaded band = ±MAE around prediction. Use the brush below the chart to zoom into any window.
              </p>
            </>
          )}

          {/* ── Future Forecast ── */}
          {tab === 'forecast' && (
            <>
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--fg)' }}>Future Forecast</p>
              <p className="text-xs mb-4" style={{ color: 'var(--fg-muted)' }}>
                {forecastData.length} predicted steps · {run.horizon_days}-day horizon beyond the dataset end
              </p>
              <ResponsiveContainer width="100%" height={360}>
                <AreaChart data={forecastData}>
                  <defs>
                    <linearGradient id="fg_grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={c.accentLine} stopOpacity={0.2}/>
                      <stop offset="95%" stopColor={c.accentLine} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />
                  <XAxis dataKey="ts" tick={{ fill: c.axisText, fontSize: 11 }}
                         tickFormatter={v => v?.slice(5, 16)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: c.axisText, fontSize: 11 }} />
                  <Tooltip {...tooltipStyle} itemStyle={{ color: c.accentLine }}
                           formatter={v => [v?.toFixed(2) + ' kWh', 'Forecast']} />
                  <Area type="monotone" dataKey="predicted" name="Forecast (kW)"
                        stroke={c.accentLine} strokeWidth={2} fill="url(#fg_grad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </>
          )}

          {/* ── History + Forecast overlay ── */}
          {tab === 'overlay' && (
            <>
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--fg)' }}>History + Forecast</p>
              <div className="flex items-center gap-4 mb-4">
                <p className="text-xs" style={{ color: 'var(--fg-muted)' }}>
                  Actual: {validationData.length} pts (test set) · Forecast: {forecastData.length} future pts
                </p>
              </div>
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={overlayData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />
                  <XAxis dataKey="ts" tick={{ fill: c.axisText, fontSize: 11 }}
                         tickFormatter={v => v?.slice(5)} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: c.axisText, fontSize: 11 }} />
                  <Tooltip {...tooltipStyle}
                    formatter={(val, name) => [val?.toFixed(2) + ' kWh', name === 'actual' ? 'Actual (test set)' : 'Forecast']} />
                  <Legend {...legendStyle} />
                  {lastActualTs && (
                    <ReferenceLine x={lastActualTs} stroke={c.refLine} strokeDasharray="4 2"
                      label={{ value: 'forecast →', fill: c.fgSubtle, fontSize: 10, position: 'insideTopRight' }} />
                  )}
                  <Line type="monotone" dataKey="actual"    name="Actual (kW)"   stroke={c.energyLine} strokeWidth={1.5} dot={false} connectNulls={false} />
                  <Line type="monotone" dataKey="predicted" name="Forecast (kW)" stroke={c.accentLine} strokeWidth={2}   dot={false} connectNulls={false} strokeDasharray="5 3" />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

          {/* ── Training Loss ── */}
          {tab === 'history' && historyData.length > 0 && (
            <>
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--fg)' }}>Training Loss</p>
              <p className="text-xs mb-4" style={{ color: 'var(--fg-muted)' }}>{historyData.length} epochs</p>
              <ResponsiveContainer width="100%" height={360}>
                <LineChart data={historyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />
                  <XAxis dataKey="epoch" tick={{ fill: c.axisText, fontSize: 11 }} />
                  <YAxis tick={{ fill: c.axisText, fontSize: 11 }} />
                  <Tooltip {...tooltipStyle} />
                  <Legend {...legendStyle} />
                  <Line type="monotone" dataKey="loss"     name="Train Loss" stroke={c.accentLine} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Val Loss"   stroke={c.warnLine}   strokeWidth={2} dot={false} strokeDasharray="5 3" />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}

        </div>
      </div>

      {compareIds.length >= 2 && (
        <div className="mt-6 flex items-center justify-between p-4 card"
             style={{ borderColor: 'var(--accent-dim)', backgroundColor: 'var(--accent-bg)' }}>
          <span className="text-sm" style={{ color: 'var(--fg-muted)' }}>{compareIds.length} runs selected</span>
          <Link to="/compare">
            <button className="btn-primary text-sm flex items-center gap-2">
              <GitCompare size={14}/> Compare Now
            </button>
          </Link>
        </div>
      )}
    </div>
  )
}


// ── ValidationChart ───────────────────────────────────────────────────────────
// Separate component so it can use its own hooks without scope issues.
// Features:
//   • Actual vs Predicted lines with clear visual distinction
//   • ±MAE shaded confidence band around the prediction line
//   • Brush/zoom control at the bottom
//   • Smart x-axis: shows date only (not time) to avoid clutter
//   • Only plots rows where both actual AND predicted exist (no flat-line tails)
//   • Error area opacity indicates model accuracy visually

import { useThemeColors as _useTC } from '../hooks/useThemeColors'

function ValidationChart({ data, mae, c: cProp, tooltipStyle }) {
  const c = cProp || _useTC()

  // Only include rows that have BOTH actual and predicted (no trailing flat lines)
  const paired = data.filter(d => d.actual != null && d.predicted != null)

  // Build chart data: add upper/lower MAE band columns
  const chartData = paired.map(d => ({
    ts:       d.ts,
    actual:   d.actual,
    predicted: d.predicted,
    bandHigh: d.predicted + (mae || 0),
    bandLow:  Math.max(0, d.predicted - (mae || 0)),
    error:    Math.abs(d.actual - d.predicted),
  }))

  // Smart x-axis tick formatter: show date only, skip time to reduce clutter
  const fmtTick = (v) => {
    if (!v) return ''
    // v is like "2024-12-16 03:00" — show "12-16" only
    return v.slice(5, 10)
  }

  // Tooltip: show actual, predicted, and absolute error
  const tooltipFmt = (val, name) => {
    if (name === 'actual')    return [val?.toFixed(1) + ' kWh', 'Actual']
    if (name === 'predicted') return [val?.toFixed(1) + ' kWh', 'Predicted']
    if (name === 'error')     return [val?.toFixed(1) + ' kWh', 'Error |A-P|']
    return [val, name]
  }

  // Brush style for theme
  const brushStyle = {
    stroke: c.borderMid,
    fill: c.bg2,
  }

  return (
    <div>
      {/* Main chart with MAE band */}
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
          <defs>
            {/* Gradient fill for MAE confidence band */}
            <linearGradient id="mae_band" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={c.accentLine} stopOpacity={0.18} />
              <stop offset="100%" stopColor={c.accentLine} stopOpacity={0.06} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />

          <XAxis
            dataKey="ts"
            tick={{ fill: c.axisText, fontSize: 11 }}
            tickFormatter={fmtTick}
            minTickGap={40}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: c.axisText, fontSize: 11 }}
            width={52}
            tickFormatter={v => v >= 1000 ? (v/1000).toFixed(1)+'k' : v}
          />

          <Tooltip
            contentStyle={{
              background: c.tooltipBg,
              border: `1px solid ${c.tooltipBorder}`,
              borderRadius: 8,
              color: c.tooltipText,
              fontSize: 12,
            }}
            labelStyle={{ color: c.tooltipLabel, fontSize: 11, marginBottom: 4 }}
            formatter={tooltipFmt}
            labelFormatter={v => v?.slice(0, 16)}
          />

          {/* ±MAE band: rendered as area between bandHigh and bandLow */}
          {/* Upper bound of band */}
          <Area
            type="monotone"
            dataKey="bandHigh"
            stroke="none"
            fill="url(#mae_band)"
            dot={false}
            legendType="none"
            tooltipType="none"
            isAnimationActive={false}
            activeDot={false}
          />
          {/* Lower bound — fills down to 0, then we subtract bandLow visually */}
          <Area
            type="monotone"
            dataKey="bandLow"
            stroke="none"
            fill={c.bg1}
            dot={false}
            legendType="none"
            tooltipType="none"
            isAnimationActive={false}
            activeDot={false}
          />

          {/* Actual energy (bold orange) */}
          <Line
            type="monotone"
            dataKey="actual"
            name="actual"
            stroke={c.energyLine}
            strokeWidth={2}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />

          {/* Predicted energy (cyan dashed) */}
          <Line
            type="monotone"
            dataKey="predicted"
            name="predicted"
            stroke={c.accentLine}
            strokeWidth={1.5}
            strokeDasharray="6 3"
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />

          {/* Brush: allows zooming into a time window */}
          <Brush
            dataKey="ts"
            height={28}
            stroke={c.borderMid}
            fill={c.bg2}
            travellerWidth={6}
            tickFormatter={fmtTick}
            startIndex={0}
            endIndex={Math.min(chartData.length - 1, 168)}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
