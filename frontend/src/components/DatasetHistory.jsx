import { useState, useEffect, useMemo } from 'react'
import { datasetsApi } from '../api/client'
import { useThemeColors } from '../hooks/useThemeColors'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Brush
} from 'recharts'
import { BarChart2, X, ChevronDown } from 'lucide-react'

const RANGE_PRESETS = [
  { label: '1 week',   days: 7   },
  { label: '2 weeks',  days: 14  },
  { label: '1 month',  days: 30  },
  { label: '3 months', days: 90  },
  { label: '6 months', days: 180 },
  { label: 'All',      days: null },
]

export default function DatasetHistory({ datasetId, datasetName, onClose }) {
  const c = useThemeColors()
  const [allData, setAllData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedPreset, setSelectedPreset] = useState('1 month')
  const [customFrom, setCustomFrom] = useState('')
  const [customTo, setCustomTo] = useState('')
  const [useCustom, setUseCustom] = useState(false)

  // Load all data once (up to 2000 rows for preview)
  useEffect(() => {
    setLoading(true)
    setError(null)
    datasetsApi.preview(datasetId, 2000)
      .then((res) => {
        const rows = res.data.rows || []
        // Find timestamp column
        const tsCol = res.data.columns?.find(c =>
          ['timestamp', '_ts', 'datetime', 'date', 'time'].includes(c.toLowerCase())
        ) || res.data.columns?.[0]
        const energyCol = res.data.columns?.find(c => c === 'energy') || 'energy'

        const parsed = rows
          .map(r => {
            const ts = new Date(r[tsCol])
            return { ts, timestamp: r[tsCol], energy: parseFloat(r[energyCol]) || 0 }
          })
          .filter(r => !isNaN(r.ts))
          .sort((a, b) => a.ts - b.ts)

        setAllData(parsed)
        // Default custom range to full span
        if (parsed.length > 0) {
          setCustomFrom(parsed[0].timestamp?.slice(0, 10) || '')
          setCustomTo(parsed[parsed.length - 1].timestamp?.slice(0, 10) || '')
        }
      })
      .catch(() => setError('Failed to load dataset history'))
      .finally(() => setLoading(false))
  }, [datasetId])

  // Filter data by selected range
  const filteredData = useMemo(() => {
    if (allData.length === 0) return []

    if (useCustom && customFrom && customTo) {
      const from = new Date(customFrom)
      const to = new Date(customTo + 'T23:59:59')
      return allData.filter(r => r.ts >= from && r.ts <= to)
    }

    const preset = RANGE_PRESETS.find(p => p.label === selectedPreset)
    if (!preset || preset.days === null) return allData

    const latest = allData[allData.length - 1].ts
    const cutoff = new Date(latest.getTime() - preset.days * 86400000)
    return allData.filter(r => r.ts >= cutoff)
  }, [allData, selectedPreset, useCustom, customFrom, customTo])

  // Downsample for chart performance (max 500 points)
  const chartData = useMemo(() => {
    if (filteredData.length <= 500) return filteredData
    const step = Math.ceil(filteredData.length / 500)
    return filteredData.filter((_, i) => i % step === 0)
  }, [filteredData])

  const stats = useMemo(() => {
    if (chartData.length === 0) return null
    const vals = chartData.map(d => d.energy)
    return {
      min:  Math.min(...vals).toFixed(1),
      max:  Math.max(...vals).toFixed(1),
      mean: (vals.reduce((a,b) => a+b, 0) / vals.length).toFixed(1),
      pts:  filteredData.length,
    }
  }, [chartData, filteredData])

  return (
    <div className="border-t overflow-hidden" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--bg-0)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3"
           style={{ borderBottom: `1px solid var(--border)`, backgroundColor: 'var(--bg-2)' }}>
        <div className="flex items-center gap-2">
          <BarChart2 size={15} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--fg)' }}>
            Historical Data — {datasetName}
          </span>
          {stats && (
            <div className="flex items-center gap-4 ml-4">
              {[
                { label: 'Points', val: stats.pts.toLocaleString() },
                { label: 'Min',    val: stats.min },
                { label: 'Max',    val: stats.max },
                { label: 'Avg',    val: stats.mean },
              ].map(({ label, val }) => (
                <span key={label} className="text-xs" style={{ color: 'var(--fg-muted)' }}>
                  <span style={{ color: 'var(--fg-subtle)' }}>{label}: </span>
                  <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{val}</span>
                </span>
              ))}
            </div>
          )}
        </div>
        <button onClick={onClose} style={{ color: 'var(--fg-subtle)' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--fg)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--fg-subtle)'}>
          <X size={16} />
        </button>
      </div>

      {/* Range controls */}
      <div className="flex items-center gap-3 px-5 py-3 flex-wrap"
           style={{ borderBottom: `1px solid var(--border)` }}>
        {/* Preset buttons */}
        <div className="flex items-center gap-1 flex-wrap">
          {RANGE_PRESETS.map(preset => (
            <button key={preset.label}
              onClick={() => { setSelectedPreset(preset.label); setUseCustom(false) }}
              className="text-xs px-3 py-1.5 rounded-lg border transition-all duration-150"
              style={{
                borderColor: !useCustom && selectedPreset === preset.label ? 'var(--accent)' : 'var(--border-mid)',
                backgroundColor: !useCustom && selectedPreset === preset.label ? 'var(--accent-bg)' : 'transparent',
                color: !useCustom && selectedPreset === preset.label ? 'var(--accent)' : 'var(--fg-muted)',
                fontWeight: !useCustom && selectedPreset === preset.label ? 600 : 400,
              }}>
              {preset.label}
            </button>
          ))}
        </div>

        {/* Separator */}
        <div className="w-px h-5 mx-1" style={{ backgroundColor: 'var(--border-mid)' }} />

        {/* Custom range */}
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--fg-muted)' }}>Custom:</span>
          <input type="date" className="input text-xs py-1 px-2 w-36"
                 value={customFrom} onChange={e => { setCustomFrom(e.target.value); setUseCustom(true) }} />
          <span className="text-xs" style={{ color: 'var(--fg-subtle)' }}>→</span>
          <input type="date" className="input text-xs py-1 px-2 w-36"
                 value={customTo} onChange={e => { setCustomTo(e.target.value); setUseCustom(true) }} />
          {useCustom && (
            <button className="text-xs px-2 py-1 rounded"
                    style={{ color: 'var(--fg-subtle)' }}
                    onClick={() => setUseCustom(false)}>
              ✕
            </button>
          )}
        </div>

        <span className="text-xs ml-auto" style={{ color: 'var(--fg-subtle)' }}>
          {chartData.length < filteredData.length
            ? `Showing ${chartData.length} of ${filteredData.length} pts (downsampled)`
            : `${filteredData.length} data points`}
        </span>
      </div>

      {/* Chart */}
      <div className="px-5 py-4" style={{ height: 280 }}>
        {loading ? (
          <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--fg-muted)' }}>
            Loading data...
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-sm" style={{ color: '#ef4444' }}>
            {error}
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-sm" style={{ color: 'var(--fg-muted)' }}>
            No data in selected range
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <defs>
                <linearGradient id={`dh_${datasetId}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={c.accentLine} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={c.accentLine} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={c.gridStroke} />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: c.axisText, fontSize: 10 }}
                tickFormatter={v => v?.slice(5, 16)}
                interval="preserveStartEnd"
                minTickGap={60}
              />
              <YAxis tick={{ fill: c.axisText, fontSize: 10 }} width={52} />
              <Tooltip
                contentStyle={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.tooltipText, fontSize: 12 }}
                labelStyle={{ color: c.tooltipLabel, fontSize: 11 }}
                itemStyle={{ color: c.accentLine }}
                formatter={(v) => [v?.toFixed(2) + ' kWh', 'Energy']}
                labelFormatter={v => v?.slice(0, 16)}
              />
              <Area
                type="monotone"
                dataKey="energy"
                stroke={c.accentLine}
                strokeWidth={1.5}
                fill={`url(#dh_${datasetId})`}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
