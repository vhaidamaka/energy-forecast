import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import { datasetsApi, runsApi, modelsApi } from '../api/client'
import toast from 'react-hot-toast'
import { ChevronDown, ChevronUp, Play } from 'lucide-react'
import clsx from 'clsx'

// Model metadata — using hex colors (not Tailwind) so they work in both themes
const MODEL_META = {
  nbeats:       { label: 'N-BEATS', color: '#06b6d4', desc: 'Doubly residual stacked blocks with trend + seasonality + generic Fourier/polynomial basis. State-of-the-art on M4/M5, interpretable, no recurrence.' },
  arima:        { label: 'ARIMA / SARIMAX', color: '#eab308', desc: 'Classical statistical model. Fast, interpretable. Best for univariate data with clear seasonality.' },
  lstm:         { label: 'LSTM',            color: 'var(--accent)', desc: 'Long Short-Term Memory neural network. Learns complex temporal patterns from multivariate data.' },
  bilstm:       { label: 'BiLSTM',          color: '#f97316', desc: 'Bidirectional LSTM. Processes sequence in both directions — often more accurate than LSTM.' },
  gru:          { label: 'GRU',             color: '#a855f7', desc: 'Gated Recurrent Unit. Faster to train than LSTM with fewer parameters — great for smaller datasets.' },
  bigru:        { label: 'BiGRU',           color: '#ec4899', desc: 'Bidirectional GRU. Combines GRU speed with bidirectional context — strong all-round performer.' },
  wavelet_lstm: { label: 'Wavelet + LSTM',  color: '#22c55e', desc: 'DWT decomposes energy signal into frequency components (db4/haar/sym8) then feeds LSTM — best for noisy data.' },
  transformer:  { label: 'Transformer',     color: '#38bdf8', desc: 'Multi-head self-attention encoder with positional encoding. Captures long-range dependencies without recurrence.' },
  tft:          { label: 'TFT',             color: '#fb7185', desc: 'Temporal Fusion Transformer — variable selection, LSTM encoder, static enrichment + interpretable attention.' },
}

export default function Configure() {
  const navigate = useNavigate()
  const { datasets, setDatasets, addRun } = useStore()

  const [defaults, setDefaults] = useState({})
  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedModel, setSelectedModel] = useState('lstm')
  const [horizon, setHorizon] = useState(7)
  const [runName, setRunName] = useState('')
  const [params, setParams] = useState({})
  const [expanded, setExpanded] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    datasetsApi.list().then((r) => setDatasets(r.data))
    modelsApi.defaults().then((r) => { setDefaults(r.data); setParams(r.data['lstm'] || {}) })
  }, [])

  const handleModelChange = (model) => { setSelectedModel(model); setParams(defaults[model] || {}) }
  const handleParamChange = (key, value) => setParams((p) => ({ ...p, [key]: value }))

  const handleSubmit = async () => {
    if (!selectedDataset) return toast.error('Select a dataset first')
    setSubmitting(true)
    try {
      const res = await runsApi.create({ dataset_id: parseInt(selectedDataset), model: selectedModel, hyperparams: params, horizon_days: horizon, name: runName || undefined })
      const run = res.data; addRun(run)
      await runsApi.start(run.id)
      toast.success(`Run #${run.id} started!`)
      navigate(`/training/${run.id}`)
    } catch (e) { toast.error(e.response?.data?.detail || 'Failed to start run') }
    finally { setSubmitting(false) }
  }

  return (
    <div className="p-8 animate-fade-in max-w-3xl">
      <div className="mb-8">
        <h1 className="section-title mb-1">Configure Run</h1>
        <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>Select a dataset, choose a model, and tune hyperparameters</p>
      </div>

      {/* 1. Dataset */}
      <div className="card p-6 mb-4">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>1. Dataset</h2>
        {datasets.length === 0 ? (
          <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
            No datasets available. <a href="/datasets" style={{ color: 'var(--accent)' }}>Upload one first →</a>
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {datasets.map((ds) => {
              const sel = selectedDataset === String(ds.id)
              return (
                <button key={ds.id} onClick={() => setSelectedDataset(String(ds.id))}
                  className="text-left p-4 rounded-xl border transition-all duration-150"
                  style={{
                    borderColor: sel ? 'var(--accent)' : 'var(--border-mid)',
                    backgroundColor: sel ? 'var(--accent-bg)' : 'var(--bg-2)',
                  }}>
                  <div className="text-sm font-medium truncate" style={{ color: 'var(--fg)' }}>{ds.name}</div>
                  <div className="text-xs mt-1" style={{ color: 'var(--fg-muted)' }}>
                    {ds.rows?.toLocaleString()} rows · {ds.granularity} · {ds.file_format?.toUpperCase()}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* 2. Model */}
      <div className="card p-6 mb-4">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>2. Algorithm</h2>
        <div className="grid grid-cols-4 gap-3">
          {Object.entries(MODEL_META).map(([key, m]) => {
            const sel = selectedModel === key
            const hex = m.color.startsWith('var') ? undefined : m.color
            return (
              <button key={key} onClick={() => handleModelChange(key)}
                className="text-left p-4 rounded-xl border transition-all duration-150"
                style={{
                  borderColor: sel ? (hex || 'var(--accent)') : 'var(--border-mid)',
                  backgroundColor: sel ? (hex ? `${hex}18` : 'var(--accent-bg)') : 'var(--bg-2)',
                }}>
                <div className="text-sm font-semibold mb-1"
                     style={{ color: sel ? (hex || 'var(--accent)') : 'var(--fg)' }}>
                  {m.label}
                </div>
                <div className="text-xs leading-relaxed" style={{ color: 'var(--fg-muted)' }}>{m.desc}</div>
              </button>
            )
          })}
        </div>

        <div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-0)', border: '1px solid var(--border)' }}>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--fg-subtle)' }}>
            <span style={{ color: 'var(--fg-muted)' }}>GRU vs LSTM:</span> GRU trains ~30% faster with fewer parameters — good for limited data.
            LSTM retains longer memory — better for multi-week patterns.{' '}
            <span style={{ color: 'var(--fg-muted)' }}>Bidirectional variants</span> add backward context at ~2× compute cost.{' '}
            <span style={{ color: 'var(--fg-muted)' }}>Transformer / TFT</span> excel at long-range patterns — use lookback ≥ 48 for best results.
          </p>
        </div>
      </div>

      {/* 3. Hyperparameters */}
      <div className="card p-6 mb-4">
        <button className="flex items-center justify-between w-full" onClick={() => setExpanded((v) => !v)}>
          <h2 className="text-sm font-medium" style={{ color: 'var(--fg)' }}>3. Hyperparameters</h2>
          <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--fg-subtle)' }}>
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {expanded ? 'collapse' : 'expand'}
          </div>
        </button>

        {expanded && (
          <div className="mt-5 space-y-4 animate-slide-up">
            <ParamGrid model={selectedModel} params={params} onChange={handleParamChange} />
          </div>
        )}

        {!expanded && (
          <div className="mt-3 flex flex-wrap gap-2">
            {Object.entries(params).slice(0, 6).map(([k, v]) => (
              <span key={k} className="badge font-mono text-xs"
                    style={{ backgroundColor: 'var(--bg-2)', border: '1px solid var(--border-mid)', color: 'var(--fg-muted)' }}>
                {k}={String(v)}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* 4. Run settings */}
      <div className="card p-6 mb-6">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>4. Run Settings</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Forecast Horizon (days)</label>
            <input type="number" className="input" value={horizon} min={1} max={90}
                   onChange={(e) => setHorizon(parseInt(e.target.value) || 7)} />
          </div>
          <div>
            <label className="label">Run Name (optional)</label>
            <input className="input" value={runName} onChange={(e) => setRunName(e.target.value)}
                   placeholder={`${selectedModel.toUpperCase()} run`} />
          </div>
        </div>
      </div>

      <button className="btn-primary w-full py-3 flex items-center justify-center gap-2 text-base"
              onClick={handleSubmit} disabled={submitting || !selectedDataset}>
        <Play size={16} />
        {submitting ? 'Starting...' : 'Start Training'}
      </button>
    </div>
  )
}

function ParamGrid({ model, params, onChange }) {
  const PARAM_DEFS = {
    stack_types:  { label: 'Stack types (comma-sep)', type: 'text' },
    num_blocks:   { label: 'Blocks per stack', type: 'number', min: 1, max: 8 },
    layer_width:  { label: 'FC layer width', type: 'number', min: 32, max: 1024 },
    num_layers:   { label: 'FC layers per block', type: 'number', min: 2, max: 8 },
    p: { label: 'AR order (p)', type: 'number', min: 0, max: 10 },
    d: { label: 'Diff order (d)', type: 'number', min: 0, max: 2 },
    q: { label: 'MA order (q)', type: 'number', min: 0, max: 10 },
    seasonal: { label: 'Seasonal', type: 'bool' },
    seasonal_m: { label: 'Seasonal period (m)', type: 'number', min: 2, max: 365 },
    seasonal_P: { label: 'Seasonal P', type: 'number', min: 0, max: 3 },
    seasonal_D: { label: 'Seasonal D', type: 'number', min: 0, max: 2 },
    seasonal_Q: { label: 'Seasonal Q', type: 'number', min: 0, max: 3 },
    units_1: { label: 'Layer 1 units', type: 'number', min: 8, max: 256 },
    units_2: { label: 'Layer 2 units', type: 'number', min: 8, max: 128 },
    dropout: { label: 'Dropout rate', type: 'float', min: 0, max: 0.5, step: 0.05 },
    lookback: { label: 'Lookback window (steps)', type: 'number', min: 4, max: 336 },
    epochs: { label: 'Epochs', type: 'number', min: 5, max: 500 },
    batch_size: { label: 'Batch size', type: 'number', min: 8, max: 256 },
    learning_rate: { label: 'Learning rate', type: 'float', min: 0.0001, max: 0.1, step: 0.0001 },
    merge_mode: { label: 'Merge mode', type: 'select', options: ['concat', 'sum', 'mul', 'ave'] },
    wavelet: { label: 'Wavelet family', type: 'select', options: ['db4', 'db2', 'haar', 'sym8', 'coif1'] },
    level: { label: 'Decomp. levels', type: 'number', min: 1, max: 5 },
    wave_window: { label: 'Wavelet window (steps)', type: 'number', min: 32, max: 512 },
    d_model: { label: 'Model dim (d_model)', type: 'number', min: 16, max: 512 },
    n_heads: { label: 'Attention heads', type: 'number', min: 1, max: 16 },
    n_layers: { label: 'Encoder layers', type: 'number', min: 1, max: 8 },
    ffn_dim: { label: 'FFN inner dim', type: 'number', min: 32, max: 1024 },
    lstm_layers: { label: 'LSTM layers (TFT)', type: 'number', min: 1, max: 4 },
    train_split: { label: 'Train split', type: 'float', min: 0.5, max: 0.95, step: 0.05 },
    validation_split: { label: 'Validation split', type: 'float', min: 0.05, max: 0.3, step: 0.05 },
    optimizer: { label: 'Optimizer', type: 'select', options: ['adam', 'adamw', 'sgd', 'rmsprop'] },
    dense_units: { label: 'Head dense units', type: 'number', min: 8, max: 256 },
    dense_activation: { label: 'Head activation', type: 'select', options: ['relu', 'tanh', 'elu', 'gelu', 'selu'] },
    early_stopping: { label: 'Early stopping', type: 'bool' },
    es_patience: { label: 'Early stop patience', type: 'number', min: 1, max: 50 },
    reduce_lr: { label: 'Reduce LR on plateau', type: 'bool' },
    lr_factor: { label: 'LR reduction factor', type: 'float', min: 0.1, max: 0.9, step: 0.1 },
    lr_patience: { label: 'LR patience', type: 'number', min: 1, max: 30 },
  }

  const SECTIONS = [
    { title: 'Architecture', keys: ['units_1','units_2','d_model','n_heads','n_layers','ffn_dim','lstm_layers','merge_mode','wavelet','level','wave_window','stack_types','num_blocks','layer_width','num_layers'] },
    { title: 'Sequence',     keys: ['lookback','dropout'] },
    { title: 'Training',     keys: ['epochs','batch_size','learning_rate','optimizer','train_split','validation_split','dense_units','dense_activation'] },
    { title: 'ARIMA Orders', keys: ['p','d','q','seasonal','seasonal_m','seasonal_P','seasonal_D','seasonal_Q'] },
    { title: 'Early Stopping', keys: ['early_stopping','es_patience'] },
    { title: 'LR Schedule',  keys: ['reduce_lr','lr_factor','lr_patience'] },
  ]

  const grouped = {}
  SECTIONS.forEach(s => s.keys.forEach(k => { grouped[k] = s.title }))
  const bySection = {}
  Object.entries(params).forEach(([key, val]) => {
    const def = PARAM_DEFS[key]; if (!def) return
    const section = grouped[key] || 'Other'
    if (!bySection[section]) bySection[section] = []
    bySection[section].push([key, val, def])
  })

  const renderField = (key, val, def) => (
    <div key={key}>
      <label className="label">{def.label}</label>
      {def.type === 'bool' ? (
        <button onClick={() => onChange(key, !val)}
          className="flex items-center gap-3 mt-1 group" type="button">
          <span className="relative inline-flex w-11 h-6 flex-shrink-0 rounded-full border-2 border-transparent transition-colors duration-200"
                style={{ backgroundColor: val ? 'var(--accent)' : 'var(--bg-3)' }}>
            <span className={clsx('pointer-events-none inline-block w-5 h-5 rounded-full bg-white shadow transform transition-transform duration-200',
              val ? 'translate-x-5' : 'translate-x-0')} />
          </span>
          <span className="text-xs" style={{ color: val ? 'var(--accent)' : 'var(--fg-muted)' }}>
            {val ? 'Enabled' : 'Disabled'}
          </span>
        </button>
      ) : def.type === 'text' ? (
        <input type="text" className="input text-sm font-mono" value={val}
          onChange={(e) => onChange(key, e.target.value)} />
      ) : def.type === 'select' ? (
        <select className="input text-sm" value={val} onChange={(e) => onChange(key, e.target.value)}>
          {def.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input type="number" className="input text-sm font-mono" value={val}
          min={def.min} max={def.max} step={def.step || 1}
          onChange={(e) => onChange(key, def.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value))} />
      )}
    </div>
  )

  return (
    <div className="space-y-5">
      {Object.entries(bySection).map(([section, fields]) => (
        <div key={section}>
          <div className="text-xs font-medium uppercase tracking-wider mb-3 flex items-center gap-2"
               style={{ color: 'var(--fg-subtle)' }}>
            <span className="w-8 h-px inline-block" style={{ backgroundColor: 'var(--border-mid)' }} />
            {section}
          </div>
          <div className="grid grid-cols-2 gap-3">
            {fields.map(([key, val, def]) => renderField(key, val, def))}
          </div>
        </div>
      ))}
    </div>
  )
}
