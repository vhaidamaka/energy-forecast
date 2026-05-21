import { useEffect, useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useStore } from '../store'
import { datasetsApi } from '../api/client'
import toast from 'react-hot-toast'
import { Upload, Database, Trash2, Eye, Edit2, Check, X, FileText, Calendar, Hash, Zap, LineChart } from 'lucide-react'
import DatasetHistory from '../components/DatasetHistory'
import clsx from 'clsx'

export default function Datasets() {
  const { datasets, setDatasets, addDataset, removeDataset } = useStore()
  const [uploading, setUploading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [previewData, setPreviewData] = useState(null)
  const [editId, setEditId] = useState(null)
  const [editName, setEditName] = useState('')
  const [uploadName, setUploadName] = useState('')
  const [uploadDesc, setUploadDesc] = useState('')
  const [pendingFile, setPendingFile] = useState(null)
  const [historyId, setHistoryId] = useState(null)

  useEffect(() => { datasetsApi.list().then((r) => setDatasets(r.data)) }, [])

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) { setPendingFile(accepted[0]); setUploadName(accepted[0].name.replace(/\.[^.]+$/, '')) }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] },
    multiple: false,
  })

  const handleUpload = async () => {
    if (!pendingFile || !uploadName) return
    setUploading(true)
    try {
      const res = await datasetsApi.upload(pendingFile, uploadName, uploadDesc)
      addDataset(res.data)
      setPendingFile(null); setUploadName(''); setUploadDesc('')
      toast.success(`Dataset "${res.data.name}" uploaded successfully`)
    } catch (e) { toast.error(e.response?.data?.detail || 'Upload failed') }
    finally { setUploading(false) }
  }

  const handleDelete = async (id, name) => {
    if (!confirm(`Delete dataset "${name}" and all its runs?`)) return
    try { await datasetsApi.delete(id); removeDataset(id); toast.success('Dataset deleted') }
    catch { toast.error('Delete failed') }
  }

  const handlePreview = async (id) => {
    if (preview === id) { setPreview(null); setPreviewData(null); return }
    setPreview(id)
    try { const res = await datasetsApi.preview(id, 50); setPreviewData(res.data) }
    catch { toast.error('Preview failed') }
  }

  const handleRename = async (id) => {
    try {
      await datasetsApi.update(id, { name: editName })
      setDatasets(datasets.map((d) => d.id === id ? { ...d, name: editName } : d))
      setEditId(null); toast.success('Renamed')
    } catch { toast.error('Rename failed') }
  }

  return (
    <div className="p-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="section-title mb-1">Datasets</h1>
        <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>Upload and manage your energy consumption data files</p>
      </div>

      {/* Upload zone */}
      <div className="card p-6 mb-8">
        <h2 className="text-sm font-medium mb-4" style={{ color: 'var(--fg)' }}>Upload New Dataset</h2>

        {!pendingFile ? (
          <div {...getRootProps()}
            className="border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200"
            style={{ borderColor: isDragActive ? 'var(--accent)' : 'var(--border-mid)', backgroundColor: isDragActive ? 'var(--accent-bg)' : 'transparent' }}>
            <input {...getInputProps()} />
            <Upload size={32} className="mx-auto mb-3" style={{ color: isDragActive ? 'var(--accent)' : 'var(--fg-subtle)' }} />
            <p className="text-sm" style={{ color: 'var(--fg-muted)' }}>
              {isDragActive ? 'Drop the file here' : 'Drag & drop CSV or XLSX, or click to browse'}
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--fg-subtle)' }}>Supports: CSV (5-min, hourly) and XLSX (multi-feature)</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-3 p-3 rounded-lg"
                 style={{ backgroundColor: 'var(--bg-2)', border: '1px solid var(--accent-dim)' }}>
              <FileText size={16} style={{ color: 'var(--accent)' }} className="flex-shrink-0" />
              <span className="text-sm flex-1 truncate" style={{ color: 'var(--fg)' }}>{pendingFile.name}</span>
              <span className="text-xs" style={{ color: 'var(--fg-muted)' }}>{(pendingFile.size / 1024).toFixed(0)} KB</span>
              <button onClick={() => setPendingFile(null)} style={{ color: 'var(--fg-muted)' }}><X size={14} /></button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div><label className="label">Dataset Name *</label>
                <input className="input" value={uploadName} onChange={(e) => setUploadName(e.target.value)} placeholder="My energy dataset" /></div>
              <div><label className="label">Description (optional)</label>
                <input className="input" value={uploadDesc} onChange={(e) => setUploadDesc(e.target.value)} placeholder="Hospital building 2023" /></div>
            </div>
            <div className="flex gap-3">
              <button className="btn-primary" onClick={handleUpload} disabled={uploading || !uploadName}>
                {uploading ? 'Uploading...' : 'Upload Dataset'}
              </button>
              <button className="btn-ghost" onClick={() => setPendingFile(null)}>Cancel</button>
            </div>
          </div>
        )}
      </div>

      {/* Dataset list */}
      {datasets.length === 0 ? (
        <div className="card p-16 text-center">
          <Database size={40} className="mx-auto mb-4" style={{ color: 'var(--border-mid)' }} />
          <p style={{ color: 'var(--fg-subtle)' }}>No datasets yet. Upload your first file above.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {datasets.map((ds) => (
            <div key={ds.id} className="card overflow-hidden">
              <div className="p-4 flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                     style={{ backgroundColor: 'var(--accent-bg)', border: '1px solid var(--border-mid)' }}>
                  <Database size={16} style={{ color: 'var(--accent)' }} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    {editId === ds.id ? (
                      <div className="flex items-center gap-2">
                        <input className="input w-48 text-sm py-1" value={editName}
                               onChange={(e) => setEditName(e.target.value)}
                               onKeyDown={(e) => e.key === 'Enter' && handleRename(ds.id)} />
                        <button onClick={() => handleRename(ds.id)} style={{ color: '#22c55e' }}><Check size={14}/></button>
                        <button onClick={() => setEditId(null)} style={{ color: 'var(--fg-muted)' }}><X size={14}/></button>
                      </div>
                    ) : (
                      <span className="text-sm font-semibold" style={{ color: 'var(--fg)' }}>{ds.name}</span>
                    )}
                    <span className="badge text-xs"
                          style={{ backgroundColor: 'var(--bg-2)', border: '1px solid var(--border-mid)', color: 'var(--fg-muted)' }}>
                      {ds.file_format?.toUpperCase()}
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
                    {[
                      { icon: Hash, val: `${ds.rows?.toLocaleString()} rows` },
                      { icon: Zap, val: ds.energy_column },
                      { icon: Calendar, val: `${ds.date_range_start?.slice(0,10)} → ${ds.date_range_end?.slice(0,10)}` },
                    ].map(({ icon: Icon, val }) => val && (
                      <div key={val} className="flex items-center gap-1 text-xs" style={{ color: 'var(--fg-muted)' }}>
                        <Icon size={11} />{val}
                      </div>
                    ))}
                  </div>

                  {ds.description && <p className="text-xs mt-1" style={{ color: 'var(--fg-subtle)' }}>{ds.description}</p>}
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  <button onClick={() => setHistoryId(historyId === ds.id ? null : ds.id)}
                          className="btn-ghost text-xs px-3 py-1.5 flex items-center gap-1.5"
                          style={historyId === ds.id ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : {}}>
                    <LineChart size={13} /> History
                  </button>
                  <button onClick={() => handlePreview(ds.id)}
                          className={clsx('btn-ghost text-xs px-3 py-1.5 flex items-center gap-1.5')}
                          style={preview === ds.id ? { borderColor: 'var(--accent)', color: 'var(--accent)' } : {}}>
                    <Eye size={13} /> Preview
                  </button>
                  <button onClick={() => { setEditId(ds.id); setEditName(ds.name) }}
                          className="btn-ghost text-xs px-3 py-1.5 flex items-center gap-1.5">
                    <Edit2 size={13} /> Rename
                  </button>
                  <button onClick={() => handleDelete(ds.id, ds.name)}
                          className="btn-danger text-xs px-3 py-1.5 flex items-center gap-1.5">
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              {historyId === ds.id && (
                <DatasetHistory
                  datasetId={ds.id}
                  datasetName={ds.name}
                  onClose={() => setHistoryId(null)}
                />
              )}

              {preview === ds.id && previewData && (
                <div className="border-t p-4 overflow-x-auto"
                     style={{ borderColor: 'var(--border)', backgroundColor: 'var(--bg-0)' }}>
                  <table className="text-xs w-full">
                    <thead>
                      <tr>
                        {previewData.columns.slice(0, 8).map((col) => (
                          <th key={col} className="text-left font-mono pb-2 pr-4 whitespace-nowrap"
                              style={{ color: 'var(--fg-subtle)' }}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.rows.slice(0, 10).map((row, i) => (
                        <tr key={i} className="border-t" style={{ borderColor: 'var(--border)' }}>
                          {previewData.columns.slice(0, 8).map((col) => (
                            <td key={col} className="py-1.5 pr-4 font-mono whitespace-nowrap"
                                style={{ color: 'var(--fg-muted)' }}>
                              {typeof row[col] === 'number' ? row[col].toFixed(3) : String(row[col] ?? '').slice(0, 20)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-xs mt-2" style={{ color: 'var(--fg-subtle)' }}>
                    Showing 10 of {ds.rows?.toLocaleString()} rows · {previewData.columns.length} columns
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
