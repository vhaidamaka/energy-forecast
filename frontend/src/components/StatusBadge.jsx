import clsx from 'clsx'

export function StatusBadge({ status, className = '' }) {
  const styles = {
    done:     { bg: 'rgba(34,197,94,0.1)',  color: '#22c55e', border: 'rgba(34,197,94,0.25)' },
    training: { bg: 'var(--accent-bg)',      color: 'var(--accent)', border: 'var(--accent-dim)' },
    pending:  { bg: 'var(--border)',         color: 'var(--fg-muted)', border: 'var(--border-mid)' },
    failed:   { bg: 'rgba(239,68,68,0.1)',  color: '#ef4444', border: 'rgba(239,68,68,0.25)' },
  }
  const s = styles[status] || styles.pending
  return (
    <span
      className={clsx('badge', className)}
      style={{ backgroundColor: s.bg, color: s.color, border: `1px solid ${s.border}` }}
    >
      {status === 'training' && (
        <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: 'var(--accent)' }} />
      )}
      {status}
    </span>
  )
}
