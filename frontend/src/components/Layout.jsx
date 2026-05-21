import { useEffect } from 'react'
import { Outlet, NavLink } from 'react-router-dom'
import { Zap, Database, Settings, Activity, GitCompare, Home, Sun, Moon, Layers, GitMerge } from 'lucide-react'
import { useStore } from '../store'
import { StatusBadge } from './StatusBadge'
import clsx from 'clsx'

const nav = [
  { to: '/', icon: Home, label: 'Dashboard', exact: true },
  { to: '/datasets', icon: Database, label: 'Datasets' },
  { to: '/configure', icon: Settings, label: 'New Run' },
  { to: '/batch',     icon: Layers,    label: 'Batch Run' },
  { to: '/ensemble',  icon: GitMerge,  label: 'Ensemble' },
  { to: '/compare', icon: GitCompare, label: 'Compare' },
]

export default function Layout() {
  const runs      = useStore((s) => s.runs)
  const theme     = useStore((s) => s.theme)
  const toggleTheme = useStore((s) => s.toggleTheme)
  const initTheme = useStore((s) => s.initTheme)
  const recentRun = runs.find((r) => r.status === 'training') || runs[0]

  // Apply stored theme on mount
  useEffect(() => { initTheme() }, [])

  const isLight = theme === 'light'

  return (
    <div className="flex h-screen overflow-hidden" style={{ backgroundColor: 'var(--bg-0)' }}>
      {/* Sidebar */}
      <aside className="sidebar w-56 flex-shrink-0 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-6" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                 style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-dim)' }}>
              <Zap size={16} style={{ color: 'var(--accent)' }} />
            </div>
            <div>
              <div className="font-display font-bold text-sm leading-none" style={{ color: 'var(--fg)' }}>ENERGY</div>
              <div className="font-display font-bold text-sm leading-none" style={{ color: 'var(--accent)' }}>PREDICTOR</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {nav.map(({ to, icon: Icon, label, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) => clsx('nav-link', isActive && 'active')}
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Theme toggle + recent run */}
        <div className="px-3 pb-4 space-y-3">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-all duration-150"
            style={{
              border: '1px solid var(--border-mid)',
              color: 'var(--fg-muted)',
              backgroundColor: 'transparent',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent-dim)'}
            onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-mid)'}
          >
            <span className="flex items-center gap-2">
              {isLight ? <Moon size={14} /> : <Sun size={14} />}
              {isLight ? 'Dark mode' : 'Light mode'}
            </span>
            {/* Mini toggle track */}
            <span className="relative inline-flex w-9 h-5 rounded-full transition-colors duration-200"
                  style={{ backgroundColor: isLight ? 'var(--accent)' : 'var(--bg-3)' }}>
              <span className={clsx(
                'absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200',
                isLight ? 'translate-x-4' : 'translate-x-0.5'
              )} />
            </span>
          </button>

          {/* Recent run */}
          {recentRun && (
            <NavLink to={`/results/${recentRun.id}`}>
              <div className="card p-3 cursor-pointer transition-colors"
                   style={{ borderColor: 'var(--border)' }}>
                <div className="flex items-center gap-2 mb-1">
                  <Activity size={12} className={recentRun.status === 'training' ? 'animate-pulse' : ''}
                             style={{ color: recentRun.status === 'training' ? 'var(--accent)' : 'var(--fg-subtle)' }} />
                  <span className="text-xs" style={{ color: 'var(--fg-subtle)' }}>Latest run</span>
                </div>
                <div className="text-xs font-medium truncate" style={{ color: "var(--fg)" }}>
                  {recentRun.name || `Run #${recentRun.id}`}
                </div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--fg-muted)' }}>{recentRun.model?.toUpperCase()}</div>
                <StatusBadge status={recentRun.status} className="mt-2" />
              </div>
            </NavLink>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
