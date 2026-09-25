import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { useAuth } from '../auth'
import { useLiveEvents } from '../events'
import type { Counts } from '../types'

export function Layout() {
  const { session, logout } = useAuth()
  const nav = useNavigate()
  const { connected, incident, dismissIncident } = useLiveEvents(!!session)
  const counts = useQuery({ queryKey: ['counts'], queryFn: () => api<Counts>('/api/tickets/counts'), refetchInterval: 30000 })
  const c = counts.data
  const awaiting = ['STANDARD', 'SENIOR'].reduce((n, q) => n + ((c?.[q] as { awaitingReview: number } | undefined)?.awaitingReview ?? 0), 0)

  const theme = () => {
    const el = document.documentElement
    const dark = el.dataset.theme ? el.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches
    el.dataset.theme = dark ? 'light' : 'dark'
  }

  return (
    <div className="app">
      <aside className="side">
        <div className="brand"><span className="dot" /> Ticket Intelligence</div>
        <nav className="nav">
          <NavLink to="/" end>Queue <span className="chip info" title="Drafts awaiting review">{awaiting}</span></NavLink>
          <NavLink to="/incidents">Incidents {!!c?.activeIncidents && <span className="chip bad">{c.activeIncidents}</span>}</NavLink>
          <NavLink to="/upload">Upload CSV</NavLink>
          <NavLink to="/metrics">Metrics</NavLink>
        </nav>
        <div className="spacer" />
        <div className="small muted" title="Live updates from the server">{connected ? '● live' : '○ reconnecting…'}</div>
        <div className="small">{session?.name} <span className="muted">· {session?.role.toLowerCase()}</span></div>
        <div className="row">
          <button className="btn ghost small" onClick={theme}>Theme</button>
          <button className="btn ghost small" onClick={() => { logout(); nav('/login') }}>Sign out</button>
        </div>
      </aside>
      <main className="main">
        {incident && (
          <div className="banner bad" role="alert">
            🔥 <b>Possible incident detected</b> — {String(incident.data.size)} similar reports on {String(incident.data.product)}.
            <button className="btn small" onClick={() => { nav('/incidents'); dismissIncident() }}>View</button>
            <button className="btn ghost small" onClick={dismissIncident}>Dismiss</button>
          </div>
        )}
        <Outlet />
      </main>
    </div>
  )
}
