import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { Spark } from '../components/Charts'
import { timeAgo } from '../lib'
import type { ClusterRow } from '../types'

interface ClusterDetail {
  timeline: { minute: string; tickets: number }[]
  members: { id: number; subject: string }[]
  draft: { id: number; body: string | null; grounding: string | null }[]
}

function IncidentCard({ c }: { c: ClusterRow }) {
  const d = useQuery({ queryKey: ['clusters', c.id], queryFn: () => api<ClusterDetail>(`/api/clusters/${c.id}`) })
  const minutes = (new Date(c.last_seen_at).getTime() - new Date(c.first_seen_at).getTime()) / 60000
  return (
    <div className="card" data-testid="incident-card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ marginBottom: 0 }}>{c.is_incident ? '🔥 ' : ''}{c.title}</h2>
        <span className={`chip ${c.is_incident ? 'bad' : 'info'}`}>{c.size} tickets</span>
      </div>
      <div className="small muted">{c.product} · {c.customers} customers · {c.open_tickets} open · first seen {timeAgo(c.first_seen_at)} · spanning {Math.max(1, Math.round(minutes))} min</div>
      <div style={{ height: 120, marginTop: 6 }}>{d.data && <Spark data={d.data.timeline} x="minute" y="tickets" />}</div>
      {d.data?.draft?.[0]?.body && (
        <div className="stack">
          <h3>Canonical reply (reused for every ticket in this incident)</h3>
          <div className="draft-view small">{d.data.draft[0].body}</div>
        </div>
      )}
      {d.data?.members?.[0] && <p className="small"><Link to={`/tickets/${d.data.members[0].id}`}>Open latest ticket →</Link> · <Link to={`/?clusterId=${c.id}`}>All members</Link></p>}
    </div>
  )
}

export function Incidents() {
  const q = useQuery({ queryKey: ['clusters', 'active'], queryFn: () => api<ClusterRow[]>('/api/clusters?status=ACTIVE'), refetchInterval: 20000 })
  const list = q.data ?? []
  const incidents = list.filter((c) => c.is_incident)
  const groups = list.filter((c) => !c.is_incident && c.size >= 3)
  return (
    <>
      <div className="page-head"><h1>Incidents & duplicate groups</h1></div>
      {q.isLoading && <p className="muted">Loading…</p>}
      {!q.isLoading && !incidents.length && <div className="banner info">No active incidents. An incident is flagged when 5+ similar reports from 3+ customers arrive within an hour.</div>}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))' }}>
        {incidents.map((c) => <IncidentCard key={c.id} c={c} />)}
      </div>
      {!!groups.length && (
        <>
          <h3 style={{ marginTop: 22 }}>Duplicate groups (not incidents)</h3>
          <div className="card" style={{ padding: 0 }}>
            <table>
              <thead><tr><th>Group</th><th>Product</th><th>Tickets</th><th>Open</th><th>Last seen</th></tr></thead>
              <tbody>{groups.map((c) => <tr key={c.id} style={{ cursor: 'default' }}><td>{c.title}</td><td>{c.product}</td><td>{c.size}</td><td>{c.open_tickets}</td><td className="muted small">{timeAgo(c.last_seen_at)}</td></tr>)}</tbody>
            </table>
          </div>
        </>
      )}
    </>
  )
}
