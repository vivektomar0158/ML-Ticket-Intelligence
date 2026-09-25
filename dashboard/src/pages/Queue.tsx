import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { api, qs } from '../api'
import { CategoryChip, DupBadge, PriorityBadge, RiskBar, SlaChip, StatusChip } from '../components/Badges'
import { label, timeAgo } from '../lib'
import { CATEGORIES, PRIORITIES, type Counts, type TicketPage } from '../types'

type Tab = 'mine' | 'STANDARD' | 'SENIOR' | 'all'

export function Queue() {
  const nav = useNavigate()
  const [sp, setSp] = useSearchParams()
  const clusterId = sp.get('clusterId') ?? ''
  const [tab, setTab] = useState<Tab>('all')
  const [openOnly, setOpenOnly] = useState(true)
  const [category, setCategory] = useState('')
  const [priority, setPriority] = useState('')
  const [q, setQ] = useState('')
  const [sort, setSort] = useState('risk')
  const [sel, setSel] = useState(0)
  const [needsReview, setNeedsReview] = useState(false)

  const params = { open: openOnly, category, priority, q, sort, limit: 50, clusterId, mine: tab === 'mine', queue: tab === 'STANDARD' || tab === 'SENIOR' ? tab : '', status: needsReview ? 'DRAFTED' : '' }
  const list = useInfiniteQuery({
    queryKey: ['tickets', params],
    queryFn: ({ pageParam }) => api<TicketPage>(`/api/tickets${qs({ ...params, cursor: pageParam || undefined })}`),
    initialPageParam: '',
    getNextPageParam: (p) => p.nextCursor ?? undefined,
  })
  const counts = useQuery({ queryKey: ['counts'], queryFn: () => api<Counts>('/api/tickets/counts') })
  const rows = useMemo(() => list.data?.pages.flatMap((p) => p.items) ?? [], [list.data])

  // keyboard: j/k move, Enter opens (ignored while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) return
      if (e.key === 'j') setSel((s) => Math.min(s + 1, rows.length - 1))
      else if (e.key === 'k') setSel((s) => Math.max(s - 1, 0))
      else if (e.key === 'Enter' && rows[sel]) nav(`/tickets/${rows[sel].id}`)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [rows, sel, nav])
  useEffect(() => setSel(0), [tab, category, priority, q, sort, openOnly, needsReview])

  const c = counts.data
  const tabCount = (k: string) => { const v = c?.[k]; return v && typeof v === 'object' ? v.open : null }

  return (
    <>
      <div className="page-head">
        <h1>Ticket queue</h1>
        <span className="muted small">j / k to move · Enter to open</span>
      </div>
      {clusterId && <div className="banner info">Showing only tickets in duplicate group #{clusterId}. <button className="btn small" onClick={() => setSp({})}>Clear filter</button></div>}
      <div className="toolbar">
        <div className="tabs" role="tablist">
          {([['all', 'All'], ['mine', 'My tickets'], ['STANDARD', 'Standard'], ['SENIOR', 'Senior']] as [Tab, string][]).map(([k, t]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>
              {t}{(k === 'STANDARD' || k === 'SENIOR') && tabCount(k) != null ? ` (${tabCount(k)})` : ''}
            </button>
          ))}
        </div>
        <input placeholder="Search tickets…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search" />
        <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
          <option value="">All categories</option>{CATEGORIES.map((c) => <option key={c} value={c}>{label(c)}</option>)}
        </select>
        <select value={priority} onChange={(e) => setPriority(e.target.value)} aria-label="Priority">
          <option value="">All priorities</option>{PRIORITIES.map((p) => <option key={p} value={p}>{label(p)}</option>)}
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort">
          <option value="risk">Highest risk</option><option value="sla">SLA soonest</option><option value="newest">Newest</option>
        </select>
        <label className="small"><input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} /> Open only</label>
        <label className="small"><input type="checkbox" checked={needsReview} onChange={(e) => setNeedsReview(e.target.checked)} /> Needs review</label>
      </div>

      <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
        <table>
          <thead>
            <tr><th>SLA</th><th>Priority</th><th>Risk</th><th>Category</th><th>Subject</th><th>Dupes</th><th>State</th><th>Age</th></tr>
          </thead>
          <tbody>
            {rows.map((t, i) => (
              <tr key={t.id} className={i === sel ? 'sel' : ''} onClick={() => nav(`/tickets/${t.id}`)} data-testid="ticket-row">
                <td><SlaChip t={t} /></td>
                <td><PriorityBadge p={t.priority} /></td>
                <td><RiskBar risk={t.escalation_risk} /></td>
                <td><CategoryChip t={t} /></td>
                <td className="subject" title={t.subject}>{t.queue === 'SENIOR' && <span className="chip warn" style={{ marginRight: 6 }}>senior</span>}{t.customer_tier === 'ENTERPRISE' && <span className="chip info" style={{ marginRight: 6 }} title="Enterprise customer">ent</span>}{t.subject}</td>
                <td><DupBadge t={t} /></td>
                <td><StatusChip t={t} /></td>
                <td className="small muted">{timeAgo(t.created_at)}</td>
              </tr>
            ))}
            {!list.isLoading && rows.length === 0 && (
              <tr><td colSpan={8} className="muted" style={{ textAlign: 'center', padding: 28 }}>No tickets match these filters.</td></tr>
            )}
          </tbody>
        </table>
        {list.isLoading && <div className="muted" style={{ padding: 16 }}>Loading…</div>}
        {list.isError && <div className="banner bad" style={{ margin: 12 }}>Could not load tickets: {(list.error as Error).message}</div>}
      </div>
      {list.hasNextPage && (
        <div style={{ textAlign: 'center', marginTop: 12 }}>
          <button className="btn" onClick={() => list.fetchNextPage()} disabled={list.isFetchingNextPage}>{list.isFetchingNextPage ? 'Loading…' : 'Load more'}</button>
        </div>
      )}
    </>
  )
}
