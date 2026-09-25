import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { CategoryChip, DupBadge, PriorityBadge, RiskBar, SlaChip } from '../components/Badges'
import { DraftPanel } from '../components/DraftPanel'
import { factorLabel, label, pct, timeAgo } from '../lib'
import type { Prediction, TicketDetail } from '../types'

function latest(preds: Prediction[], task: Prediction['task']) {
  return preds.filter((p) => p.task === task)
}

export function Ticket() {
  const { id } = useParams()
  const nav = useNavigate()
  const [hot, setHot] = useState<number | null>(null)
  const q = useQuery({ queryKey: ['ticket', Number(id)], queryFn: () => api<TicketDetail>(`/api/tickets/${id}`) })

  if (q.isLoading) return <p className="muted">Loading…</p>
  if (q.isError || !q.data) return <div className="banner bad">Could not load ticket: {(q.error as Error)?.message}</div>
  const d = q.data
  const t = d.ticket
  const cat = latest(d.predictions, 'CATEGORY')
  const pri = latest(d.predictions, 'PRIORITY')[0]
  const esc = latest(d.predictions, 'ESCALATION')[0]
  const factors = ((esc?.details?.topFactors as { feature: string; contribution: number }[]) ?? [])
  const signals = ((pri?.details?.signals as string[]) ?? [])
  const cited = new Set((d.draft?.citations ?? []).map((c) => c.ticketId))

  return (
    <>
      <div className="page-head">
        <div className="row"><button className="btn ghost" onClick={() => nav(-1)}>← Back</button><h1>{t.subject}</h1></div>
        <div className="row"><SlaChip t={t} /><PriorityBadge p={t.priority} /><DupBadge t={{ cluster_id: t.cluster_id, cluster_size: d.cluster?.size ?? null, is_incident: !!d.cluster?.is_incident }} /></div>
      </div>

      <div className="grid two">
        <div>
          <div className="card">
            <div className="row muted small" style={{ marginBottom: 8 }}>
              <span>#{t.id}</span><span>·</span><span>{label(t.customer_tier)} customer {t.customer_external_id ?? ''}</span><span>·</span><span>{t.product}</span><span>·</span><span>{timeAgo(t.created_at)}</span>
            </div>
            <div className="ticket-body" data-testid="ticket-body">{t.body}</div>
          </div>

          {d.cluster?.is_incident && (
            <div className="banner bad" style={{ marginTop: 12 }}>🔥 Part of an incident: {d.cluster.size} similar reports. <Link to="/incidents">View incident →</Link></div>
          )}

          <DraftPanel detail={d} hot={hot} onHot={setHot} />
        </div>

        <div>
          <div className="card">
            <h2>AI analysis</h2>
            <dl className="kv">
              <dt>Category</dt>
              <dd className="stack">
                {cat.map((p) => (
                  <div key={p.model_name + p.model_version}>
                    <span className="chip">{label(p.label)}</span> <span className="muted small">{pct(p.score)} · {p.model_name === 'gemini_zeroshot' ? 'LLM' : 'model'}</span>
                  </div>
                ))}
                <div className="small muted">Now: <CategoryChip t={t} /></div>
                {cat.length > 1 && cat[0].label !== cat[1].label && <div className="small" style={{ color: 'var(--warn)' }}>The cheap model and the LLM disagree: worth a look.</div>}
              </dd>
              <dt>Priority</dt>
              <dd>
                <PriorityBadge p={t.priority} /> <span className="muted small">{pct(pri?.score)}</span>
                {!!signals.length && <div className="row" style={{ marginTop: 4 }}>{signals.map((s) => <span key={s} className="chip info">{s.startsWith('deadline:') ? `deadline: “${s.slice(9)}”` : label(s)}</span>)}</div>}
              </dd>
              <dt>Escalation</dt>
              <dd>
                <RiskBar risk={t.escalation_risk} /> <span className={`chip ${t.queue === 'SENIOR' ? 'warn' : ''}`}>→ {label(t.queue)} queue</span>
                {!!factors.length && (
                  <ul className="small" style={{ margin: '6px 0 0', paddingLeft: 18 }} aria-label="Why this risk">
                    {factors.map((f) => <li key={f.feature}>{factorLabel(f.feature)} <span className="muted">({f.contribution > 0 ? '+' : ''}{f.contribution.toFixed(2)})</span></li>)}
                  </ul>
                )}
              </dd>
            </dl>
          </div>

          <div className="card">
            <h2>Similar resolved tickets</h2>
            {!d.sourceTickets.length && <p className="muted small">No similar resolved tickets were found.</p>}
            {d.sourceTickets.map((s) => (
              <div key={s.id} id={`src-${s.id}`} className={`source ${hot === s.id ? 'hot' : ''}`} onMouseEnter={() => setHot(s.id)} onMouseLeave={() => setHot(null)}>
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <b className="small">T-{s.id} · {s.subject}</b>
                  <span className="row">{cited.has(s.id) && <span className="chip good">cited</span>}<span className="chip">{pct(d.draft?.sources?.find((x) => x.id === s.id)?.score)}</span></span>
                </div>
                <div className="small muted" style={{ marginTop: 4 }}>{s.resolution}</div>
              </div>
            ))}
          </div>

          {!!d.duplicates.length && (
            <div className="card">
              <h2>Possible duplicates</h2>
              {d.duplicates.slice(0, 6).map((x) => (
                <div key={x.id} className="row small" style={{ justifyContent: 'space-between', padding: '3px 0' }}>
                  <Link to={`/tickets/${x.id}`}>#{x.id} {x.subject}</Link><span className="chip">{pct(x.similarity)}</span>
                </div>
              ))}
              {d.cluster && <Link className="small" to={`/incidents`}>{d.cluster.size} tickets in this group →</Link>}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
