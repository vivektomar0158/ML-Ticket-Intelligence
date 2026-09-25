import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { diffWords } from 'diff'
import { api, ApiError } from '../api'
import { diffStats, label, pct, splitCitations } from '../lib'
import { CATEGORIES, PRIORITIES, REJECT_REASONS, type TicketDetail } from '../types'

interface Props {
  detail: TicketDetail
  hot: number | null
  onHot: (id: number | null) => void
  onDone?: () => void
}

const GROUND_TONE = { STRONG: 'good', WEAK: 'warn', NONE: 'bad' } as const
const GROUND_HELP = {
  STRONG: 'Closely matches resolved tickets',
  WEAK: 'Partial match: verify the steps before sending',
  NONE: 'No similar resolved tickets: the draft only asks for details',
} as const

export function DraftPanel({ detail, hot, onHot, onDone }: Props) {
  const qc = useQueryClient()
  const { ticket, draft } = detail
  const resolved = ticket.status === 'RESOLVED' || ticket.status === 'CLOSED'
  const reviewable = draft?.status === 'GENERATED' && !resolved

  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(draft?.body ?? '')
  const [showDiff, setShowDiff] = useState(false)
  const [rejecting, setRejecting] = useState(false)
  const [reason, setReason] = useState('')
  const [instruction, setInstruction] = useState('')
  const [rating, setRating] = useState(0)
  const [category, setCategory] = useState('')
  const [priority, setPriority] = useState('')
  const [manual, setManual] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const openedAt = useRef(Date.now())

  useEffect(() => { setText(draft?.body ?? ''); setEditing(false) }, [draft?.id, draft?.body])

  const done = () => {
    qc.invalidateQueries({ queryKey: ['ticket', ticket.id] })
    qc.invalidateQueries({ queryKey: ['tickets'] })
    qc.invalidateQueries({ queryKey: ['counts'] })
    onDone?.()
  }
  const fail = (e: unknown) => setErr(e instanceof ApiError ? e.message : 'Something went wrong')

  const labels = { correctedCategory: category || undefined, correctedPriority: priority || undefined, rating: rating || undefined }
  const review = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api(`/api/drafts/${draft!.id}/review`, { method: 'POST', json: { ...labels, timeToReviewMs: Date.now() - openedAt.current, ...body } }),
    onSuccess: () => { setErr(null); setShowDiff(false); setRejecting(false); done() }, onError: fail,
  })
  const regenerate = useMutation({
    mutationFn: () => api(`/api/tickets/${ticket.id}/draft:regenerate`, { method: 'POST', json: { instruction: instruction || undefined } }),
    onSuccess: () => { setErr(null); setInstruction(''); done() }, onError: fail,
  })
  const resolve = useMutation({
    mutationFn: () => api(`/api/tickets/${ticket.id}/resolve`, { method: 'POST', json: { body: manual, ...labels, rating: undefined } }),
    onSuccess: () => { setErr(null); done() }, onError: fail,
  })

  const changed = draft?.body != null && text.trim() !== draft.body.trim()
  const approve = () => (editing && changed ? setShowDiff(true) : review.mutate({ action: 'APPROVE' }))

  // shortcuts: a = approve, e = edit (not while typing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = (e.target as HTMLElement).tagName
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(t) || !reviewable || showDiff || rejecting) return
      if (e.key === 'a') approve()
      if (e.key === 'e') setEditing(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const src = new Set((draft?.sources ?? []).map((s) => s.id))
  return (
    <div className="card" aria-label="Draft reply">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2>Draft reply</h2>
        {draft?.grounding && (
          <span className={`chip ${GROUND_TONE[draft.grounding]}`} title={GROUND_HELP[draft.grounding]}>
            {draft.grounding === 'STRONG' ? '● ' : draft.grounding === 'WEAK' ? '◐ ' : '○ '}{label(draft.grounding)} grounding
          </span>
        )}
      </div>

      {ticket.draft_state === 'WAITING_LLM' && <div className="banner warn">⏸ Drafting is paused: the LLM is unavailable. The draft will appear automatically; you can also reply manually below.</div>}
      {ticket.draft_state === 'PENDING' && !draft && <div className="banner info">⏳ Drafting a reply from similar resolved tickets…</div>}
      {ticket.draft_state === 'FAILED' && <div className="banner bad">Draft generation failed. Reply manually or regenerate.</div>}
      {ticket.draft_state === 'SKIPPED' && <div className="banner info">No draft was generated for this ticket (low priority under load). Click “Regenerate” to create one.</div>}
      {draft?.reused_from_draft_id && <div className="banner info">♻ Reused from the incident’s canonical draft: one reply for all affected customers.</div>}
      {err && <div className="banner bad" role="alert">{err}</div>}

      {draft?.body && (
        <>
          {!editing ? (
            <div className="draft-view" data-testid="draft-view">
              {splitCitations(draft.body).map((p, i) =>
                p.type === 'text' ? <span key={i}>{p.text}</span> : (
                  <button key={i} className={`chip cite ${hot === p.id ? 'hot' : ''} ${src.has(p.id) ? '' : 'faint'}`} onMouseEnter={() => onHot(p.id)} onMouseLeave={() => onHot(null)}
                    onClick={() => document.getElementById(`src-${p.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })} title={src.has(p.id) ? 'Source ticket used for this reply' : 'Not in the retrieved sources'}>T-{p.id}</button>
                ),
              )}
            </div>
          ) : (
            <textarea value={text} onChange={(e) => setText(e.target.value)} aria-label="Edit draft" autoFocus />
          )}
          {!!draft.needs_info?.length && (
            <div className="banner info" style={{ marginTop: 10 }}>
              <div><b>Ask the customer:</b><ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>{draft.needs_info.map((q) => <li key={q}>{q}</li>)}</ul></div>
            </div>
          )}
          <div className="small muted" style={{ marginTop: 6 }}>
            Model confidence {pct(draft.llm_confidence)} · {draft.model}{draft.citations?.length ? ` · ${draft.citations.length} source${draft.citations.length > 1 ? 's' : ''} cited` : ''}
          </div>
        </>
      )}

      {reviewable && (
        <div className="stack" style={{ marginTop: 12 }}>
          <div className="row">
            <button className="btn primary" onClick={approve} disabled={review.isPending}>{editing && changed ? 'Review changes & send' : 'Approve (a)'}</button>
            {!editing ? <button className="btn" onClick={() => setEditing(true)}>Edit (e)</button> : <button className="btn" onClick={() => { setEditing(false); setText(draft!.body ?? '') }}>Cancel edit</button>}
            <button className="btn danger" onClick={() => setRejecting(true)}>Reject</button>
          </div>
          <details>
            <summary className="small muted">Correct labels · rate this draft</summary>
            <div className="row" style={{ marginTop: 8 }}>
              <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Correct category">
                <option value="">Category: {label(ticket.category)} (keep)</option>{CATEGORIES.map((c) => <option key={c} value={c}>{label(c)}</option>)}
              </select>
              <select value={priority} onChange={(e) => setPriority(e.target.value)} aria-label="Correct priority">
                <option value="">Priority: {label(ticket.priority)} (keep)</option>{PRIORITIES.map((p) => <option key={p} value={p}>{label(p)}</option>)}
              </select>
              <span role="radiogroup" aria-label="Rating">{[1, 2, 3, 4, 5].map((n) => (
                <button key={n} className="btn ghost" role="radio" aria-checked={rating === n} onClick={() => setRating(n)} style={{ padding: '0 3px', color: n <= rating ? 'var(--warn)' : 'var(--text-3)' }}>★</button>))}
              </span>
            </div>
          </details>
        </div>
      )}

      {!resolved && (
        <div className="row" style={{ marginTop: 12 }}>
          <input placeholder="Regenerate with an instruction, e.g. “more formal”" value={instruction} onChange={(e) => setInstruction(e.target.value)} style={{ flex: 1, minWidth: 220 }} aria-label="Regenerate instruction" />
          <button className="btn" onClick={() => regenerate.mutate()} disabled={regenerate.isPending}>Regenerate</button>
        </div>
      )}

      {!resolved && (!draft?.body || draft.status === 'REJECTED') && (
        <div className="stack" style={{ marginTop: 12 }}>
          <h3>Write the reply yourself</h3>
          <textarea value={manual} onChange={(e) => setManual(e.target.value)} placeholder="Type your reply…" aria-label="Manual reply" />
          <button className="btn primary" onClick={() => resolve.mutate()} disabled={manual.trim().length < 10 || resolve.isPending}>Send & resolve</button>
        </div>
      )}

      {resolved && (
        <div className="stack" style={{ marginTop: 10 }}>
          <div className="banner info">✅ Resolved. This reply is now searchable evidence for future drafts.</div>
          {ticket.resolution && <div className="draft-view">{ticket.resolution}</div>}
        </div>
      )}

      {showDiff && draft?.body && (
        <div className="modal-bg" role="dialog" aria-modal="true" aria-label="Review your changes">
          <div className="modal">
            <h2>Your changes to the draft</h2>
            <p className="small muted">{(() => { const s = diffStats(draft.body!, text); return `+${s.added} words, −${s.removed} words. This edit is logged to measure draft quality.` })()}</p>
            <div className="draft-view diff" data-testid="diff">
              {diffWords(draft.body, text).map((p, i) => p.added ? <ins key={i}>{p.value}</ins> : p.removed ? <del key={i}>{p.value}</del> : <span key={i}>{p.value}</span>)}
            </div>
            <div className="row" style={{ marginTop: 12, justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => setShowDiff(false)}>Back</button>
              <button className="btn primary" onClick={() => review.mutate({ action: 'EDIT_APPROVE', finalBody: text })} disabled={review.isPending || text.trim().length < 10}>Send edited reply</button>
            </div>
          </div>
        </div>
      )}

      {rejecting && (
        <div className="modal-bg" role="dialog" aria-modal="true" aria-label="Reject draft">
          <div className="modal">
            <h2>Why is this draft not usable?</h2>
            <select value={reason} onChange={(e) => setReason(e.target.value)} aria-label="Reject reason" autoFocus>
              <option value="">Choose a reason…</option>{REJECT_REASONS.map((r) => <option key={r} value={r}>{label(r)}</option>)}
            </select>
            <div className="row" style={{ marginTop: 12, justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => setRejecting(false)}>Cancel</button>
              <button className="btn danger" disabled={!reason || review.isPending} onClick={() => review.mutate({ action: 'REJECT', rejectReason: reason })}>Reject draft</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
