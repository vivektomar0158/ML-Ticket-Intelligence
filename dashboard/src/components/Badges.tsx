import type { TicketRow } from '../types'
import { label, pct, riskTone, slaState } from '../lib'

const PRIO_TONE: Record<string, string> = { URGENT: 'bad', HIGH: 'warn', MEDIUM: 'info', LOW: '' }

export function PriorityBadge({ p }: { p: string | null }) {
  if (!p) return <span className="chip faint">—</span>
  return <span className={`chip ${PRIO_TONE[p] ?? ''}`}>{p === 'URGENT' ? '▲ ' : ''}{label(p)}</span>
}

export function RiskBar({ risk }: { risk: number | null }) {
  if (risk == null) return <span className="muted">—</span>
  return (
    <span className={`riskbar ${riskTone(risk)}`} title={`Escalation risk ${pct(risk)}`}>
      <span className="track"><span className="fill" style={{ width: `${Math.round(risk * 100)}%` }} /></span>
      {risk.toFixed(2)}
    </span>
  )
}

/** Category + confidence; dimmed below the cascade threshold, tagged when the LLM or an agent chose it. */
export function CategoryChip({ t }: { t: Pick<TicketRow, 'category' | 'category_conf' | 'category_source'> }) {
  if (!t.category) return <span className="chip faint">classifying…</span>
  const low = t.category_source === 'MODEL' && (t.category_conf ?? 1) < 0.8
  return (
    <span className={`chip ${low ? 'faint' : ''}`} title={`Confidence ${pct(t.category_conf)} · set by ${t.category_source?.toLowerCase()}`}>
      {label(t.category)}
      {t.category_source === 'LLM' && <b className="small">· LLM</b>}
      {t.category_source === 'AGENT' && <b className="small">· you</b>}
      {t.category_source === 'MODEL' && <span className="muted small">{pct(t.category_conf)}</span>}
    </span>
  )
}

export function DupBadge({ t }: { t: Pick<TicketRow, 'cluster_id' | 'cluster_size' | 'is_incident'> }) {
  if (!t.cluster_id || !t.cluster_size || t.cluster_size < 2) return null
  return <span className={`chip ${t.is_incident ? 'bad' : 'info'}`}>{t.is_incident ? '🔥 incident · ' : '×'}{t.cluster_size}</span>
}

export function SlaChip({ t }: { t: Pick<TicketRow, 'sla_due_at' | 'status' | 'sla_breached'> }) {
  const s = slaState(t.sla_due_at, t.status, t.sla_breached)
  return <span className={`chip ${s.tone === 'muted' ? '' : s.tone}`}>{s.label}</span>
}

const DRAFT_TXT: Record<string, [string, string]> = {
  READY: ['draft ready', 'good'], PENDING: ['drafting…', 'info'], WAITING_LLM: ['LLM paused', 'warn'],
  SKIPPED: ['no draft', ''], FAILED: ['draft failed', 'bad'], NONE: ['', ''],
}

export function StatusChip({ t }: { t: Pick<TicketRow, 'status' | 'draft_state'> }) {
  if (t.status === 'RESOLVED' || t.status === 'CLOSED') return <span className="chip good">{label(t.status)}</span>
  const [txt, tone] = DRAFT_TXT[t.draft_state] ?? ['', '']
  return <span className={`chip ${tone}`}>{txt || label(t.status)}</span>
}
