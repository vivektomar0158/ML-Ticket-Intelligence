import { diffWords } from 'diff'

export type Part = { type: 'text'; text: string } | { type: 'cite'; id: number }

/** Splits a draft into text and [T-123] citation parts so citations can render as clickable chips. */
export function splitCitations(reply: string): Part[] {
  const out: Part[] = []
  const re = /\[T-(\d+)\]/g
  let last = 0
  for (let m = re.exec(reply); m; m = re.exec(reply)) {
    if (m.index > last) out.push({ type: 'text', text: reply.slice(last, m.index) })
    out.push({ type: 'cite', id: Number(m[1]) })
    last = m.index + m[0].length
  }
  if (last < reply.length) out.push({ type: 'text', text: reply.slice(last) })
  return out
}

export type SlaState = { label: string; tone: 'good' | 'warn' | 'bad' | 'muted' }

/** Human SLA countdown: "in 2h 10m" / "overdue 35m". Resolved tickets have no SLA pressure. */
export function slaState(dueAt: string | null, status: string, breached: boolean, now = Date.now()): SlaState {
  if (status === 'RESOLVED' || status === 'CLOSED') return { label: 'done', tone: 'muted' }
  if (!dueAt) return { label: '—', tone: 'muted' }
  const ms = new Date(dueAt).getTime() - now
  const abs = Math.abs(ms)
  const m = Math.floor(abs / 60000)
  const txt = m >= 1440 ? `${Math.floor(m / 1440)}d ${Math.floor((m % 1440) / 60)}h` : m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`
  if (ms < 0 || breached) return { label: `overdue ${txt}`, tone: 'bad' }
  return { label: `in ${txt}`, tone: ms < 15 * 60000 ? 'warn' : 'good' }
}

export function timeAgo(iso: string, now = Date.now()): string {
  const s = Math.max(0, Math.floor((now - new Date(iso).getTime()) / 1000))
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}

export const pct = (x: number | null | undefined, digits = 0) => (x == null ? '—' : `${(x * 100).toFixed(digits)}%`)

export const label = (s: string | null) => (s ? s.replace(/_/g, ' ').toLowerCase().replace(/^./, (c) => c.toUpperCase()) : '—')

export interface DiffStats { added: number; removed: number; changed: boolean }

export function diffStats(before: string, after: string): DiffStats {
  let added = 0, removed = 0
  for (const p of diffWords(before, after)) {
    const n = p.value.trim() ? p.value.trim().split(/\s+/).length : 0
    if (p.added) added += n
    if (p.removed) removed += n
  }
  return { added, removed, changed: before.trim() !== after.trim() }
}

export function riskTone(r: number | null): 'high' | '' { return r != null && r >= 0.4 ? 'high' : '' }

export const FACTOR_LABELS: Record<string, string> = {
  tier: 'Customer tier', prior_tickets_7d: 'Recent tickets from customer', similar_recent: 'Similar reports right now',
  off_hours: 'Outside business hours', hour: 'Time of day', weekday: 'Day of week', text_embedding: 'Ticket wording',
  p_pri_URGENT: 'Predicted urgent', p_pri_HIGH: 'Predicted high priority', p_pri_MEDIUM: 'Predicted medium priority', p_pri_LOW: 'Predicted low priority',
  sig_deadline: 'Deadline language', sig_outage: 'Outage language', sig_legal: 'Legal / breach language', sig_churn: 'Cancellation language',
  sig_blocked: 'Blocked / failing language', vader_neg: 'Negative sentiment', exclaim: 'Exclamation marks', caps_ratio: 'Shouting (caps)',
}

export const factorLabel = (f: string) => FACTOR_LABELS[f] ?? f.replace(/^p_cat_/, 'Category: ').replace(/_/g, ' ')
