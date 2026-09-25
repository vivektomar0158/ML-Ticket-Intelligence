import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import { Bars, ChartCard } from '../components/Charts'
import { label, pct } from '../lib'

type Row = Record<string, string | number | null>
interface Overview {
  totals: { tickets: number; resolved: number; senior_queue: number; sla_breached: number; avg_risk: number }
  volumeByDay: Row[]; categoryMix: Row[]; priorityMix: Row[]; riskHistogram: Row[]
  drafts: { approved: number; edited: number; rejected: number; manual: number; reviewed: number; approvalRate: number | null; approvedAsIsRate: number | null
    mean_edit_ratio: number; avg_review_ms: number; rejectReasons: Row[]; byGrounding: Row[]; statusCounts: Row[] }
  llm: { draft_calls: number; reused_drafts: number; draft_cost_usd: number; avg_draft_latency_ms: number; classifyCalls: number; llmClassifiedShare: number; costByDay: Row[] }
  labelCorrections: { category_corrections: number; priority_corrections: number }
  latencySeconds: { triage_p50: number; triage_p95: number }
  draftLatencySeconds: { draft_p50: number; draft_p95: number }
  queues: Record<string, number>
  activeIncidents: number
}

const Stat = ({ name, value, hint }: { name: string; value: string; hint?: string }) => (
  <div className="card" style={{ marginTop: 0 }}><div className="small muted">{name}</div><div className="stat">{value}</div>{hint && <div className="small muted">{hint}</div>}</div>
)
const sec = (s: number) => (s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`)

export function Metrics() {
  const [days, setDays] = useState(7)
  const q = useQuery({ queryKey: ['metrics', days], queryFn: () => api<Overview>(`/api/metrics/overview?days=${days}`), refetchInterval: 30000 })
  const m = q.data
  if (q.isLoading) return <p className="muted">Loading…</p>
  if (!m) return <div className="banner bad">Could not load metrics.</div>
  const d = m.drafts
  const outcomes: Row[] = [{ outcome: 'Approved as-is', n: d.approved }, { outcome: 'Edited then approved', n: d.edited }, { outcome: 'Rejected', n: d.rejected }, { outcome: 'Written manually', n: d.manual }]
  const saved = m.llm.reused_drafts

  return (
    <>
      <div className="page-head">
        <h1>Metrics</h1>
        <select value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Window">
          {[1, 7, 30, 90].map((n) => <option key={n} value={n}>Last {n} day{n > 1 ? 's' : ''}</option>)}
        </select>
      </div>

      <div className="grid cards">
        <Stat name="Tickets" value={String(m.totals.tickets)} hint={`${m.totals.resolved} resolved`} />
        <Stat name="Draft approval rate" value={pct(d.approvalRate)} hint={`${pct(d.approvedAsIsRate)} approved as-is · ${d.reviewed} reviewed`} />
        <Stat name="Mean edit ratio" value={d.mean_edit_ratio.toFixed(2)} hint="0 = sent unchanged, 1 = rewritten" />
        <Stat name="LLM spend" value={`$${m.llm.draft_cost_usd.toFixed(4)}`} hint={`${m.llm.draft_calls} draft calls · ${saved} drafts reused`} />
        <Stat name="Triage latency" value={sec(m.latencySeconds.triage_p95)} hint={`p95 · p50 ${sec(m.latencySeconds.triage_p50)}`} />
        <Stat name="Draft ready in" value={sec(m.draftLatencySeconds.draft_p95)} hint={`p95 · p50 ${sec(m.draftLatencySeconds.draft_p50)}`} />
        <Stat name="LLM-classified" value={pct(m.llm.llmClassifiedShare)} hint={`${m.llm.classifyCalls} cascade calls`} />
        <Stat name="Label corrections" value={String(m.labelCorrections.category_corrections + m.labelCorrections.priority_corrections)} hint="agent overrides (retraining signal)" />
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', marginTop: 12 }}>
        <ChartCard title="Ticket volume" subtitle="per day" table={{ cols: ['Day', 'Tickets'], rows: m.volumeByDay.map((r) => [String(r.day), r.tickets]) }}>
          <Bars data={m.volumeByDay} x="day" y="tickets" />
        </ChartCard>
        <ChartCard title="Category mix" table={{ cols: ['Category', 'Tickets'], rows: m.categoryMix.map((r) => [label(String(r.category)), r.tickets]) }}>
          <Bars data={m.categoryMix.map((r) => ({ ...r, category: label(String(r.category)).slice(0, 10) }))} x="category" y="tickets" />
        </ChartCard>
        <ChartCard title="Escalation risk distribution" subtitle="tickets per 0.1 risk bucket" table={{ cols: ['Risk bucket', 'Tickets'], rows: m.riskHistogram.map((r) => [`${((Number(r.bucket) - 1) / 10).toFixed(1)}–${(Number(r.bucket) / 10).toFixed(1)}`, r.tickets]) }}>
          <Bars data={m.riskHistogram.map((r) => ({ ...r, bucket: ((Number(r.bucket) - 1) / 10).toFixed(1) }))} x="bucket" y="tickets" />
        </ChartCard>
        <ChartCard title="What agents did with drafts" table={{ cols: ['Outcome', 'Count'], rows: outcomes.map((r) => [String(r.outcome), r.n]) }}>
          <Bars data={outcomes.map((r) => ({ ...r, outcome: String(r.outcome).split(' ')[0] }))} x="outcome" y="n" />
        </ChartCard>
        <ChartCard title="Approval rate by grounding" subtitle="do well-grounded drafts get approved more?" table={{ cols: ['Grounding', 'Reviewed', 'Approval', 'Mean edit ratio'], rows: d.byGrounding.map((r) => [String(r.grounding), r.reviewed, pct(Number(r.approval_rate)), r.mean_edit_ratio]) }}>
          <Bars data={d.byGrounding.map((r) => ({ grounding: label(String(r.grounding)), rate: Number(r.approval_rate) * 100 }))} x="grounding" y="rate" fmt={(v) => `${Math.round(v)}%`} />
        </ChartCard>
        <ChartCard title="LLM spend" subtitle="estimated USD per day" table={{ cols: ['Day', 'USD', 'Calls'], rows: m.llm.costByDay.map((r) => [String(r.day), Number(r.cost_usd).toFixed(5), r.calls]) }}>
          <Bars data={m.llm.costByDay.map((r) => ({ ...r, cost: Number(r.cost_usd) }))} x="day" y="cost" fmt={(v) => `$${v.toFixed(3)}`} />
        </ChartCard>
      </div>

      <div className="grid two" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>Why drafts were rejected</h2>
          {d.rejectReasons.length ? (
            <table><tbody>{d.rejectReasons.map((r) => <tr key={String(r.reason)} style={{ cursor: 'default' }}><td>{label(String(r.reason))}</td><td className="mono">{r.n}</td></tr>)}</tbody></table>
          ) : <p className="muted small">No rejections in this window.</p>}
        </div>
        <div className="card">
          <h2>Background queues</h2>
          {Object.keys(m.queues).length ? (
            <table><tbody>{Object.entries(m.queues).map(([k, v]) => <tr key={k} style={{ cursor: 'default' }}><td className="mono">{k}</td><td className="mono">{v}</td></tr>)}</tbody></table>
          ) : <p className="muted small">All queues are empty ✓</p>}
        </div>
      </div>
    </>
  )
}
