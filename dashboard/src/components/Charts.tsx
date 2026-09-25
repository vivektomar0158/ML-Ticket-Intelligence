import { useState, type ReactNode } from 'react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export interface Datum { [k: string]: string | number | null }

/** Card with a chart/table toggle: every chart has a table view (accessibility + exact values). */
export function ChartCard({ title, subtitle, table, children }: { title: string; subtitle?: string; table: { cols: string[]; rows: (string | number | null)[][] }; children: ReactNode }) {
  const [asTable, setAsTable] = useState(false)
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div><h2 style={{ marginBottom: 0 }}>{title}</h2>{subtitle && <div className="small muted">{subtitle}</div>}</div>
        <button className="btn ghost small" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>{asTable ? 'Show chart' : 'Show table'}</button>
      </div>
      {asTable ? (
        <table style={{ marginTop: 8 }}>
          <thead><tr>{table.cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{table.rows.map((r, i) => <tr key={i} style={{ cursor: 'default' }}>{r.map((v, j) => <td key={j} className={j ? 'mono' : ''}>{v ?? '—'}</td>)}</tr>)}</tbody>
        </table>
      ) : (
        <div style={{ height: 210, marginTop: 8 }}>{children}</div>
      )}
    </div>
  )
}

const axis = { stroke: 'var(--border)', tick: { fill: 'var(--text-2)', fontSize: 11 } }

function Tip({ active, payload, label, fmt }: { active?: boolean; payload?: { value: number }[]; label?: string; fmt: (v: number) => string }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: '6px 10px', boxShadow: 'var(--shadow)' }}>
      <div className="small muted">{label}</div>
      <b style={{ fontVariantNumeric: 'tabular-nums' }}>{fmt(payload[0].value)}</b>
    </div>
  )
}

/** Single-series vertical bars: thin, 4px rounded data-end, recessive horizontal grid, tooltip on hover. */
export function Bars({ data, x, y, fmt = (v) => String(v) }: { data: Datum[]; x: string; y: string; fmt?: (v: number) => string }) {
  if (!data.length) return <div className="muted small" style={{ paddingTop: 70, textAlign: 'center' }}>No data in this window.</div>
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
        <XAxis dataKey={x} {...axis} tickLine={false} interval="preserveStartEnd" />
        <YAxis {...axis} axisLine={false} tickLine={false} width={52} tickFormatter={(v) => fmt(v as number)} />
        <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<Tip fmt={fmt} />} />
        <Bar dataKey={y} fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={28} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function Spark({ data, x, y }: { data: Datum[]; x: string; y: string }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
        <XAxis dataKey={x} {...axis} tickLine={false} minTickGap={40} tickFormatter={(v) => String(v).slice(11, 16)} />
        <YAxis {...axis} axisLine={false} tickLine={false} allowDecimals={false} width={40} />
        <Tooltip content={<Tip fmt={(v) => `${v} tickets`} />} />
        <Area type="monotone" dataKey={y} stroke="var(--series-1)" strokeWidth={2} fill="var(--series-1)" fillOpacity={0.15} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}
