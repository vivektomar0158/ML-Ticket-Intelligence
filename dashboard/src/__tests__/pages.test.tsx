import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, qs, saveSession, setUnauthorizedHandler } from '../api'
import { Queue } from '../pages/Queue'
import { Upload } from '../pages/Upload'
import type { TicketRow } from '../types'

const row = (id: number, over: Partial<TicketRow> = {}): TicketRow => ({
  id, external_id: null, subject: `Ticket ${id}`, customer_tier: 'PRO', product: 'web-app', status: 'DRAFTED', draft_state: 'READY', category: 'BILLING',
  category_conf: 0.95, category_source: 'MODEL', priority: 'HIGH', escalation_risk: 0.6, queue: 'SENIOR', cluster_id: null, cluster_size: null, is_incident: false,
  sla_due_at: null, sla_breached: false, created_at: '2026-06-01T10:00:00Z', assigned_agent_id: null, ...over,
})

function json(body: unknown, status = 200) { return Promise.resolve(new Response(JSON.stringify(body), { status })) }
const wrap = (ui: React.ReactElement) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}><MemoryRouter initialEntries={['/']}><Routes><Route path="/" element={ui} /><Route path="/tickets/:id" element={<div>detail page</div>} /></Routes></MemoryRouter></QueryClientProvider>)
}

beforeEach(() => { saveSession({ accessToken: 't', username: 'agent', name: 'Agent', role: 'AGENT' }) })

describe('api client', () => {
  it('builds query strings without empty values', () => {
    expect(qs({ a: 1, b: '', c: null, d: false, e: true, f: 'x y' })).toBe('?a=1&e=true&f=x+y')
    expect(qs({})).toBe('')
  })
  it('sends the bearer token, parses problem+json errors, and signals 401', async () => {
    const f = vi.fn()
      .mockImplementationOnce(() => json({ ok: 1 }))
      .mockImplementationOnce(() => json({ detail: 'nope' }, 400))
      .mockImplementationOnce(() => json({}, 401))
    vi.stubGlobal('fetch', f)
    await api('/api/x')
    expect((f.mock.calls[0][1].headers as Headers).get('Authorization')).toBe('Bearer t')
    await expect(api('/api/y')).rejects.toMatchObject({ status: 400, message: 'nope' })
    const out = vi.fn()
    setUnauthorizedHandler(out)
    await expect(api('/api/z')).rejects.toBeInstanceOf(ApiError)
    expect(out).toHaveBeenCalled()
  })
})

describe('Queue page', () => {
  it('renders rows with risk, category and routing, and opens a ticket on click', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => url.includes('/counts')
      ? json({ activeIncidents: 0, SENIOR: { open: 2, awaitingReview: 1, breached: 0 } })
      : json({ items: [row(1, { subject: 'SSO down', is_incident: true, cluster_id: 3, cluster_size: 12 }), row(2, { category_source: 'LLM' })], nextCursor: null })))
    wrap(<Queue />)
    expect(await screen.findByText('SSO down')).toBeInTheDocument()
    expect(screen.getByText(/incident · 12/)).toBeInTheDocument()
    expect(screen.getByText('· LLM')).toBeInTheDocument()
    expect(screen.getAllByTestId('ticket-row')).toHaveLength(2)
    await userEvent.click(screen.getByText('SSO down'))
    expect(await screen.findByText('detail page')).toBeInTheDocument()
  })

  it('passes filters to the API and supports keyboard navigation', async () => {
    const f = vi.fn().mockImplementation((url: string) => url.includes('/counts') ? json({ activeIncidents: 0 }) : json({ items: [row(1), row(2)], nextCursor: null }))
    vi.stubGlobal('fetch', f)
    wrap(<Queue />)
    await screen.findByText('Ticket 1')
    await userEvent.selectOptions(screen.getByLabelText('Priority'), 'URGENT')
    await waitFor(() => expect(f.mock.calls.some(([u]) => String(u).includes('priority=URGENT'))).toBe(true))
    await userEvent.click(document.body)   // leave the dropdown: shortcuts are ignored while a form control has focus
    await userEvent.keyboard('j')
    await waitFor(() => expect(screen.getAllByTestId('ticket-row')[1]).toHaveClass('sel'))
    await userEvent.keyboard('{Enter}')
    expect(await screen.findByText('detail page')).toBeInTheDocument()
  })

  it('shows an empty state and an error state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => url.includes('/counts') ? json({ activeIncidents: 0 }) : json({ items: [], nextCursor: null })))
    const { unmount } = wrap(<Queue />)
    expect(await screen.findByText(/No tickets match/)).toBeInTheDocument()
    unmount()
    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => url.includes('/counts') ? json({ activeIncidents: 0 }) : json({ detail: 'db down' }, 500)))
    wrap(<Queue />)
    expect(await screen.findByText(/Could not load tickets: db down/)).toBeInTheDocument()
  })
})

describe('Upload page', () => {
  it('uploads the chosen CSV and reports accepted / rejected rows with line numbers', async () => {
    const f = vi.fn().mockImplementation(() => json({ batchId: 5, total: 3, accepted: 1, duplicates: 1, rejected: 1, errors: [{ row: 4, error: 'product must be one of …' }] }, 202))
    vi.stubGlobal('fetch', f)
    wrap(<Upload />)
    const file = new File(['subject,body\nx,y'], 'tickets.csv', { type: 'text/csv' })
    await userEvent.upload(screen.getByLabelText('CSV file'), file)
    await userEvent.click(screen.getByRole('button', { name: /Upload & triage/ }))
    const res = await screen.findByTestId('upload-result')
    expect(res).toHaveTextContent('1 accepted')
    expect(res).toHaveTextContent('1 rejected')
    expect(res).toHaveTextContent('product must be one of')
    expect(f.mock.calls[0][1].body).toBeInstanceOf(FormData)
  })
})
