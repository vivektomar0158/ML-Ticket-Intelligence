import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DraftPanel } from '../components/DraftPanel'
import type { Draft, TicketDetail } from '../types'

const draft = (over: Partial<Draft> = {}): Draft => ({
  id: 7, status: 'GENERATED', body: 'Please clear your SSO session and sign in again. [T-881]', citations: [{ ticketId: 881, why: 'same 403' }],
  grounding: 'STRONG', llm_confidence: 0.82, needs_info: ['Which SSO provider do you use?'], sources: [{ id: 881, subject: '403 after 4.2', score: 0.83 }],
  reused_from_draft_id: null, model: 'gemini-3.1-flash-lite', error: null, created_at: '2026-06-01T10:00:00Z', ...over,
})

const detail = (over: Partial<TicketDetail['ticket']> = {}, d: Draft | null = draft()): TicketDetail => ({
  ticket: { id: 42, external_id: null, subject: 'Cannot login', body: 'Getting 403', customer_tier: 'PRO', product: 'web-app', status: 'DRAFTED', draft_state: 'READY',
    category: 'LOGIN_ACCESS', category_conf: 0.9, category_source: 'MODEL', priority: 'HIGH', escalation_risk: 0.5, queue: 'SENIOR', cluster_id: null, cluster_size: null,
    is_incident: false, sla_due_at: null, sla_breached: false, created_at: '2026-06-01T09:00:00Z', assigned_agent_id: null, resolution: null,
    customer_external_id: 'C1', assigned_agent: null, priority_conf: 0.8, ...over },
  predictions: [], duplicates: [], draft: d, drafts: d ? [d] : [], sourceTickets: [], feedback: [],
})

let fetchMock: ReturnType<typeof vi.fn>
const calls = () => fetchMock.mock.calls.map(([url, init]) => ({ url: String(url), method: init?.method, body: init?.body ? JSON.parse(init.body as string) : undefined }))

function renderPanel(d: TicketDetail, hot: number | null = null) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}><DraftPanel detail={d} hot={hot} onHot={() => {}} /></QueryClientProvider>)
}

beforeEach(() => {
  fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200 }))
  vi.stubGlobal('fetch', fetchMock)
})

describe('DraftPanel', () => {
  it('renders citations as chips, grounding, and the questions to ask', () => {
    renderPanel(detail())
    expect(screen.getByRole('button', { name: 'T-881' })).toBeInTheDocument()
    expect(screen.getByText(/Strong grounding/)).toBeInTheDocument()
    expect(screen.getByText('Which SSO provider do you use?')).toBeInTheDocument()
  })

  it('flags a citation that is not among the retrieved sources', () => {
    renderPanel(detail({}, draft({ body: 'Try this [T-999].', sources: [{ id: 881, subject: 'x', score: 0.8 }] })))
    expect(screen.getByRole('button', { name: 'T-999' })).toHaveClass('faint')
  })

  it('approves as-is with timing and no body', async () => {
    renderPanel(detail())
    await userEvent.click(screen.getByRole('button', { name: /Approve/ }))
    await waitFor(() => expect(calls().some((c) => c.url === '/api/drafts/7/review')).toBe(true))
    const c = calls().find((x) => x.url === '/api/drafts/7/review')!
    expect(c.method).toBe('POST')
    expect(c.body.action).toBe('APPROVE')
    expect(c.body.finalBody).toBeUndefined()
    expect(typeof c.body.timeToReviewMs).toBe('number')
  })

  it('edit -> shows a diff of the changes -> sends EDIT_APPROVE with the final body', async () => {
    renderPanel(detail())
    await userEvent.click(screen.getByRole('button', { name: /Edit/ }))
    const box = screen.getByLabelText('Edit draft')
    await userEvent.clear(box)
    await userEvent.type(box, 'Please sign out everywhere and log in again.')
    await userEvent.click(screen.getByRole('button', { name: /Review changes/ }))
    const dlg = screen.getByRole('dialog')
    expect(within(dlg).getByTestId('diff')).toBeInTheDocument()
    await userEvent.click(within(dlg).getByRole('button', { name: /Send edited reply/ }))
    await waitFor(() => expect(calls().some((c) => c.body?.action === 'EDIT_APPROVE')).toBe(true))
    expect(calls().find((c) => c.body?.action === 'EDIT_APPROVE')!.body.finalBody).toBe('Please sign out everywhere and log in again.')
  })

  it('reject requires a reason', async () => {
    renderPanel(detail())
    await userEvent.click(screen.getByRole('button', { name: 'Reject' }))
    const dlg = screen.getByRole('dialog')
    const confirm = within(dlg).getByRole('button', { name: 'Reject draft' })
    expect(confirm).toBeDisabled()
    await userEvent.selectOptions(within(dlg).getByLabelText('Reject reason'), 'NOT_GROUNDED')
    expect(confirm).toBeEnabled()
    await userEvent.click(confirm)
    await waitFor(() => expect(calls().some((c) => c.body?.action === 'REJECT')).toBe(true))
    expect(calls().find((c) => c.body?.action === 'REJECT')!.body.rejectReason).toBe('NOT_GROUNDED')
  })

  it('sends label corrections and rating with the review', async () => {
    renderPanel(detail())
    await userEvent.click(screen.getByText(/Correct labels/))
    await userEvent.selectOptions(screen.getByLabelText('Correct category'), 'INTEGRATION')
    await userEvent.click(screen.getAllByRole('radio')[3])
    await userEvent.click(screen.getByRole('button', { name: /Approve/ }))
    await waitFor(() => expect(calls().some((c) => c.body?.action === 'APPROVE')).toBe(true))
    const b = calls().find((c) => c.body?.action === 'APPROVE')!.body
    expect(b.correctedCategory).toBe('INTEGRATION')
    expect(b.rating).toBe(4)
  })

  it('surfaces a server conflict (draft already reviewed)', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'draft already reviewed' }), { status: 409 }))
    renderPanel(detail())
    await userEvent.click(screen.getByRole('button', { name: /Approve/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('draft already reviewed')
  })

  it('shows the LLM-paused banner and lets the agent reply manually', async () => {
    renderPanel(detail({ draft_state: 'WAITING_LLM', status: 'TRIAGED' }, null))
    expect(screen.getByText(/Drafting is paused/)).toBeInTheDocument()
    const send = screen.getByRole('button', { name: /Send & resolve/ })
    expect(send).toBeDisabled()
    await userEvent.type(screen.getByLabelText('Manual reply'), 'We are looking into this right now.')
    await userEvent.click(send)
    await waitFor(() => expect(calls().some((c) => c.url === '/api/tickets/42/resolve')).toBe(true))
  })

  it('regenerates with an instruction', async () => {
    renderPanel(detail())
    await userEvent.type(screen.getByLabelText('Regenerate instruction'), 'more formal')
    await userEvent.click(screen.getByRole('button', { name: 'Regenerate' }))
    await waitFor(() => expect(calls().some((c) => c.url === '/api/tickets/42/draft:regenerate')).toBe(true))
    expect(calls().find((c) => c.url.endsWith('draft:regenerate'))!.body.instruction).toBe('more formal')
  })

  it('resolved tickets show the final reply and no review controls', () => {
    renderPanel(detail({ status: 'RESOLVED', resolution: 'We fixed it.' }, draft({ status: 'APPROVED' })))
    expect(screen.getByText(/Resolved/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Approve/ })).not.toBeInTheDocument()
  })
})
