import { describe, expect, it } from 'vitest'
import { diffStats, factorLabel, label, pct, riskTone, slaState, splitCitations, timeAgo } from '../lib'

describe('splitCitations', () => {
  it('turns [T-id] tags into cite parts and keeps surrounding text', () => {
    expect(splitCitations('Clear cache [T-881] then retry [T-12].')).toEqual([
      { type: 'text', text: 'Clear cache ' }, { type: 'cite', id: 881 }, { type: 'text', text: ' then retry ' }, { type: 'cite', id: 12 }, { type: 'text', text: '.' },
    ])
  })
  it('handles text without citations and adjacent tags', () => {
    expect(splitCitations('plain')).toEqual([{ type: 'text', text: 'plain' }])
    expect(splitCitations('[T-1][T-2]')).toEqual([{ type: 'cite', id: 1 }, { type: 'cite', id: 2 }])
    expect(splitCitations('')).toEqual([])
  })
  it('does not treat malformed tags as citations', () => {
    expect(splitCitations('see [T-abc] and [T-]')).toEqual([{ type: 'text', text: 'see [T-abc] and [T-]' }])
  })
})

describe('slaState', () => {
  const now = Date.parse('2026-06-01T12:00:00Z')
  it('counts down and warns inside 15 minutes', () => {
    expect(slaState('2026-06-01T14:10:00Z', 'TRIAGED', false, now)).toEqual({ label: 'in 2h 10m', tone: 'good' })
    expect(slaState('2026-06-01T12:10:00Z', 'TRIAGED', false, now)).toEqual({ label: 'in 10m', tone: 'warn' })
  })
  it('shows overdue in red, and stays overdue when flagged breached', () => {
    expect(slaState('2026-06-01T11:25:00Z', 'TRIAGED', false, now)).toEqual({ label: 'overdue 35m', tone: 'bad' })
    expect(slaState('2026-06-01T13:00:00Z', 'TRIAGED', true, now).tone).toBe('bad')
  })
  it('has no pressure for resolved tickets or missing SLA, and formats days', () => {
    expect(slaState('2026-06-01T11:00:00Z', 'RESOLVED', true, now).label).toBe('done')
    expect(slaState(null, 'NEW', false, now).label).toBe('—')
    expect(slaState('2026-06-03T15:00:00Z', 'TRIAGED', false, now).label).toBe('in 2d 3h')
  })
})

describe('formatting helpers', () => {
  it('timeAgo', () => {
    const now = Date.parse('2026-06-01T12:00:00Z')
    expect(timeAgo('2026-06-01T11:59:30Z', now)).toBe('30s ago')
    expect(timeAgo('2026-06-01T11:00:00Z', now)).toBe('1h ago')
    expect(timeAgo('2026-05-30T12:00:00Z', now)).toBe('2d ago')
  })
  it('pct / label / riskTone / factorLabel', () => {
    expect(pct(0.934)).toBe('93%')
    expect(pct(null)).toBe('—')
    expect(label('LOGIN_ACCESS')).toBe('Login access')
    expect(riskTone(0.45)).toBe('high')
    expect(riskTone(0.1)).toBe('')
    expect(factorLabel('prior_tickets_7d')).toBe('Recent tickets from customer')
    expect(factorLabel('p_cat_BILLING')).toBe('Category: BILLING')
  })
})

describe('diffStats', () => {
  it('counts added and removed words', () => {
    const s = diffStats('Please clear your cache and retry.', 'Please clear your browser cache, then retry today.')
    expect(s.changed).toBe(true)
    expect(s.added).toBeGreaterThan(0)
    expect(s.removed).toBeGreaterThan(0)
  })
  it('reports unchanged text (ignoring surrounding whitespace)', () => {
    expect(diffStats('same text', ' same text ')).toEqual({ added: 0, removed: 0, changed: false })
  })
})
