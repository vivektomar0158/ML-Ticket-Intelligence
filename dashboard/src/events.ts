import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { loadSession } from './api'

const EVENTS = ['ticket.created', 'ticket.triaged', 'ticket.drafted', 'ticket.updated', 'ticket.resolved', 'incident.detected', 'cluster.updated', 'sla.breached']

export interface LiveEvent { type: string; data: Record<string, unknown> }

/**
 * Server-sent events -> React Query invalidation (one invalidation per animation burst, so 100 events/s do not cause 100 refetches).
 * Reconnects with backoff. Returns connection state and the most recent incident alert for the banner.
 */
export function useLiveEvents(enabled: boolean) {
  const qc = useQueryClient()
  const [connected, setConnected] = useState(false)
  const [incident, setIncident] = useState<LiveEvent | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    if (!enabled) return
    let es: EventSource | null = null
    let retry = 1000
    let stopped = false
    const flush = () => {
      timer.current = undefined
      qc.invalidateQueries({ queryKey: ['tickets'] })
      qc.invalidateQueries({ queryKey: ['counts'] })
      qc.invalidateQueries({ queryKey: ['clusters'] })
    }
    const connect = () => {
      const token = loadSession()?.accessToken
      if (!token || stopped) return
      es = new EventSource(`/api/events?access_token=${encodeURIComponent(token)}`)
      es.onopen = () => { setConnected(true); retry = 1000 }
      es.onerror = () => {
        setConnected(false)
        es?.close()
        if (!stopped) window.setTimeout(connect, (retry = Math.min(retry * 2, 15000)))
      }
      for (const name of EVENTS) {
        es.addEventListener(name, (e) => {
          let data: Record<string, unknown> = {}
          try { data = JSON.parse((e as MessageEvent).data) } catch { /* ignore malformed */ }
          if (name === 'incident.detected') setIncident({ type: name, data })
          const id = data.ticketId
          if (typeof id === 'number') qc.invalidateQueries({ queryKey: ['ticket', id] })
          if (timer.current === undefined) timer.current = window.setTimeout(flush, 400)
        })
      }
    }
    connect()
    return () => { stopped = true; es?.close(); if (timer.current) window.clearTimeout(timer.current) }
  }, [enabled, qc])

  return { connected, incident, dismissIncident: () => setIncident(null) }
}
