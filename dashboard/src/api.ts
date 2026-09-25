// Thin fetch wrapper: bearer token from memory/sessionStorage, RFC 7807 errors surfaced as ApiError, 401 -> logout.

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

const KEY = 'ti.session'
export interface Session { accessToken: string; username: string; name: string; role: 'AGENT' | 'SENIOR' | 'ADMIN' }

let session: Session | null = null
let onUnauthorized: () => void = () => {}

export function setUnauthorizedHandler(fn: () => void) { onUnauthorized = fn }

export function loadSession(): Session | null {
  if (session) return session
  try {
    const raw = sessionStorage.getItem(KEY)
    session = raw ? (JSON.parse(raw) as Session) : null
  } catch { session = null }
  return session
}

export function saveSession(s: Session | null) {
  session = s
  try {
    if (s) sessionStorage.setItem(KEY, JSON.stringify(s))
    else sessionStorage.removeItem(KEY)
  } catch { /* storage unavailable: keep in memory only */ }
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const s = loadSession()
  if (s) headers.set('Authorization', `Bearer ${s.accessToken}`)
  let body = init.body
  if (init.json !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(init.json)
  }
  const res = await fetch(path, { ...init, headers, body })
  if (res.status === 401 && !path.startsWith('/api/auth/login')) {
    onUnauthorized()
    throw new ApiError(401, 'Session expired')
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const p = await res.json()
      msg = p.detail || p.title || msg
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, msg)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export async function login(username: string, password: string): Promise<Session> {
  const r = await api<{ accessToken: string; name: string; role: Session['role']; username: string }>('/api/auth/login', {
    method: 'POST', json: { username, password },
  })
  const s: Session = { accessToken: r.accessToken, name: r.name, role: r.role, username: r.username }
  saveSession(s)
  return s
}

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '' && v !== false) p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}
