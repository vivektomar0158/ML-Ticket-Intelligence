import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'

export function Login() {
  const { login } = useAuth()
  const nav = useNavigate()
  const [u, setU] = useState('agent')
  const [p, setP] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true); setErr(null)
    try { await login(u, p); nav('/') }
    catch { setErr('Invalid username or password') }
    finally { setBusy(false) }
  }

  return (
    <form className="card login" onSubmit={submit}>
      <h1>Sign in</h1>
      <p className="muted small">Support Ticket Intelligence — agent console</p>
      <label htmlFor="u">Username</label>
      <input id="u" value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" autoFocus />
      <label htmlFor="p">Password</label>
      <input id="p" type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" />
      {err && <div className="banner bad" role="alert">{err}</div>}
      <button className="btn primary" disabled={busy || !u || !p} style={{ width: '100%' }}>{busy ? 'Signing in…' : 'Sign in'}</button>
      <p className="muted small">Demo users: agent, senior, admin (password = username)</p>
    </form>
  )
}
