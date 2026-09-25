import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { loadSession, login as apiLogin, saveSession, setUnauthorizedHandler, type Session } from './api'

interface AuthCtx {
  session: Session | null
  login: (u: string, p: string) => Promise<void>
  logout: () => void
}

const Ctx = createContext<AuthCtx | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(() => loadSession())
  const logout = useCallback(() => { saveSession(null); setSession(null) }, [])
  useEffect(() => { setUnauthorizedHandler(logout) }, [logout])
  const value = useMemo<AuthCtx>(() => ({
    session,
    login: async (u, p) => setSession(await apiLogin(u, p)),
    logout,
  }), [session, logout])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAuth(): AuthCtx {
  const c = useContext(Ctx)
  if (!c) throw new Error('useAuth outside AuthProvider')
  return c
}
