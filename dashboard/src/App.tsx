import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider, useAuth } from './auth'
import { Layout } from './components/Layout'
import { Incidents } from './pages/Incidents'
import { Login } from './pages/Login'
import { Metrics } from './pages/Metrics'
import { Queue } from './pages/Queue'
import { Ticket } from './pages/Ticket'
import { Upload } from './pages/Upload'
import type { ReactElement } from 'react'

const client = new QueryClient({ defaultOptions: { queries: { staleTime: 5000, retry: 1, refetchOnWindowFocus: false } } })

function Guard({ children }: { children: ReactElement }) {
  const { session } = useAuth()
  return session ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <QueryClientProvider client={client}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route element={<Guard><Layout /></Guard>}>
              <Route index element={<Queue />} />
              <Route path="tickets/:id" element={<Ticket />} />
              <Route path="incidents" element={<Incidents />} />
              <Route path="upload" element={<Upload />} />
              <Route path="metrics" element={<Metrics />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  )
}
