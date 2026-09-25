import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, ApiError } from '../api'

interface Result { batchId: number; total: number; accepted: number; duplicates: number; rejected: number; errors: { row: number; error: string }[] }

export function Upload() {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const m = useMutation({
    mutationFn: (f: File) => { const fd = new FormData(); fd.append('file', f); return api<Result>('/api/ingest/csv', { method: 'POST', body: fd }) },
  })
  const pick = (f?: File | null) => { if (f) { setFile(f); m.reset() } }

  return (
    <>
      <div className="page-head"><h1>Upload tickets (CSV)</h1></div>
      <div className="grid two">
        <div>
          <div className={`drop ${over ? 'over' : ''}`} onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]) }} data-testid="dropzone">
            <p><b>{file ? file.name : 'Drag a CSV file here'}</b></p>
            <p className="muted small">or</p>
            <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => pick(e.target.files?.[0])} aria-label="CSV file" />
            <button className="btn" onClick={() => input.current?.click()}>Choose file</button>
          </div>
          <div style={{ marginTop: 12 }}>
            <button className="btn primary" disabled={!file || m.isPending} onClick={() => file && m.mutate(file)}>{m.isPending ? 'Uploading…' : 'Upload & triage'}</button>
          </div>
          {m.isError && <div className="banner bad" style={{ marginTop: 12 }} role="alert">{m.error instanceof ApiError ? m.error.message : 'Upload failed'}</div>}
          {m.data && (
            <div className="card" style={{ marginTop: 12 }} data-testid="upload-result">
              <h2>Batch #{m.data.batchId}</h2>
              <div className="row">
                <span className="chip good">{m.data.accepted} accepted</span>
                <span className="chip">{m.data.duplicates} duplicates skipped</span>
                <span className={`chip ${m.data.rejected ? 'bad' : ''}`}>{m.data.rejected} rejected</span>
                <span className="muted small">of {m.data.total} rows</span>
              </div>
              {!!m.data.errors.length && (
                <table style={{ marginTop: 10 }}>
                  <thead><tr><th>Line</th><th>Problem</th></tr></thead>
                  <tbody>{m.data.errors.map((e) => <tr key={e.row} style={{ cursor: 'default' }}><td className="mono">{e.row}</td><td>{e.error}</td></tr>)}</tbody>
                </table>
              )}
            </div>
          )}
        </div>
        <div className="card">
          <h2>Format</h2>
          <p className="small">Header row required. Bad rows are reported, never fail the whole file.</p>
          <table>
            <thead><tr><th>Column</th><th></th></tr></thead>
            <tbody>
              {[['subject', 'required, ≤ 300 chars'], ['body', 'required'], ['product', 'web-app · mobile-app · api · desktop-sync · admin-console'],
                ['customer_tier', 'FREE · PRO · ENTERPRISE'], ['ticket_uid / external_id', 'optional; re-uploads skip duplicates'], ['created_at', 'optional ISO timestamp'], ['customer_id', 'optional']].map(([a, b]) => (
                <tr key={a} style={{ cursor: 'default' }}><td className="mono">{a}</td><td className="small muted">{b}</td></tr>))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
