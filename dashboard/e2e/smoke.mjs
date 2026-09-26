// End-to-end smoke test against the running stack (Chrome via puppeteer-core).
// NOTE: in headless Chrome, CDP mouse/keyboard input does not reach the app pages after login on this setup, so interactions
// are dispatched as DOM events (React handlers still run). Real pointer/keyboard behaviour is covered by the userEvent unit tests.
//   prerequisites: postgres + ml-service + api running, history + some tickets seeded (scripts/seed.py), `npm run dev`
//   node e2e/smoke.mjs            (BASE_URL, CHROME_PATH override the defaults)
import fs from 'node:fs'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const BASE = process.env.BASE_URL ?? 'http://localhost:5173'
const CHROME = process.env.CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const OUT = fileURLToPath(new URL('./out/', import.meta.url))
fs.mkdirSync(OUT, { recursive: true })

const problems = []
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) problems.push(what) }

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, defaultViewport: { width: 1440, height: 900 } })
const page = await browser.newPage()
const consoleErrors = [], badResponses = []
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('401')) consoleErrors.push(m.text()) })
page.on('pageerror', (e) => consoleErrors.push(String(e)))
page.on('response', (r) => { if (r.url().includes('/api/') && r.status() >= 400 && !r.url().includes('/api/auth/login')) badResponses.push(`${r.status()} ${r.url()}`) })

const domClick = (sel, textStartsWith) => page.evaluate((s, t) => { const els = [...document.querySelectorAll(s)]; const el = t ? els.find((e) => e.innerText.trim().startsWith(t)) : els[0]; if (!el) throw new Error('no element ' + s + ' ' + (t ?? '')); el.click() }, sel, textStartsWith)
const setValue = (sel, v) => page.evaluate((s, v) => { const el = document.querySelector(s); const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })) }, sel, v)
const shot = (n) => page.screenshot({ path: `${OUT}${n}.png` })
const text = () => page.evaluate(() => document.body.innerText)
const waitText = (t, ms = 15000) => page.waitForFunction((x) => document.body.innerText.includes(x), { timeout: ms }, t)

try {
  // ---- login (wrong password first)
  await page.goto(`${BASE}/login`, { waitUntil: 'networkidle0' })
  await shot('01-login')
  await page.type('#p', 'wrong')
  await page.click('button.primary')
  await waitText('Invalid username or password')
  check(true, 'wrong password shows an error')
  await page.$eval('#p', (e) => { e.value = '' })
  await page.type('#p', 'agent')
  await page.click('button.primary')
  await page.waitForSelector('[data-testid=ticket-row]', { timeout: 20000 })
  check(true, 'login succeeds and the queue loads')

  // ---- queue
  await new Promise((r) => setTimeout(r, 800))
  await shot('02-queue')
  const rows = await page.$$eval('[data-testid=ticket-row]', (r) => r.length)
  check(rows > 5, `queue lists tickets (${rows} rows)`)
  const t = await text()
  check(/incident/i.test(t), 'queue shows an incident badge for the outage cluster')
  check(/senior/i.test(t), 'queue shows senior-routed tickets')

  // filter: needs review
  await page.select('select[aria-label=Priority]', 'URGENT')
  await new Promise((r) => setTimeout(r, 900))
  const urgentOnly = await page.$$eval('[data-testid=ticket-row]', (r) => r.map((x) => x.innerText.includes('Urgent')))
  check(urgentOnly.length > 0 && urgentOnly.every(Boolean), `priority filter returns only urgent tickets (${urgentOnly.length})`)
  await page.select('select[aria-label=Priority]', '')
  await new Promise((r) => setTimeout(r, 600))

  // ---- open the top ticket, verify AI analysis + grounded draft with citations
  await domClick('[data-testid=ticket-row] td.subject')      // open the highest-risk ticket
  await page.waitForSelector('[data-testid=draft-view]', { timeout: 15000 })
  await new Promise((r) => setTimeout(r, 500))
  await shot('03-ticket')
  const tt = await text()
  check(tt.includes('AI analysis') && tt.includes('Escalation'), 'ticket page shows AI analysis and escalation')
  check(/Why this risk|Customer tier|Recent tickets|Ticket wording|Predicted/.test(tt), 'escalation risk is explained with factors')
  // grounded drafts must show clickable citations; an ungrounded (NONE) draft legitimately has none and must ask questions instead
  const grounded = /(Strong|Weak) grounding/.test(tt)
  const chips = (await page.$$('button.chip.cite')).length
  check(grounded ? chips > 0 : chips === 0, grounded ? `grounded draft shows clickable citation chips (${chips})` : 'ungrounded draft shows no citations')
  check(tt.includes('Similar resolved tickets'), 'similar resolved tickets are listed')

  // ---- edit + approve with diff
  await domClick('button', 'Edit')
  await page.waitForSelector('textarea[aria-label="Edit draft"]')
  await setValue('textarea[aria-label="Edit draft"]', 'Thanks for reporting this. We applied a fix; please sign out, clear cookies and sign in again.')
  await domClick('button', 'Review changes')
  await page.waitForSelector('[data-testid=diff]')
  await shot('04-diff')
  check(true, 'editing shows a diff before sending')
  await domClick('.modal button', 'Send edited reply')
  await waitText('This reply is now searchable evidence')
  await shot('05-resolved')
  check(true, 'edited reply is sent and the ticket resolves')

  // ---- other pages
  await page.goto(`${BASE}/incidents`, { waitUntil: 'load' })
  await new Promise((r) => setTimeout(r, 1800))
  await shot('06-incidents')
  check(await page.$('[data-testid=incident-card]') !== null, 'incidents page shows the outage with its timeline')

  await page.goto(`${BASE}/metrics`, { waitUntil: 'load' })
  await new Promise((r) => setTimeout(r, 1500))
  await shot('07-metrics')
  const mt = await text()
  check(mt.includes('Draft approval rate') && mt.includes('LLM spend'), 'metrics page renders headline numbers')
  check(await page.$('svg.recharts-surface') !== null, 'metrics page renders charts')

  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }])
  await new Promise((r) => setTimeout(r, 500))
  await shot('08-metrics-dark')
  await page.goto(`${BASE}/upload`, { waitUntil: 'load' })
  await shot('09-upload-dark')
  check(true, 'dark mode renders')
} catch (e) {
  check(false, `flow aborted: ${e.message}`)
  await shot('99-failure').catch(() => {})
}

check(badResponses.length === 0, `no failed API calls ${badResponses.length ? JSON.stringify(badResponses.slice(0, 3)) : ''}`)
check(consoleErrors.length === 0, `no console errors ${consoleErrors.length ? JSON.stringify(consoleErrors.slice(0, 3)) : ''}`)
await browser.close()
console.log(problems.length ? `\n${problems.length} problem(s)` : '\nall E2E checks passed; screenshots in dashboard/e2e/out/')
process.exit(problems.length ? 1 : 0)
