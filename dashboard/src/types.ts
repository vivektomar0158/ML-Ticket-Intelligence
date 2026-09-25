// Shapes returned by the Spring API. List/detail endpoints return DB column names (snake_case); auth returns camelCase.

export type Priority = 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT'

export interface TicketRow {
  id: number
  external_id: string | null
  subject: string
  customer_tier: 'FREE' | 'PRO' | 'ENTERPRISE'
  product: string
  status: string
  draft_state: 'NONE' | 'PENDING' | 'READY' | 'SKIPPED' | 'FAILED' | 'WAITING_LLM'
  category: string | null
  category_conf: number | null
  category_source: 'MODEL' | 'LLM' | 'AGENT' | null
  priority: Priority | null
  escalation_risk: number | null
  queue: 'STANDARD' | 'SENIOR'
  cluster_id: number | null
  cluster_size: number | null
  is_incident: boolean
  sla_due_at: string | null
  sla_breached: boolean
  created_at: string
  assigned_agent_id: number | null
}

export interface TicketPage {
  items: TicketRow[]
  nextCursor: string | null
}

export interface Prediction {
  task: 'CATEGORY' | 'PRIORITY' | 'ESCALATION'
  model_name: string
  model_version: string
  label: string
  score: number
  details: Record<string, unknown> | null
  created_at: string
}

export interface Citation { ticketId: number; why: string }
export interface DraftSource { id: number; subject: string; score: number }

export interface Draft {
  id: number
  status: 'GENERATED' | 'APPROVED' | 'EDITED' | 'REJECTED' | 'SKIPPED' | 'FAILED'
  body: string | null
  citations: Citation[] | null
  grounding: 'STRONG' | 'WEAK' | 'NONE' | null
  llm_confidence: number | null
  needs_info: string[] | null
  sources: DraftSource[] | null
  reused_from_draft_id: number | null
  model: string | null
  error: string | null
  created_at: string
}

export interface SourceTicket { id: number; subject: string; body: string; resolution: string | null; category: string | null }

export interface TicketDetail {
  ticket: TicketRow & { body: string; resolution: string | null; customer_external_id: string | null; assigned_agent: string | null; priority_conf: number | null }
  predictions: Prediction[]
  cluster?: { id: number; title: string; size: number; is_incident: boolean; status: string; first_seen_at: string; last_seen_at: string }
  clusterMembers?: { id: number; subject: string; status: string; created_at: string; customer_tier: string }[]
  duplicates: { id: number; similarity: number; subject: string; status: string; created_at: string }[]
  draft: Draft | null
  drafts: Draft[]
  sourceTickets: SourceTicket[]
  feedback: { id: number; action: string; edit_ratio: number | null; reject_reason: string | null; created_at: string }[]
}

export interface Counts {
  activeIncidents: number
  [queue: string]: { open: number; awaitingReview: number; breached: number } | number
}

export interface ClusterRow {
  id: number; title: string; product: string; size: number; is_incident: boolean; status: string
  first_seen_at: string; last_seen_at: string; open_tickets: number; customers: number
}

export const CATEGORIES = ['BILLING', 'LOGIN_ACCESS', 'BUG', 'FEATURE_REQUEST', 'PERFORMANCE', 'INTEGRATION', 'ACCOUNT_MANAGEMENT', 'DATA_PRIVACY']
export const PRIORITIES: Priority[] = ['LOW', 'MEDIUM', 'HIGH', 'URGENT']
export const REJECT_REASONS = ['WRONG_INFO', 'NOT_GROUNDED', 'TONE', 'INCOMPLETE', 'OTHER']
