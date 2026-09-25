"""Request/response contract of the ML service (the OpenAPI schema is exported to docs/contracts/)."""
from typing import Optional

from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    id: int
    subject: str = Field(max_length=500)
    body: str = Field(max_length=20000)
    tier: str = "FREE"


class AnalyzeRequest(BaseModel):
    tickets: list[TicketIn] = Field(min_length=1, max_length=128)
    includeEmbedding: bool = True


class CategoryPred(BaseModel):
    label: str
    confidence: float
    model: str
    probs: dict[str, float]


class PriorityPred(BaseModel):
    label: str
    confidence: float
    model: str
    signals: list[str]
    probs: dict[str, float]


class AnalyzeItem(BaseModel):
    ticketId: int
    embedding: Optional[list[float]] = None
    embeddingModel: str
    category: CategoryPred
    priority: PriorityPred
    needsLlm: bool


class AnalyzeResponse(BaseModel):
    items: list[AnalyzeItem]
    latencyMs: float


class EscalationIn(BaseModel):
    id: int
    subject: str
    body: str
    tier: str = "FREE"
    product: str = "web-app"
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    priorTickets7d: int = 0
    similarRecent: int = 0            # tickets in the last 3h, same product, cosine >= tau (duplicate-cluster size)
    embedding: Optional[list[float]] = None


class EscalationRequest(BaseModel):
    tickets: list[EscalationIn] = Field(min_length=1, max_length=128)


class Contribution(BaseModel):
    feature: str
    contribution: float               # log-odds contribution; positive pushes risk up


class EscalationItem(BaseModel):
    ticketId: int
    risk: float                       # calibrated probability
    rawScore: float
    route: str                        # SENIOR | STANDARD
    topFactors: list[Contribution]
    model: str


class EscalationResponse(BaseModel):
    items: list[EscalationItem]
    seniorThreshold: float


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str


class LlmClassifyRequest(BaseModel):
    subject: str
    body: str


class LlmMeta(BaseModel):
    model: str
    latencyMs: int
    inputTokens: int
    outputTokens: int
    costUsd: float
    cached: bool
    promptVersion: str


class LlmClassifyResponse(BaseModel):
    category: str
    priority: str
    confidence: float
    rationale: str
    meta: LlmMeta


class Source(BaseModel):
    id: int
    subject: str
    problem: str
    resolution: str
    score: float = 0.0               # retrieval score (cosine) - used to decide grounding strength


class DraftTicket(BaseModel):
    subject: str
    body: str
    category: Optional[str] = None
    priority: Optional[str] = None
    isIncident: bool = False
    clusterSize: int = 1


class DraftRequest(BaseModel):
    ticket: DraftTicket
    sources: list[Source] = Field(default_factory=list, max_length=8)
    instruction: Optional[str] = Field(default=None, max_length=300)


class CitationOut(BaseModel):
    ticketId: int
    why: str


class DraftResponse(BaseModel):
    reply: str
    citations: list[CitationOut]
    confidence: float
    needsInfo: list[str]
    grounding: str                    # STRONG | WEAK | NONE
    strippedCitations: int            # invalid ids the model invented and we removed
    meta: LlmMeta


class JudgeRequest(BaseModel):
    ticket: DraftTicket
    draft: str
    sources: list[Source] = Field(default_factory=list)
    goldFix: str


class JudgeResponse(BaseModel):
    groundedness: int
    correctness: int
    completeness: int
    tone: int
    rationale: str
    meta: LlmMeta
