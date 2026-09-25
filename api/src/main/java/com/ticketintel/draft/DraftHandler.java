package com.ticketintel.draft;

import com.ticketintel.common.AppProperties;
import com.ticketintel.common.Util;
import com.ticketintel.events.SseHub;
import com.ticketintel.jobs.JobHandler;
import com.ticketintel.jobs.JobRepository;
import com.ticketintel.jobs.JobRepository.Job;
import com.ticketintel.mlclient.MlClient;
import com.ticketintel.mlclient.MlClient.LlmUnavailableException;
import com.ticketintel.mlclient.MlDtos.*;
import com.ticketintel.retrieval.HybridRetriever;
import com.ticketintel.retrieval.HybridRetriever.Hit;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionTemplate;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * DRAFT job (slow path): retrieve similar resolved tickets -> LLM drafts a grounded reply -> store for agent review.
 * Cost/latency controls: incident members reuse one canonical draft, low-value tickets are skipped under backlog, and an
 * unavailable LLM defers the job (without burning attempts) instead of failing it.
 */
@Component
public class DraftHandler implements JobHandler {
    private static final Logger log = LoggerFactory.getLogger(DraftHandler.class);

    private final JdbcTemplate jdbc;
    private final TransactionTemplate tx;
    private final MlClient ml;
    private final HybridRetriever retriever;
    private final JobRepository jobs;
    private final SseHub events;
    private final JsonMapper json;
    private final AppProperties props;

    public DraftHandler(JdbcTemplate jdbc, TransactionTemplate tx, MlClient ml, HybridRetriever retriever, JobRepository jobs,
                        SseHub events, JsonMapper json, AppProperties props) {
        this.jdbc = jdbc;
        this.tx = tx;
        this.ml = ml;
        this.retriever = retriever;
        this.jobs = jobs;
        this.events = events;
        this.json = json;
        this.props = props;
    }

    @Override public String type() { return "DRAFT"; }
    @Override public int batchSize() { return 1; }
    @Override public int threads() { return props.workers().draftThreads(); }

    @Override
    public void handle(List<Job> batch, Context ctx) {
        for (Job job : batch) {
            List<Map<String, Object>> rows = jdbc.queryForList("""
                    SELECT t.id, t.subject, t.body, t.category, t.priority, t.escalation_risk, t.cluster_id, t.status, t.created_at,
                           c.is_incident, c.size AS cluster_size
                    FROM tickets t LEFT JOIN duplicate_clusters c ON c.id = t.cluster_id WHERE t.id = ?
                    """, job.ticketId());
            if (rows.isEmpty()) continue;
            var t = rows.get(0);
            String status = (String) t.get("status");
            if (status.equals("RESOLVED") || status.equals("CLOSED") || status.equals("NEW")) continue;
            String instruction = instruction(job);
            boolean incident = Boolean.TRUE.equals(t.get("is_incident"));
            Long clusterId = (Long) t.get("cluster_id");
            double risk = t.get("escalation_risk") == null ? 0 : ((Number) t.get("escalation_risk")).doubleValue();

            // 1) load shedding: under backlog, low-value tickets get "generate on demand" instead of a draft
            if (instruction == null && "LOW".equals(t.get("priority")) && risk < 0.2
                    && jobs.pendingDepth("DRAFT") > props.draft().shedQueueDepth()) {
                skip(job.ticketId(), "shed: draft queue backlog");
                continue;
            }
            // 2) incident reuse: one canonical draft for the whole outage instead of N LLM calls
            if (instruction == null && incident && clusterId != null && reuse(job.ticketId(), clusterId)) continue;

            // 3) retrieval (RAG, retrieval half)
            String emb = jdbc.queryForObject("SELECT embedding::text FROM ticket_embeddings WHERE ticket_id = ?", String.class, job.ticketId());
            List<Hit> hits = retriever.retrieve(job.ticketId(), Util.parseVec(emb), (String) t.get("subject"), (String) t.get("body"));
            var sources = hits.stream().map(h -> new Source(h.id(), h.subject(), h.problem(), h.resolution(), h.score())).toList();

            // 4) generation (RAG, generation half)
            DraftRes res;
            try {
                res = ml.draft(new DraftReq(new DraftTicket((String) t.get("subject"), (String) t.get("body"), (String) t.get("category"),
                        (String) t.get("priority"), incident, t.get("cluster_size") == null ? 1 : ((Number) t.get("cluster_size")).intValue()),
                        sources, instruction));
            } catch (LlmUnavailableException e) {
                Instant created = ((java.time.OffsetDateTime) jdbc.queryForObject("SELECT ingested_at FROM tickets WHERE id = ?",
                        java.time.OffsetDateTime.class, job.ticketId())).toInstant();
                if (Duration.between(created, Instant.now()).toHours() >= props.draft().maxWaitHours()) {
                    skip(job.ticketId(), "LLM unavailable for " + props.draft().maxWaitHours() + "h");
                } else {
                    jdbc.update("UPDATE tickets SET draft_state = 'WAITING_LLM' WHERE id = ? AND draft_state = 'PENDING'", job.ticketId());
                    ctx.defer(job, 60, "LLM unavailable");
                }
                continue;
            }
            store(job.ticketId(), res, sources, instruction);
        }
    }

    private boolean reuse(long ticketId, long clusterId) {
        List<Map<String, Object>> src = jdbc.queryForList("""
                SELECT d.id, d.body, d.citations::text AS citations, d.grounding, d.llm_confidence, d.needs_info::text AS needs_info, d.sources::text AS sources
                FROM drafts d JOIN tickets t ON t.id = d.ticket_id
                WHERE t.cluster_id = ? AND t.id <> ? AND d.status IN ('GENERATED','APPROVED','EDITED') AND d.reused_from_draft_id IS NULL
                  AND d.created_at > now() - make_interval(hours => ?) ORDER BY d.created_at DESC LIMIT 1
                """, clusterId, ticketId, props.draft().incidentReuseHours());
        if (src.isEmpty()) return false;
        var d = src.get(0);
        tx.executeWithoutResult(s -> {
            jdbc.update("UPDATE drafts SET status = 'SUPERSEDED' WHERE ticket_id = ? AND status = 'GENERATED'", ticketId);
            jdbc.update("""
                    INSERT INTO drafts (ticket_id, status, body, citations, grounding, llm_confidence, needs_info, sources, reused_from_draft_id, model, prompt_version, cost_usd)
                    VALUES (?, 'GENERATED', ?, CAST(? AS jsonb), ?, ?, CAST(? AS jsonb), CAST(? AS jsonb), ?, 'incident-reuse', 'reuse', 0)
                    """, ticketId, d.get("body"), d.get("citations"), d.get("grounding"), d.get("llm_confidence"), d.get("needs_info"),
                    d.get("sources"), d.get("id"));
            markDrafted(ticketId);
        });
        return true;
    }

    private void store(long ticketId, DraftRes r, List<Source> sources, String instruction) {
        var srcJson = sources.stream().map(s -> Map.of("id", s.id(), "subject", s.subject(), "score", s.score())).toList();
        tx.executeWithoutResult(s -> {
            jdbc.update("UPDATE drafts SET status = 'SUPERSEDED' WHERE ticket_id = ? AND status = 'GENERATED'", ticketId);
            jdbc.update("""
                    INSERT INTO drafts (ticket_id, status, body, citations, grounding, llm_confidence, needs_info, sources, instruction,
                                        model, prompt_version, input_tokens, output_tokens, latency_ms, cost_usd)
                    VALUES (?, 'GENERATED', ?, CAST(? AS jsonb), ?, ?, CAST(? AS jsonb), CAST(? AS jsonb), ?, ?, ?, ?, ?, ?, ?)
                    """, ticketId, r.reply(), json.writeValueAsString(r.citations()), r.grounding(), (float) r.confidence(),
                    json.writeValueAsString(r.needsInfo()), json.writeValueAsString(srcJson), instruction, r.meta().model(),
                    r.meta().promptVersion(), r.meta().inputTokens(), r.meta().outputTokens(), r.meta().latencyMs(), r.meta().costUsd());
            markDrafted(ticketId);
        });
    }

    private void markDrafted(long ticketId) {
        jdbc.update("UPDATE tickets SET status = CASE WHEN status IN ('TRIAGED','DRAFTED','TRIAGE_FAILED') THEN 'DRAFTED' ELSE status END, draft_state = 'READY', drafted_at = now(), version = version + 1 WHERE id = ?", ticketId);
        events.publish("ticket.drafted", Map.of("ticketId", ticketId));
    }

    private void skip(long ticketId, String why) {
        tx.executeWithoutResult(s -> {
            jdbc.update("INSERT INTO drafts (ticket_id, status, error) VALUES (?, 'SKIPPED', ?)", ticketId, why);
            jdbc.update("UPDATE tickets SET draft_state = 'SKIPPED', version = version + 1 WHERE id = ?", ticketId);
        });
        events.publish("ticket.updated", Map.of("ticketId", ticketId, "draftState", "SKIPPED"));
        log.info("draft skipped for ticket {}: {}", ticketId, why);
    }

    private String instruction(Job job) {
        if (job.payload() == null) return null;
        JsonNode n = json.readTree(job.payload()).path("instruction");
        String s = n.isMissingNode() || n.isNull() ? null : n.stringValue(null);
        return s == null || s.isBlank() ? null : s;
    }
}
