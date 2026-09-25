package com.ticketintel.admin;

import com.ticketintel.common.ApiException;
import com.ticketintel.common.Util;
import com.ticketintel.events.SseHub;
import com.ticketintel.jobs.JobRepository;
import com.ticketintel.mlclient.MlClient;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;
import tools.jackson.databind.json.JsonMapper;

@RestController
@RequestMapping("/api/admin")
public class AdminController {

    public record HistoryItem(String externalId, Instant createdAt, String customerId, String customerTier, String product,
                              String subject, String body, String resolution, String category, String priority, float[] embedding) {}

    private final JdbcTemplate jdbc;
    private final TransactionTemplate tx;
    private final JobRepository jobs;
    private final MlClient ml;
    private final SseHub events;
    private final JsonMapper json;

    public AdminController(JdbcTemplate jdbc, TransactionTemplate tx, JobRepository jobs, MlClient ml, SseHub events, JsonMapper json) {
        this.jdbc = jdbc;
        this.tx = tx;
        this.jobs = jobs;
        this.ml = ml;
        this.events = events;
        this.json = json;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        return Map.of("mlFastCircuit", ml.fastState().name(), "mlLlmCircuit", ml.llmState().name(),
                "jobs", jobs.depthByStatus(), "sseSubscribers", events.subscribers());
    }

    @GetMapping("/jobs")
    public List<Map<String, Object>> deadJobs(@RequestParam(defaultValue = "50") int limit) {
        return jobs.dead(Math.min(limit, 200));
    }

    @PostMapping("/jobs/{id}:retry")
    public Map<String, Object> retry(@PathVariable long id) {
        if (!jobs.retryDead(id)) throw ApiException.notFound("dead job");
        return Map.of("retried", id);
    }

    /**
     * Bulk-loads already-resolved historical tickets (with resolution + embedding) as the RAG corpus.
     * Used by scripts/seed.py; idempotent on externalId.
     */
    @PostMapping("/history")
    public Map<String, Object> importHistory(@RequestBody List<HistoryItem> items) {
        if (items.isEmpty() || items.size() > 500) throw ApiException.badRequest("1..500 items per call");
        int[] counts = {0, 0};
        Map<String, Long> customers = new HashMap<>();
        tx.executeWithoutResult(s -> {
            for (HistoryItem h : items) {
                if (h.embedding() == null || h.embedding().length != 384 || h.resolution() == null || h.subject() == null || h.body() == null
                        || !Util.PRODUCTS.contains(h.product()) || !Util.TIERS.contains(h.customerTier())) { counts[1]++; continue; }
                Long cust = h.customerId() == null ? null : customers.computeIfAbsent(h.customerId(), k -> jdbc.queryForObject(
                        "INSERT INTO customers (external_id, tier) VALUES (?, ?) ON CONFLICT (external_id) DO UPDATE SET tier = EXCLUDED.tier RETURNING id",
                        Long.class, k, h.customerTier()));
                List<Long> ids = jdbc.queryForList("""
                        INSERT INTO tickets (external_id, source, customer_id, customer_tier, product, subject, body, content_hash, created_at, status,
                                             category, category_conf, category_source, priority, priority_source, resolution, resolved_at, first_response_at, draft_state)
                        VALUES (?, 'SEED', ?, ?, ?, ?, ?, ?, ?, 'RESOLVED', ?, 1, 'AGENT', ?, 'AGENT', ?, ? + interval '1 hour', ? + interval '1 hour', 'READY')
                        ON CONFLICT (external_id) DO NOTHING RETURNING id
                        """, Long.class, h.externalId(), cust, h.customerTier(), h.product(), h.subject(), h.body(), Util.sha256(h.subject() + "\n" + h.body()),
                        Util.utc(h.createdAt()), h.category(), h.priority(), h.resolution(), Util.utc(h.createdAt()), Util.utc(h.createdAt()));
                if (ids.isEmpty()) { counts[1]++; continue; }
                jdbc.update("""
                        INSERT INTO ticket_embeddings (ticket_id, model_version, is_resolved, is_open, product, created_at, embedding)
                        VALUES (?, 'minilm-l6-v2@1', true, false, ?, ?, CAST(? AS vector))
                        """, ids.get(0), h.product(), Util.utc(h.createdAt()), Util.vec(h.embedding()));
                counts[0]++;
            }
        });
        return Map.of("imported", counts[0], "skipped", counts[1]);
    }

    /**
     * Retraining export (NDJSON): agent-corrected labels override model labels; escalation outcomes come from real
     * SLA/escalation behaviour. Consumed by ml-service/training/retrain.py.
     */
    @GetMapping(value = "/export/training-data", produces = "application/x-ndjson")
    public StreamingResponseBody export(@RequestParam(required = false) Instant since) {
        Instant from = since == null ? Instant.EPOCH : since;
        return out -> jdbc.query("""
                SELECT t.id, t.subject, t.body, t.customer_tier, t.product, t.category, t.category_source, t.priority, t.priority_source,
                       t.escalation_risk, t.sla_breached, t.queue = 'SENIOR' AS senior_queue, t.created_at,
                       (SELECT corrected_category FROM feedback f WHERE f.ticket_id = t.id AND corrected_category IS NOT NULL ORDER BY f.created_at DESC LIMIT 1) AS corrected_category,
                       (SELECT corrected_priority FROM feedback f WHERE f.ticket_id = t.id AND corrected_priority IS NOT NULL ORDER BY f.created_at DESC LIMIT 1) AS corrected_priority
                FROM tickets t WHERE t.created_at >= ? AND t.source <> 'SEED' AND t.category IS NOT NULL ORDER BY t.id
                """, rs -> {
            Map<String, Object> row = new java.util.LinkedHashMap<>();
            var md = rs.getMetaData();
            for (int i = 1; i <= md.getColumnCount(); i++) row.put(md.getColumnLabel(i), rs.getObject(i));
            try {
                out.write((json.writeValueAsString(row) + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }, Util.utc(from));
    }
}
