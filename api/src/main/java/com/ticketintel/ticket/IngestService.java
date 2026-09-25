package com.ticketintel.ticket;

import com.ticketintel.common.ApiException;
import com.ticketintel.common.Util;
import com.ticketintel.events.SseHub;
import com.ticketintel.jobs.JobRepository;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class IngestService {

    public record TicketRequest(String externalId, String subject, String body, String product, String customerId,
                                String customerTier, Instant createdAt) {}

    public record IngestResult(Long id, String status, String error) {
        static IngestResult created(long id) { return new IngestResult(id, "CREATED", null); }
        static IngestResult duplicate(long id) { return new IngestResult(id, "DUPLICATE", null); }
        static IngestResult rejected(String why) { return new IngestResult(null, "REJECTED", why); }
    }

    private final JdbcTemplate jdbc;
    private final JobRepository jobs;
    private final SseHub events;

    public IngestService(JdbcTemplate jdbc, JobRepository jobs, SseHub events) {
        this.jdbc = jdbc;
        this.jobs = jobs;
        this.events = events;
    }

    /** Field validation shared by API, batch and CSV paths. Returns null when valid. */
    public static String validate(TicketRequest r) {
        if (r.subject() == null || r.subject().isBlank()) return "subject is required";
        if (r.subject().length() > 300) return "subject exceeds 300 chars";
        if (r.body() == null || r.body().isBlank()) return "body is required";
        if (r.body().length() > 20000) return "body exceeds 20000 chars";
        if (r.product() == null || !Util.PRODUCTS.contains(r.product())) return "product must be one of " + Util.PRODUCTS;
        if (r.customerTier() == null || !Util.TIERS.contains(r.customerTier())) return "customerTier must be one of " + Util.TIERS;
        if (r.customerId() != null && r.customerId().length() > 100) return "customerId too long";
        return null;
    }

    /** Ticket row + TRIAGE job in ONE transaction: never a ticket without its job, never a job without its ticket. */
    @Transactional
    public IngestResult ingest(TicketRequest r, String source, Long batchId, Map<String, Long> customerCache) {
        String err = validate(r);
        if (err != null) return IngestResult.rejected(err);
        Long customerId = null;
        if (r.customerId() != null && !r.customerId().isBlank()) {
            customerId = customerCache != null ? customerCache.get(r.customerId()) : null;
            if (customerId == null) {
                customerId = jdbc.queryForObject(
                        "INSERT INTO customers (external_id, tier) VALUES (?, ?) ON CONFLICT (external_id) DO UPDATE SET tier = EXCLUDED.tier RETURNING id",
                        Long.class, r.customerId(), r.customerTier());
                if (customerCache != null) customerCache.put(r.customerId(), customerId);
            }
        }
        Instant created = r.createdAt() != null ? r.createdAt() : Instant.now();
        List<Long> ids = jdbc.queryForList("""
                INSERT INTO tickets (external_id, source, ingest_batch_id, customer_id, customer_tier, product, subject, body, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (external_id) DO NOTHING RETURNING id
                """, Long.class, r.externalId(), source, batchId, customerId, r.customerTier(), r.product(), r.subject().trim(),
                r.body().trim(), Util.sha256(r.subject() + "\n" + r.body()), Util.utc(created));
        if (ids.isEmpty()) {
            Long existing = jdbc.queryForObject("SELECT id FROM tickets WHERE external_id = ?", Long.class, r.externalId());
            return IngestResult.duplicate(existing);
        }
        long id = ids.get(0);
        int prio = switch (r.customerTier()) { case "ENTERPRISE" -> 20; case "PRO" -> 10; default -> 0; };
        jobs.enqueue("TRIAGE", id, prio, null);
        events.publish("ticket.created", Map.of("ticketId", id));
        return IngestResult.created(id);
    }

    public Map<String, Object> batchStatus(long id) {
        List<Map<String, Object>> rows = jdbc.queryForList("SELECT id, filename, status, total, accepted, duplicates, rejected, errors::text AS errors, created_at FROM ingest_batches WHERE id = ?", id);
        if (rows.isEmpty()) throw ApiException.notFound("batch");
        return new HashMap<>(rows.get(0));
    }
}
