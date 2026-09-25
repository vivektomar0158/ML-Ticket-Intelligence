package com.ticketintel.jobs;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import static com.ticketintel.common.Util.instant;

/**
 * Postgres-backed queue. Enqueue happens in the caller's transaction (transactional outbox: a ticket and its job are
 * committed together or not at all). Claiming uses FOR UPDATE SKIP LOCKED so concurrent workers never double-process.
 */
@Repository
public class JobRepository {

    public record Job(long id, String type, long ticketId, int attempts, int maxAttempts, int priority, String payload,
                      Instant createdAt) {}

    private final JdbcTemplate jdbc;

    public JobRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** Idempotent: at most one PENDING/RUNNING job per (ticket, type). Returns true if a new job was created. */
    public boolean enqueue(String type, long ticketId, int priority, String payloadJson) {
        return jdbc.update("""
                INSERT INTO jobs (type, ticket_id, priority, payload) VALUES (?, ?, ?, CAST(? AS jsonb))
                ON CONFLICT (ticket_id, type) WHERE status IN ('PENDING','RUNNING') DO NOTHING
                """, type, ticketId, priority, payloadJson) > 0;
    }

    public List<Job> claim(String type, int batch, String worker) {
        return jdbc.query("""
                UPDATE jobs SET status = 'RUNNING', locked_by = ?, locked_at = now(), attempts = attempts + 1, updated_at = now()
                WHERE id IN (SELECT id FROM jobs WHERE status = 'PENDING' AND type = ? AND run_after <= now()
                             ORDER BY priority DESC, id LIMIT ? FOR UPDATE SKIP LOCKED)
                RETURNING id, type, ticket_id, attempts, max_attempts, priority, payload::text AS payload, created_at
                """, (rs, i) -> new Job(rs.getLong("id"), rs.getString("type"), rs.getLong("ticket_id"), rs.getInt("attempts"),
                rs.getInt("max_attempts"), rs.getInt("priority"), rs.getString("payload"), instant(rs, "created_at")),
                worker, type, batch);
    }

    public void complete(long id) {
        jdbc.update("UPDATE jobs SET status = 'DONE', locked_by = NULL, updated_at = now() WHERE id = ?", id);
    }

    /** Retry with exponential backoff + jitter, or DEAD when out of attempts / not retryable. */
    public void fail(Job job, String error, boolean retryable) {
        String err = error == null ? null : error.substring(0, Math.min(error.length(), 1000));
        if (!retryable || job.attempts() >= job.maxAttempts()) {
            jdbc.update("UPDATE jobs SET status = 'DEAD', last_error = ?, locked_by = NULL, updated_at = now() WHERE id = ?", err, job.id());
            // make permanent failures visible on the ticket (agents then handle it manually)
            if (job.type().equals("TRIAGE")) {
                jdbc.update("UPDATE tickets SET status = 'TRIAGE_FAILED' WHERE id = ? AND status IN ('NEW','TRIAGING')", job.ticketId());
            } else if (job.type().equals("DRAFT")) {
                jdbc.update("UPDATE tickets SET draft_state = 'FAILED' WHERE id = ? AND draft_state IN ('PENDING','WAITING_LLM')", job.ticketId());
            }
        } else {
            jdbc.update("UPDATE jobs SET status = 'PENDING', last_error = ?, locked_by = NULL, run_after = now() + make_interval(secs => ?), updated_at = now() WHERE id = ?",
                    err, backoffSeconds(job.attempts()), job.id());
        }
    }

    /** Dependency outage: try again later WITHOUT consuming an attempt. */
    public void defer(long id, int seconds, String reason) {
        jdbc.update("UPDATE jobs SET status = 'PENDING', attempts = GREATEST(attempts - 1, 0), last_error = ?, locked_by = NULL, run_after = now() + make_interval(secs => ?), updated_at = now() WHERE id = ?",
                reason, seconds, id);
    }

    /** Visibility timeout: jobs whose worker died mid-flight go back to PENDING. */
    public int reap(int timeoutMinutes) {
        return jdbc.update("UPDATE jobs SET status = 'PENDING', locked_by = NULL, updated_at = now() WHERE status = 'RUNNING' AND locked_at < now() - make_interval(mins => ?)",
                timeoutMinutes);
    }

    public int cleanup(int keepDays) {
        return jdbc.update("DELETE FROM jobs WHERE status = 'DONE' AND updated_at < now() - make_interval(days => ?)", keepDays);
    }

    public int pendingDepth(String type) {
        Integer n = jdbc.queryForObject("SELECT count(*) FROM jobs WHERE status = 'PENDING' AND type = ?", Integer.class, type);
        return n == null ? 0 : n;
    }

    public Map<String, Object> depthByStatus() {
        var m = new java.util.LinkedHashMap<String, Object>();
        jdbc.query("SELECT type, status, count(*) c FROM jobs WHERE status <> 'DONE' GROUP BY type, status ORDER BY type, status",
                rs -> { m.put(rs.getString("type") + "." + rs.getString("status"), rs.getLong("c")); });
        return m;
    }

    public List<Map<String, Object>> dead(int limit) {
        return jdbc.queryForList("SELECT id, type, ticket_id, attempts, last_error, updated_at FROM jobs WHERE status = 'DEAD' ORDER BY updated_at DESC LIMIT ?", limit);
    }

    public boolean retryDead(long id) {
        return jdbc.update("UPDATE jobs SET status = 'PENDING', attempts = 0, run_after = now(), updated_at = now() WHERE id = ? AND status = 'DEAD'", id) > 0;
    }

    static int backoffSeconds(int attempts) {
        int base = (int) Math.min(300, 2L << Math.min(attempts, 8));     // 4s, 8s, 16s, ... capped at 5 min
        return base + ThreadLocalRandom.current().nextInt(0, Math.max(1, base / 4));
    }
}
