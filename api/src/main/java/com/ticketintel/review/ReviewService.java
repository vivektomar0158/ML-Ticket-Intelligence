package com.ticketintel.review;

import com.ticketintel.common.ApiException;
import com.ticketintel.common.Util;
import com.ticketintel.events.SseHub;
import com.ticketintel.triage.SlaService;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Human-in-the-loop: every agent decision is stored with edit distance, reasons and label corrections. */
@Service
public class ReviewService {
    public static final Set<String> REJECT_REASONS = Set.of("WRONG_INFO", "NOT_GROUNDED", "TONE", "INCOMPLETE", "OTHER");

    public record ReviewRequest(String action, String finalBody, String rejectReason, String correctedCategory,
                                String correctedPriority, Integer rating, Long timeToReviewMs) {}

    private final JdbcTemplate jdbc;
    private final SseHub events;
    private final SlaService sla;

    public ReviewService(JdbcTemplate jdbc, SseHub events, SlaService sla) {
        this.jdbc = jdbc;
        this.events = events;
        this.sla = sla;
    }

    public Long agentId(String username) {
        List<Long> ids = jdbc.queryForList("SELECT id FROM agents WHERE username = ?", Long.class, username);
        return ids.isEmpty() ? null : ids.get(0);
    }

    @Transactional
    public Map<String, Object> review(long draftId, ReviewRequest req, String username) {
        var rows = jdbc.queryForList("SELECT ticket_id, status, body FROM drafts WHERE id = ? FOR UPDATE", draftId);
        if (rows.isEmpty()) throw ApiException.notFound("draft");
        var d = rows.get(0);
        if (!"GENERATED".equals(d.get("status"))) throw ApiException.conflict("draft already reviewed or superseded (status " + d.get("status") + ")");
        long ticketId = ((Number) d.get("ticket_id")).longValue();
        String draftBody = (String) d.get("body");
        String action = req.action() == null ? "" : req.action();
        Long agent = agentId(username);
        validateLabels(req);
        applyLabelCorrections(ticketId, req.correctedCategory(), req.correctedPriority());
        Integer rating = req.rating();
        if (rating != null && (rating < 1 || rating > 5)) throw ApiException.badRequest("rating must be 1..5");

        switch (action) {
            case "APPROVE" -> {
                finish(ticketId, draftBody, agent);
                jdbc.update("UPDATE drafts SET status = 'APPROVED' WHERE id = ?", draftId);
                feedback(ticketId, draftId, agent, "APPROVE", draftBody, 0, 0.0, null, req);
            }
            case "EDIT_APPROVE" -> {
                String fb = req.finalBody();
                if (fb == null || fb.trim().length() < 10) throw ApiException.badRequest("finalBody is required (min 10 chars)");
                fb = fb.trim();
                int dist = EditDistance.levenshtein(draftBody, fb);
                finish(ticketId, fb, agent);
                jdbc.update("UPDATE drafts SET status = 'EDITED' WHERE id = ?", draftId);
                feedback(ticketId, draftId, agent, "EDIT_APPROVE", fb, dist, EditDistance.ratio(draftBody, fb), null, req);
            }
            case "REJECT" -> {
                if (req.rejectReason() == null || !REJECT_REASONS.contains(req.rejectReason()))
                    throw ApiException.badRequest("rejectReason must be one of " + REJECT_REASONS);
                jdbc.update("UPDATE drafts SET status = 'REJECTED' WHERE id = ?", draftId);
                jdbc.update("UPDATE tickets SET status = 'IN_REVIEW', assigned_agent_id = COALESCE(assigned_agent_id, ?), version = version + 1 WHERE id = ? AND status <> 'RESOLVED'", agent, ticketId);
                feedback(ticketId, draftId, agent, "REJECT", null, null, null, req.rejectReason(), req);
                events.publish("ticket.updated", Map.of("ticketId", ticketId, "status", "IN_REVIEW"));
            }
            default -> throw ApiException.badRequest("action must be APPROVE | EDIT_APPROVE | REJECT");
        }
        return Map.of("ticketId", ticketId, "action", action);
    }

    /** After rejecting a draft (or with no draft at all) the agent writes the reply themselves. */
    @Transactional
    public Map<String, Object> resolveManually(long ticketId, String body, String category, String priority, Integer rating, String username) {
        if (body == null || body.trim().length() < 10) throw ApiException.badRequest("body is required (min 10 chars)");
        Integer n = jdbc.queryForObject("SELECT count(*) FROM tickets WHERE id = ? AND status NOT IN ('RESOLVED','CLOSED')", Integer.class, ticketId);
        if (n == null || n == 0) throw ApiException.conflict("ticket not found or already resolved");
        var req = new ReviewRequest("EDIT_APPROVE", body, null, category, priority, rating, null);
        validateLabels(req);
        applyLabelCorrections(ticketId, category, priority);
        Long agent = agentId(username);
        finish(ticketId, body.trim(), agent);
        feedback(ticketId, null, agent, "EDIT_APPROVE", body.trim(), body.trim().length(), 1.0, null, req);   // fully manual = ratio 1.0
        return Map.of("ticketId", ticketId, "action", "MANUAL_RESOLVE");
    }

    @Transactional
    public Map<String, Object> patch(long id, String category, String priority, Boolean assignToMe, String status, String username) {
        Integer n = jdbc.queryForObject("SELECT count(*) FROM tickets WHERE id = ?", Integer.class, id);
        if (n == null || n == 0) throw ApiException.notFound("ticket");
        Long agent = agentId(username);
        var req = new ReviewRequest("RELABEL", null, null, category, priority, null, null);
        validateLabels(req);
        if (category != null || priority != null) {
            applyLabelCorrections(id, category, priority);
            feedback(id, null, agent, "RELABEL", null, null, null, null, req);
        }
        if (Boolean.TRUE.equals(assignToMe) && agent != null) jdbc.update("UPDATE tickets SET assigned_agent_id = ? WHERE id = ?", agent, id);
        if (status != null) {
            if (!Set.of("IN_REVIEW", "CLOSED").contains(status)) throw ApiException.badRequest("status may only be set to IN_REVIEW or CLOSED here");
            jdbc.update("UPDATE tickets SET status = ? WHERE id = ?", status, id);
        }
        jdbc.update("UPDATE tickets SET version = version + 1 WHERE id = ?", id);
        events.publish("ticket.updated", Map.of("ticketId", id));
        return Map.of("ticketId", id);
    }

    // ------------------------------------------------------------------- internals
    private void validateLabels(ReviewRequest r) {
        if (r.correctedCategory() != null && !Util.CATEGORIES.contains(r.correctedCategory())) throw ApiException.badRequest("unknown category");
        if (r.correctedPriority() != null && !Util.PRIORITIES.contains(r.correctedPriority())) throw ApiException.badRequest("unknown priority");
    }

    private void applyLabelCorrections(long ticketId, String category, String priority) {
        if (category != null) {
            jdbc.update("UPDATE tickets SET category = ?, category_source = 'AGENT', category_conf = 1 WHERE id = ? AND category IS DISTINCT FROM ?",
                    category, ticketId, category);
        }
        if (priority != null) {
            var t = jdbc.queryForMap("SELECT customer_tier, created_at, priority FROM tickets WHERE id = ?", ticketId);
            if (!priority.equals(t.get("priority"))) {
                Instant created = ((java.time.OffsetDateTime) t.get("created_at")).toInstant();
                jdbc.update("UPDATE tickets SET priority = ?, priority_source = 'AGENT', priority_conf = 1, sla_due_at = ? WHERE id = ?",
                        priority, Util.utc(sla.dueAt((String) t.get("customer_tier"), priority, created)), ticketId);
            }
        }
    }

    /** Resolution = the reply the customer got. It becomes retrievable RAG evidence immediately (closing the loop). */
    private void finish(long ticketId, String finalBody, Long agent) {
        jdbc.update("""
                UPDATE tickets SET status = 'RESOLVED', resolution = ?, resolved_at = now(), first_response_at = COALESCE(first_response_at, now()),
                  assigned_agent_id = COALESCE(assigned_agent_id, ?), draft_state = 'READY', version = version + 1 WHERE id = ?
                """, finalBody, agent, ticketId);
        jdbc.update("UPDATE ticket_embeddings SET is_resolved = true, is_open = false WHERE ticket_id = ?", ticketId);
        events.publish("ticket.resolved", Map.of("ticketId", ticketId));
    }

    private void feedback(long ticketId, Long draftId, Long agent, String action, String finalBody, Integer dist, Double ratio,
                          String rejectReason, ReviewRequest r) {
        jdbc.update("""
                INSERT INTO feedback (ticket_id, draft_id, agent_id, action, final_body, edit_distance, edit_ratio, reject_reason,
                                      corrected_category, corrected_priority, rating, time_to_review_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ticketId, draftId, agent, action, finalBody, dist, ratio == null ? null : ratio.floatValue(), rejectReason,
                r.correctedCategory(), r.correctedPriority(), r.rating(), r.timeToReviewMs());
    }
}
