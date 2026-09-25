package com.ticketintel.ticket;

import com.ticketintel.common.ApiException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import tools.jackson.databind.json.JsonMapper;

@Service
public class TicketQueryService {

    public record Filters(String status, String queue, String category, String priority, Double minRisk, String product,
                          Long clusterId, Boolean incident, String q, Long assignedAgentId, Boolean open) {}

    private final JdbcTemplate jdbc;
    private final JsonMapper json;

    public TicketQueryService(JdbcTemplate jdbc, JsonMapper json) {
        this.jdbc = jdbc;
        this.json = json;
    }

    /** Keyset pagination (no OFFSET): stable and O(log n) per page even with millions of tickets. */
    public Map<String, Object> list(Filters f, String sort, String cursor, int limit) {
        limit = Math.max(1, Math.min(limit, 100));
        String expr, cast;
        boolean desc;
        switch (sort == null ? "risk" : sort) {
            case "newest" -> { expr = "t.created_at"; cast = "timestamptz"; desc = true; }
            case "sla" -> { expr = "COALESCE(t.sla_due_at, 'infinity'::timestamptz)"; cast = "timestamptz"; desc = false; }
            case "risk" -> { expr = "CAST(COALESCE(t.escalation_risk, -1) AS double precision)"; cast = "double precision"; desc = true; }
            default -> throw ApiException.badRequest("sort must be risk|newest|sla");
        }
        StringBuilder sql = new StringBuilder("""
                SELECT t.id, t.external_id, t.subject, t.customer_tier, t.product, t.status, t.draft_state, t.category, t.category_conf,
                       t.category_source, t.priority, t.escalation_risk, t.queue, t.cluster_id, c.size AS cluster_size,
                       COALESCE(c.is_incident, false) AS is_incident, t.sla_due_at, t.sla_breached, t.created_at, t.assigned_agent_id,
                """ + expr + """
                 ::text AS sort_val
                FROM tickets t LEFT JOIN duplicate_clusters c ON c.id = t.cluster_id WHERE 1=1
                """);
        List<Object> args = new ArrayList<>();
        if (f.status() != null && !f.status().isBlank()) {
            String[] st = f.status().split(",");
            sql.append(" AND t.status IN (").append("?,".repeat(st.length), 0, st.length * 2 - 1).append(")");
            args.addAll(Arrays.asList(st));
        }
        if (Boolean.TRUE.equals(f.open())) sql.append(" AND t.status NOT IN ('RESOLVED','CLOSED')");
        if (f.queue() != null) { sql.append(" AND t.queue = ?"); args.add(f.queue()); }
        if (f.category() != null) { sql.append(" AND t.category = ?"); args.add(f.category()); }
        if (f.priority() != null) { sql.append(" AND t.priority = ?"); args.add(f.priority()); }
        if (f.minRisk() != null) { sql.append(" AND t.escalation_risk >= ?"); args.add(f.minRisk().floatValue()); }
        if (f.product() != null) { sql.append(" AND t.product = ?"); args.add(f.product()); }
        if (f.clusterId() != null) { sql.append(" AND t.cluster_id = ?"); args.add(f.clusterId()); }
        if (Boolean.TRUE.equals(f.incident())) sql.append(" AND c.is_incident");
        if (f.assignedAgentId() != null) { sql.append(" AND t.assigned_agent_id = ?"); args.add(f.assignedAgentId()); }
        if (f.q() != null && !f.q().isBlank()) {
            sql.append(" AND (t.search_tsv @@ websearch_to_tsquery('english', ?) OR t.subject ILIKE ?)");
            args.add(f.q());
            args.add("%" + f.q().replace("%", "") + "%");
        }
        if (cursor != null && !cursor.isBlank()) {
            String[] c = new String(Base64.getUrlDecoder().decode(cursor), StandardCharsets.UTF_8).split("\\|", 2);
            sql.append(" AND (").append(expr).append(", t.id) ").append(desc ? "<" : ">").append(" (CAST(? AS ").append(cast).append("), ?)");
            args.add(c[0]);
            args.add(Long.parseLong(c[1]));
        }
        sql.append(" ORDER BY ").append(expr).append(desc ? " DESC, t.id DESC" : " ASC, t.id ASC").append(" LIMIT ?");
        args.add(limit + 1);

        List<Map<String, Object>> rows = jdbc.queryForList(sql.toString(), args.toArray());
        String next = null;
        if (rows.size() > limit) {
            rows = new ArrayList<>(rows.subList(0, limit));
            var last = rows.get(rows.size() - 1);
            next = Base64.getUrlEncoder().withoutPadding().encodeToString((last.get("sort_val") + "|" + last.get("id")).getBytes(StandardCharsets.UTF_8));
        }
        rows.forEach(r -> r.remove("sort_val"));
        var out = new LinkedHashMap<String, Object>();
        out.put("items", rows);
        out.put("nextCursor", next);
        return out;
    }

    public Map<String, Object> counts() {
        var m = new LinkedHashMap<String, Object>();
        jdbc.query("""
                SELECT queue, count(*) FILTER (WHERE status NOT IN ('RESOLVED','CLOSED')) AS open,
                       count(*) FILTER (WHERE status = 'DRAFTED') AS awaiting_review,
                       count(*) FILTER (WHERE sla_breached AND status NOT IN ('RESOLVED','CLOSED')) AS breached
                FROM tickets GROUP BY queue
                """, rs -> {
            m.put(rs.getString("queue"), Map.of("open", rs.getLong("open"), "awaitingReview", rs.getLong("awaiting_review"),
                    "breached", rs.getLong("breached")));
        });
        Long incidents = jdbc.queryForObject("SELECT count(*) FROM duplicate_clusters WHERE is_incident AND status = 'ACTIVE'", Long.class);
        m.put("activeIncidents", incidents);
        return m;
    }

    /** Everything the agent needs on one screen. */
    public Map<String, Object> detail(long id) {
        List<Map<String, Object>> rows = jdbc.queryForList("""
                SELECT t.*, t.search_tsv IS NULL AS _x, cu.external_id AS customer_external_id, a.display_name AS assigned_agent
                FROM tickets t LEFT JOIN customers cu ON cu.id = t.customer_id LEFT JOIN agents a ON a.id = t.assigned_agent_id
                WHERE t.id = ?
                """, id);
        if (rows.isEmpty()) throw ApiException.notFound("ticket");
        var t = new LinkedHashMap<>(rows.get(0));
        t.remove("search_tsv");
        t.remove("_x");
        t.remove("content_hash");

        var out = new LinkedHashMap<String, Object>();
        out.put("ticket", t);
        // latest prediction per task + per-model history (model-vs-LLM disagreement is visible to the agent)
        List<Map<String, Object>> preds = jdbc.queryForList("""
                SELECT task, model_name, model_version, label, score, details::text AS details, latency_ms, created_at
                FROM predictions WHERE ticket_id = ? ORDER BY created_at DESC
                """, id);
        preds.forEach(p -> p.put("details", p.get("details") == null ? null : json.readTree((String) p.get("details"))));
        out.put("predictions", preds);

        Long clusterId = (Long) t.get("cluster_id");
        if (clusterId != null) {
            out.put("cluster", jdbc.queryForMap("SELECT id, title, size, is_incident, status, first_seen_at, last_seen_at, canonical_ticket_id FROM duplicate_clusters WHERE id = ?", clusterId));
            out.put("clusterMembers", jdbc.queryForList("""
                    SELECT id, subject, status, created_at, customer_tier FROM tickets WHERE cluster_id = ? AND id <> ? ORDER BY created_at DESC LIMIT 25
                    """, clusterId, id));
        }
        out.put("duplicates", jdbc.queryForList("""
                SELECT l.similar_id AS id, l.similarity, s.subject, s.status, s.created_at FROM duplicate_links l JOIN tickets s ON s.id = l.similar_id
                WHERE l.ticket_id = ? ORDER BY l.similarity DESC LIMIT 20
                """, id));

        List<Map<String, Object>> drafts = jdbc.queryForList("""
                SELECT id, status, body, citations::text AS citations, grounding, llm_confidence, needs_info::text AS needs_info,
                       sources::text AS sources, reused_from_draft_id, instruction, model, prompt_version, cost_usd, latency_ms, error, created_at
                FROM drafts WHERE ticket_id = ? AND status <> 'SUPERSEDED' ORDER BY created_at DESC LIMIT 5
                """, id);
        for (var d : drafts) {
            for (String k : List.of("citations", "needs_info", "sources")) d.put(k, d.get(k) == null ? null : json.readTree((String) d.get(k)));
        }
        out.put("drafts", drafts);
        out.put("draft", drafts.stream().filter(d -> !"SKIPPED".equals(d.get("status"))).findFirst().orElse(null));

        // the resolved tickets the draft was grounded on (with resolutions) so the agent can verify citations
        var sourceIds = new ArrayList<Long>();
        if (!drafts.isEmpty() && drafts.get(0).get("sources") instanceof tools.jackson.databind.JsonNode sn) {
            sn.forEach(n -> sourceIds.add(n.path("id").asLong()));
        }
        if (!sourceIds.isEmpty()) {
            String ids = String.join(",", sourceIds.stream().map(String::valueOf).toList());
            out.put("sourceTickets", jdbc.queryForList("SELECT id, subject, body, resolution, category, resolved_at FROM tickets WHERE id IN (" + ids + ")"));
        } else {
            out.put("sourceTickets", List.of());
        }
        out.put("feedback", jdbc.queryForList("SELECT id, action, edit_ratio, reject_reason, corrected_category, corrected_priority, rating, created_at FROM feedback WHERE ticket_id = ? ORDER BY created_at DESC", id));
        return out;
    }
}
