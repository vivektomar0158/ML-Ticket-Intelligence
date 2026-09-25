package com.ticketintel.dedup;

import com.ticketintel.common.AppProperties;
import com.ticketintel.common.Util;
import com.ticketintel.events.SseHub;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * Duplicate detection + clustering + incident flagging (mirrors the offline replay in ml-service/training/eval_incidents.py).
 * Must run inside a transaction: dedup for one product is serialized with an advisory lock so two workers processing a
 * burst concurrently cannot split it into separate clusters.
 */
@Service
public class DuplicateService {

    public record Result(Long clusterId, int matchCount, boolean incident, boolean newlyIncident, int clusterSize) {}

    private record Match(long id, double sim, Long clusterId) {}

    private final JdbcTemplate jdbc;
    private final AppProperties.Dedup cfg;
    private final SseHub events;

    public DuplicateService(JdbcTemplate jdbc, AppProperties props, SseHub events) {
        this.jdbc = jdbc;
        this.cfg = props.dedup();
        this.events = events;
    }

    public Result assign(long ticketId, float[] embedding, String product, Instant createdAt, String subject) {
        jdbc.queryForObject("SELECT pg_advisory_xact_lock(hashtext(?))::text", String.class, "dedup:" + product);
        jdbc.execute("SET LOCAL hnsw.iterative_scan = relaxed_order");
        String q = Util.vec(embedding);
        Instant since = createdAt.minus(Duration.ofHours(cfg.windowHours()));
        List<Match> matches = jdbc.query("""
                SELECT e.ticket_id, 1 - (e.embedding <=> CAST(? AS vector)) AS sim, t.cluster_id
                FROM ticket_embeddings e JOIN tickets t ON t.id = e.ticket_id
                WHERE e.is_open AND e.product = ? AND e.created_at >= ? AND e.created_at <= ? AND e.ticket_id <> ?
                ORDER BY e.embedding <=> CAST(? AS vector) LIMIT 20
                """, (rs, i) -> new Match(rs.getLong(1), rs.getDouble(2), (Long) rs.getObject(3)),
                q, product, Util.utc(since), Util.utc(createdAt), ticketId, q).stream()
                .filter(m -> m.sim() >= cfg.tau()).toList();
        if (matches.isEmpty()) return new Result(null, 0, false, false, 1);

        Set<Long> clusterIds = matches.stream().map(Match::clusterId).filter(java.util.Objects::nonNull).collect(Collectors.toCollection(LinkedHashSet::new));
        Long target;
        if (clusterIds.isEmpty()) {
            Instant first = createdAt;
            target = jdbc.queryForObject("""
                    INSERT INTO duplicate_clusters (product, title, first_seen_at, last_seen_at)
                    VALUES (?, ?, LEAST(?, (SELECT min(created_at) FROM tickets WHERE id = ANY(CAST(? AS bigint[])))), ?) RETURNING id
                    """, Long.class, product, Util.trunc(subject, 200), Util.utc(first),
                    "{" + matches.stream().map(m -> String.valueOf(m.id())).collect(Collectors.joining(",")) + "}", Util.utc(createdAt));
        } else {
            List<Long> sorted = new ArrayList<>(clusterIds);
            sorted.sort(Long::compare);
            String inList = sorted.stream().map(String::valueOf).collect(Collectors.joining(","));
            jdbc.queryForList("SELECT id FROM duplicate_clusters WHERE id IN (" + inList + ") ORDER BY id FOR UPDATE", Long.class);
            target = sorted.get(0);
            if (sorted.size() > 1) {                                   // several clusters bridged by this ticket -> merge into the oldest
                String others = sorted.subList(1, sorted.size()).stream().map(String::valueOf).collect(Collectors.joining(","));
                jdbc.update("UPDATE tickets SET cluster_id = ? WHERE cluster_id IN (" + others + ")", target);
                jdbc.update("""
                        UPDATE duplicate_clusters SET is_incident = is_incident OR (SELECT COALESCE(bool_or(is_incident), false) FROM duplicate_clusters WHERE id IN (%s)),
                          incident_flagged_at = LEAST(incident_flagged_at, (SELECT min(incident_flagged_at) FROM duplicate_clusters WHERE id IN (%s))) WHERE id = ?
                        """.formatted(others, others), target);
                jdbc.update("UPDATE duplicate_clusters SET status = 'MERGED', merged_into = ?, size = 0 WHERE id IN (" + others + ")", target);
            }
        }
        List<Long> members = new ArrayList<>(matches.stream().map(Match::id).toList());
        members.add(ticketId);
        jdbc.update("UPDATE tickets SET cluster_id = ? WHERE id IN (" + members.stream().map(String::valueOf).collect(Collectors.joining(",")) + ")", target);
        for (Match m : matches) {
            jdbc.update("INSERT INTO duplicate_links (ticket_id, similar_id, similarity) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                    ticketId, m.id(), (float) m.sim());
        }
        jdbc.update("""
                UPDATE duplicate_clusters c SET
                  size = (SELECT count(*) FROM tickets WHERE cluster_id = c.id),
                  last_seen_at = GREATEST(c.last_seen_at, ?), first_seen_at = LEAST(c.first_seen_at, ?),
                  canonical_ticket_id = (SELECT id FROM tickets WHERE cluster_id = c.id ORDER BY created_at, id LIMIT 1),
                  title = (SELECT subject FROM tickets WHERE cluster_id = c.id ORDER BY created_at, id LIMIT 1)
                WHERE c.id = ?
                """, Util.utc(createdAt), Util.utc(createdAt), target);

        // incident = enough recent reports from enough distinct customers (a burst, not one noisy user)
        Map<String, Object> live = jdbc.queryForMap("""
                SELECT count(*) AS n, count(DISTINCT COALESCE(customer_id::text, id::text)) AS customers
                FROM tickets WHERE cluster_id = ? AND created_at BETWEEN ? AND ?
                """, target, Util.utc(createdAt.minus(Duration.ofMinutes(cfg.burstMinutes()))), Util.utc(createdAt));
        long n = ((Number) live.get("n")).longValue(), customers = ((Number) live.get("customers")).longValue();
        Map<String, Object> c = jdbc.queryForMap("SELECT is_incident, size FROM duplicate_clusters WHERE id = ?", target);
        boolean already = (Boolean) c.get("is_incident");
        int size = ((Number) c.get("size")).intValue();
        boolean newly = false;
        if (!already && n >= cfg.incidentMinTickets() && customers >= cfg.incidentMinCustomers()) {
            jdbc.update("UPDATE duplicate_clusters SET is_incident = true, incident_flagged_at = now() WHERE id = ?", target);
            events.publish("incident.detected", Map.of("clusterId", target, "size", size, "product", product));
            newly = true;
        }
        events.publish("cluster.updated", Map.of("clusterId", target, "size", size));
        return new Result(target, matches.size(), already || newly, newly, size);
    }
}
