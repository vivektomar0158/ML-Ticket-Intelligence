package com.ticketintel.triage;

import com.ticketintel.events.SseHub;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Service
public class SlaService {
    private static final Logger log = LoggerFactory.getLogger(SlaService.class);
    private final JdbcTemplate jdbc;
    private final SseHub events;
    private final Map<String, Integer> minutes = new ConcurrentHashMap<>();

    public SlaService(JdbcTemplate jdbc, SseHub events) {
        this.jdbc = jdbc;
        this.events = events;
    }

    public Instant dueAt(String tier, String priority, Instant createdAt) {
        if (minutes.isEmpty()) {
            jdbc.query("SELECT tier, priority, minutes FROM sla_policies",
                    rs -> { minutes.put(rs.getString(1) + "/" + rs.getString(2), rs.getInt(3)); });
        }
        int m = minutes.getOrDefault(tier + "/" + priority, 480);
        return createdAt.plus(Duration.ofMinutes(m));
    }

    /** Marks tickets whose first-response SLA has passed without a response. */
    @Scheduled(fixedDelay = 60_000, initialDelay = 30_000)
    void markBreaches() {
        List<Long> ids = jdbc.queryForList("""
                UPDATE tickets SET sla_breached = true
                WHERE NOT sla_breached AND first_response_at IS NULL AND status NOT IN ('RESOLVED','CLOSED')
                  AND sla_due_at IS NOT NULL AND sla_due_at < now() RETURNING id
                """, Long.class);
        for (Long id : ids) events.publish("sla.breached", Map.of("ticketId", id));
        if (!ids.isEmpty()) log.info("{} ticket(s) breached SLA", ids.size());
    }

    /** Clusters with no new tickets for 6h and no open tickets left are closed. */
    @Scheduled(fixedDelay = 300_000, initialDelay = 60_000)
    void closeStaleClusters() {
        jdbc.update("""
                UPDATE duplicate_clusters c SET status = 'CLOSED'
                WHERE c.status = 'ACTIVE' AND c.last_seen_at < now() - interval '6 hours'
                  AND NOT EXISTS (SELECT 1 FROM tickets t WHERE t.cluster_id = c.id AND t.status NOT IN ('RESOLVED','CLOSED'))
                """);
    }
}
