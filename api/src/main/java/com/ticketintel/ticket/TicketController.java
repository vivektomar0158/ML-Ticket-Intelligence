package com.ticketintel.ticket;

import com.ticketintel.common.ApiException;
import com.ticketintel.jobs.JobRepository;
import com.ticketintel.review.ReviewService;
import com.ticketintel.ticket.TicketQueryService.Filters;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import tools.jackson.databind.json.JsonMapper;

@RestController
public class TicketController {
    public record PatchRequest(String category, String priority, Boolean assignToMe, String status) {}
    public record RegenerateRequest(String instruction) {}

    private final TicketQueryService queries;
    private final ReviewService review;
    private final JdbcTemplate jdbc;
    private final JobRepository jobs;
    private final JsonMapper json;

    public TicketController(TicketQueryService queries, ReviewService review, JdbcTemplate jdbc, JobRepository jobs, JsonMapper json) {
        this.queries = queries;
        this.review = review;
        this.jdbc = jdbc;
        this.jobs = jobs;
        this.json = json;
    }

    @GetMapping("/api/tickets")
    public Map<String, Object> list(@RequestParam(required = false) String status, @RequestParam(required = false) String queue,
                                    @RequestParam(required = false) String category, @RequestParam(required = false) String priority,
                                    @RequestParam(required = false) Double minRisk, @RequestParam(required = false) String product,
                                    @RequestParam(required = false) Long clusterId, @RequestParam(required = false) Boolean incident,
                                    @RequestParam(required = false) String q, @RequestParam(required = false) Boolean open,
                                    @RequestParam(required = false) Boolean mine, @RequestParam(required = false) String sort,
                                    @RequestParam(required = false) String cursor, @RequestParam(defaultValue = "50") int limit,
                                    Authentication auth) {
        Long agent = Boolean.TRUE.equals(mine) ? review.agentId(auth.getName()) : null;
        return queries.list(new Filters(status, queue, category, priority, minRisk, product, clusterId, incident, q, agent, open),
                sort, cursor, limit);
    }

    @GetMapping("/api/tickets/counts")
    public Map<String, Object> counts() {
        return queries.counts();
    }

    @GetMapping("/api/tickets/{id}")
    public Map<String, Object> detail(@PathVariable long id) {
        return queries.detail(id);
    }

    /** Manual override of labels / assignment. Agent corrections win over model and LLM, and are logged for retraining. */
    @PatchMapping("/api/tickets/{id}")
    public Map<String, Object> patch(@PathVariable long id, @RequestBody PatchRequest req, Authentication auth) {
        return review.patch(id, req.category(), req.priority(), req.assignToMe(), req.status(), auth.getName());
    }

    @PostMapping("/api/tickets/{id}/draft:regenerate")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, Object> regenerate(@PathVariable long id, @RequestBody(required = false) RegenerateRequest req) {
        Integer exists = jdbc.queryForObject("SELECT count(*) FROM tickets WHERE id = ? AND status NOT IN ('NEW','RESOLVED','CLOSED')", Integer.class, id);
        if (exists == null || exists == 0) throw ApiException.conflict("ticket must be triaged and unresolved to regenerate a draft");
        String payload = req == null || req.instruction() == null ? null : json.writeValueAsString(Map.of("instruction", req.instruction().trim()));
        boolean queued = jobs.enqueue("DRAFT", id, 200, payload);       // high priority: an agent is waiting
        jdbc.update("UPDATE tickets SET draft_state = 'PENDING' WHERE id = ?", id);
        return Map.of("queued", queued);
    }

    @GetMapping("/api/clusters")
    public List<Map<String, Object>> clusters(@RequestParam(defaultValue = "ACTIVE") String status,
                                              @RequestParam(defaultValue = "false") boolean incidentsOnly) {
        return jdbc.queryForList("""
                SELECT c.id, c.title, c.product, c.size, c.is_incident, c.status, c.first_seen_at, c.last_seen_at, c.incident_flagged_at,
                       (SELECT count(*) FROM tickets t WHERE t.cluster_id = c.id AND t.status NOT IN ('RESOLVED','CLOSED')) AS open_tickets,
                       (SELECT count(DISTINCT customer_id) FROM tickets t WHERE t.cluster_id = c.id) AS customers
                FROM duplicate_clusters c WHERE c.status = ? AND (? = false OR c.is_incident)
                ORDER BY c.is_incident DESC, c.last_seen_at DESC LIMIT 100
                """, status, incidentsOnly);
    }

    @GetMapping("/api/clusters/{id}")
    public Map<String, Object> cluster(@PathVariable long id) {
        var rows = jdbc.queryForList("SELECT * FROM duplicate_clusters WHERE id = ?", id);
        if (rows.isEmpty()) throw ApiException.notFound("cluster");
        return Map.of("cluster", rows.get(0),
                "members", jdbc.queryForList("SELECT id, subject, status, customer_tier, category, priority, escalation_risk, created_at FROM tickets WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 200", id),
                "timeline", jdbc.queryForList("SELECT date_trunc('minute', created_at) AS minute, count(*) AS tickets FROM tickets WHERE cluster_id = ? GROUP BY 1 ORDER BY 1", id),
                "draft", jdbc.queryForList("""
                        SELECT d.id, d.ticket_id, d.status, d.body, d.grounding, d.created_at FROM drafts d JOIN tickets t ON t.id = d.ticket_id
                        WHERE t.cluster_id = ? AND d.status IN ('GENERATED','APPROVED','EDITED') AND d.reused_from_draft_id IS NULL
                        ORDER BY d.created_at DESC LIMIT 1
                        """, id));
    }
}
