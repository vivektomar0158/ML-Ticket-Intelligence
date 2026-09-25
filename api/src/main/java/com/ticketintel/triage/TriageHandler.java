package com.ticketintel.triage;

import com.ticketintel.common.AppProperties;
import com.ticketintel.common.Util;
import com.ticketintel.dedup.DuplicateService;
import com.ticketintel.events.SseHub;
import com.ticketintel.jobs.JobHandler;
import com.ticketintel.jobs.JobRepository;
import com.ticketintel.jobs.JobRepository.Job;
import com.ticketintel.mlclient.MlClient;
import com.ticketintel.mlclient.MlDtos.*;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionTemplate;
import tools.jackson.databind.json.JsonMapper;

/**
 * TRIAGE job (fast path, batched): embeddings + category + priority -> duplicate/cluster/incident -> escalation risk
 * -> routing + SLA -> enqueue slow work (LLM_CLASSIFY for low-confidence, DRAFT for everything).
 * Idempotent: already-triaged tickets are skipped; embeddings/predictions use ON CONFLICT DO NOTHING.
 */
@Component
public class TriageHandler implements JobHandler {
    private static final Logger log = LoggerFactory.getLogger(TriageHandler.class);

    private record Row(long id, String subject, String body, String tier, String product, Instant createdAt, Long customerId,
                       String status, Job job) {}

    private record Phase1(DuplicateService.Result dedup, int priorTickets) {}

    private final JdbcTemplate jdbc;
    private final TransactionTemplate tx;
    private final MlClient ml;
    private final DuplicateService dedup;
    private final JobRepository jobs;
    private final SlaService sla;
    private final SseHub events;
    private final JsonMapper json;
    private final AppProperties props;

    public TriageHandler(JdbcTemplate jdbc, TransactionTemplate tx, MlClient ml, DuplicateService dedup, JobRepository jobs,
                         SlaService sla, SseHub events, JsonMapper json, AppProperties props) {
        this.jdbc = jdbc;
        this.tx = tx;
        this.ml = ml;
        this.dedup = dedup;
        this.jobs = jobs;
        this.sla = sla;
        this.events = events;
        this.json = json;
        this.props = props;
    }

    @Override public String type() { return "TRIAGE"; }
    @Override public int batchSize() { return props.workers().triageBatch(); }
    @Override public int threads() { return props.workers().triageThreads(); }

    @Override
    public void handle(List<Job> batch, Context ctx) {
        Map<Long, Job> byTicket = batch.stream().collect(Collectors.toMap(Job::ticketId, j -> j, (a, b) -> a));
        String ids = byTicket.keySet().stream().map(String::valueOf).collect(Collectors.joining(","));
        List<Row> todo = new ArrayList<>(jdbc.query("""
                SELECT id, subject, body, customer_tier, product, created_at, customer_id, status FROM tickets
                WHERE id IN (%s) AND status IN ('NEW','TRIAGING','TRIAGE_FAILED')
                """.formatted(ids), (rs, i) -> new Row(rs.getLong("id"), rs.getString("subject"), rs.getString("body"),
                rs.getString("customer_tier"), rs.getString("product"), Util.instant(rs, "created_at"),
                (Long) rs.getObject("customer_id"), rs.getString("status"), byTicket.get(rs.getLong("id")))));
        if (todo.isEmpty()) return;                                    // already triaged (redelivery): job completes
        todo.sort(Comparator.comparing(Row::createdAt).thenComparing(Row::id));   // bursts cluster deterministically

        long t0 = System.nanoTime();
        AnalyzeRes analyzed = ml.analyze(todo.stream().map(r -> new TicketIn(r.id(), r.subject(), r.body(), r.tier())).toList(), true);
        Map<Long, AnalyzeItem> ai = analyzed.items().stream().collect(Collectors.toMap(AnalyzeItem::ticketId, x -> x));
        long mlMs = (System.nanoTime() - t0) / 1_000_000;

        // ---- phase 1 (one short transaction per ticket): embedding + predictions + dedup/cluster
        Map<Long, Phase1> p1 = new LinkedHashMap<>();
        for (Row r : todo) {
            try {
                AnalyzeItem a = ai.get(r.id());
                p1.put(r.id(), tx.execute(s -> phase1(r, a, mlMs)));
            } catch (Exception e) {
                log.warn("triage phase1 failed for ticket {}: {}", r.id(), e.toString());
                ctx.fail(r.job(), e.toString(), true);
            }
        }
        List<Row> ok = todo.stream().filter(r -> p1.containsKey(r.id())).toList();
        if (ok.isEmpty()) return;

        // ---- phase 2: batched escalation scoring (context features: cluster size, customer history)
        List<EscIn> escIn = new ArrayList<>();
        for (Row r : ok) {
            var t = r.createdAt().atOffset(ZoneOffset.UTC);
            escIn.add(new EscIn(r.id(), r.subject(), r.body(), r.tier(), r.product(), t.getHour(), t.getDayOfWeek().getValue() - 1,
                    p1.get(r.id()).priorTickets(), p1.get(r.id()).dedup().matchCount(), ai.get(r.id()).embedding()));
        }
        EscalationRes esc = ml.escalation(escIn);
        Map<Long, EscItem> ei = esc.items().stream().collect(Collectors.toMap(EscItem::ticketId, x -> x));

        // ---- phase 3: route, SLA, enqueue slow work
        for (Row r : ok) {
            try {
                tx.executeWithoutResult(s -> phase3(r, ai.get(r.id()), ei.get(r.id()), p1.get(r.id())));
            } catch (Exception e) {
                log.warn("triage phase3 failed for ticket {}: {}", r.id(), e.toString());
                ctx.fail(r.job(), e.toString(), true);
            }
        }
    }

    private Phase1 phase1(Row r, AnalyzeItem a, long mlMs) {
        jdbc.update("""
                INSERT INTO ticket_embeddings (ticket_id, model_version, product, created_at, embedding)
                VALUES (?, ?, ?, ?, CAST(? AS vector)) ON CONFLICT (ticket_id) DO NOTHING
                """, r.id(), a.embeddingModel(), r.product(), Util.utc(r.createdAt()), Util.vec(a.embedding()));
        savePrediction(r.id(), "CATEGORY", a.category().model(), a.category().label(), a.category().confidence(),
                Map.of("probs", a.category().probs(), "needsLlm", a.needsLlm()), mlMs);
        savePrediction(r.id(), "PRIORITY", a.priority().model(), a.priority().label(), a.priority().confidence(),
                Map.of("probs", a.priority().probs(), "signals", a.priority().signals()), mlMs);
        var d = dedup.assign(r.id(), a.embedding(), r.product(), r.createdAt(), r.subject());
        Integer prior = r.customerId() == null ? 0 : jdbc.queryForObject("""
                SELECT count(*) FROM tickets WHERE customer_id = ? AND id <> ? AND created_at < ? AND created_at >= ?
                """, Integer.class, r.customerId(), r.id(), Util.utc(r.createdAt()), Util.utc(r.createdAt().minusSeconds(7 * 86400)));
        return new Phase1(d, prior == null ? 0 : prior);
    }

    private void phase3(Row r, AnalyzeItem a, EscItem e, Phase1 p) {
        savePrediction(r.id(), "ESCALATION", e.model(), e.route(), e.risk(),
                Map.of("rawScore", e.rawScore(), "topFactors", e.topFactors(), "similarRecent", p.dedup().matchCount(),
                        "priorTickets7d", p.priorTickets()), 0);
        String priority = a.priority().label();
        Instant due = sla.dueAt(r.tier(), priority, r.createdAt());
        int updated = jdbc.update("""
                UPDATE tickets SET status = 'TRIAGED', triaged_at = now(), category = ?, category_conf = ?, category_source = 'MODEL',
                  priority = ?, priority_conf = ?, priority_source = 'MODEL', escalation_risk = ?, queue = ?, cluster_id = COALESCE(?, cluster_id),
                  sla_due_at = ?, draft_state = 'PENDING', version = version + 1
                WHERE id = ? AND status IN ('NEW','TRIAGING','TRIAGE_FAILED')
                """, a.category().label(), (float) a.category().confidence(), priority, (float) a.priority().confidence(),
                (float) e.risk(), e.route(), p.dedup().clusterId(), Util.utc(due), r.id());
        if (updated == 0) return;                                    // raced with another worker: nothing more to do
        int tierBonus = switch (r.tier()) { case "ENTERPRISE" -> 20; case "PRO" -> 10; default -> 0; };
        int draftPrio = (int) Math.round(e.risk() * 100) + tierBonus + (p.dedup().incident() ? 30 : 0);
        jobs.enqueue("DRAFT", r.id(), draftPrio, null);
        if (a.needsLlm()) jobs.enqueue("LLM_CLASSIFY", r.id(), 0, null);   // cascade: cheap model unsure -> ask the LLM
        var ev = new HashMap<String, Object>();
        ev.put("ticketId", r.id());
        ev.put("category", a.category().label());
        ev.put("priority", priority);
        ev.put("risk", e.risk());
        ev.put("queue", e.route());
        ev.put("clusterId", p.dedup().clusterId());
        ev.put("incident", p.dedup().incident());
        events.publish("ticket.triaged", ev);
    }

    void savePrediction(long ticketId, String task, String model, String label, double score, Object details, long latencyMs) {
        String[] mv = model.contains("@") ? model.split("@", 2) : new String[] {model, "1"};
        jdbc.update("""
                INSERT INTO predictions (ticket_id, task, model_name, model_version, label, score, details, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, CAST(? AS jsonb), ?) ON CONFLICT (ticket_id, task, model_name, model_version) DO NOTHING
                """, ticketId, task, mv[0], mv[1], label, (float) score, json.writeValueAsString(details), (int) latencyMs);
    }
}
