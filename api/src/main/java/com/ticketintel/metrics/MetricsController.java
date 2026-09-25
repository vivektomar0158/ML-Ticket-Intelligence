package com.ticketintel.metrics;

import com.ticketintel.jobs.JobRepository;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/** Aggregates behind the dashboard's Metrics page: volume, quality of drafts, cost, latency, queue health. */
@RestController
public class MetricsController {
    private final JdbcTemplate jdbc;
    private final JobRepository jobs;

    public MetricsController(JdbcTemplate jdbc, JobRepository jobs) {
        this.jdbc = jdbc;
        this.jobs = jobs;
    }

    @GetMapping("/api/metrics/overview")
    public Map<String, Object> overview(@RequestParam(defaultValue = "7") int days) {
        days = Math.max(1, Math.min(days, 365));
        java.time.OffsetDateTime since = Instant.now().minus(days, ChronoUnit.DAYS).atOffset(java.time.ZoneOffset.UTC);
        var out = new LinkedHashMap<String, Object>();

        out.put("totals", jdbc.queryForMap("""
                SELECT count(*) AS tickets, count(*) FILTER (WHERE status IN ('RESOLVED','CLOSED')) AS resolved,
                       count(*) FILTER (WHERE queue = 'SENIOR') AS senior_queue, count(*) FILTER (WHERE sla_breached) AS sla_breached,
                       COALESCE(avg(escalation_risk), 0) AS avg_risk
                FROM tickets WHERE source <> 'SEED' AND ingested_at >= ?
                """, since));
        out.put("volumeByDay", jdbc.queryForList("SELECT to_char(date_trunc('day', ingested_at), 'YYYY-MM-DD') AS day, count(*) AS tickets FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? GROUP BY 1 ORDER BY 1", since));
        out.put("categoryMix", jdbc.queryForList("SELECT COALESCE(category, 'UNCLASSIFIED') AS category, count(*) AS tickets FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? GROUP BY 1 ORDER BY 2 DESC", since));
        out.put("priorityMix", jdbc.queryForList("SELECT COALESCE(priority, 'UNSET') AS priority, count(*) AS tickets FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? GROUP BY 1 ORDER BY 2 DESC", since));
        out.put("riskHistogram", jdbc.queryForList("""
                SELECT width_bucket(escalation_risk, 0, 1, 10) AS bucket, count(*) AS tickets FROM tickets
                WHERE ingested_at >= ? AND escalation_risk IS NOT NULL GROUP BY 1 ORDER BY 1
                """, since));

        // ---- draft quality (the feedback loop's headline numbers)
        var q = jdbc.queryForMap("""
                SELECT count(*) FILTER (WHERE action = 'APPROVE') AS approved, count(*) FILTER (WHERE action = 'EDIT_APPROVE' AND draft_id IS NOT NULL) AS edited,
                       count(*) FILTER (WHERE action = 'REJECT') AS rejected, count(*) FILTER (WHERE action = 'EDIT_APPROVE' AND draft_id IS NULL) AS manual,
                       COALESCE(avg(edit_ratio) FILTER (WHERE action = 'EDIT_APPROVE' AND draft_id IS NOT NULL), 0) AS mean_edit_ratio,
                       COALESCE(avg(time_to_review_ms), 0) AS avg_review_ms, COALESCE(avg(rating), 0) AS avg_rating
                FROM feedback WHERE created_at >= ?
                """, since);
        long approved = ((Number) q.get("approved")).longValue(), edited = ((Number) q.get("edited")).longValue(), rejected = ((Number) q.get("rejected")).longValue();
        long reviewed = approved + edited + rejected;
        var drafts = new LinkedHashMap<String, Object>(q);
        drafts.put("reviewed", reviewed);
        drafts.put("approvalRate", reviewed == 0 ? null : (double) (approved + edited) / reviewed);       // approved as-is or after edits
        drafts.put("approvedAsIsRate", reviewed == 0 ? null : (double) approved / reviewed);
        drafts.put("rejectReasons", jdbc.queryForList("SELECT reject_reason AS reason, count(*) AS n FROM feedback WHERE action = 'REJECT' AND created_at >= ? GROUP BY 1 ORDER BY 2 DESC", since));
        drafts.put("byGrounding", jdbc.queryForList("""
                SELECT d.grounding, count(*) AS reviewed,
                       round((count(*) FILTER (WHERE f.action IN ('APPROVE','EDIT_APPROVE')))::numeric / count(*), 3) AS approval_rate,
                       round(COALESCE(avg(f.edit_ratio), 0)::numeric, 3) AS mean_edit_ratio
                FROM feedback f JOIN drafts d ON d.id = f.draft_id WHERE f.created_at >= ? GROUP BY 1 ORDER BY 1
                """, since));
        drafts.put("statusCounts", jdbc.queryForList("SELECT status, count(*) AS n FROM drafts WHERE created_at >= ? GROUP BY 1 ORDER BY 2 DESC", since));
        out.put("drafts", drafts);

        // ---- LLM usage / cost
        var llm = new LinkedHashMap<String, Object>(jdbc.queryForMap("""
                SELECT count(*) FILTER (WHERE reused_from_draft_id IS NULL AND model IS NOT NULL AND model <> 'incident-reuse') AS draft_calls,
                       count(*) FILTER (WHERE reused_from_draft_id IS NOT NULL) AS reused_drafts,
                       COALESCE(sum(cost_usd), 0) AS draft_cost_usd, COALESCE(avg(latency_ms) FILTER (WHERE latency_ms IS NOT NULL), 0) AS avg_draft_latency_ms,
                       COALESCE(sum(input_tokens), 0) AS input_tokens, COALESCE(sum(output_tokens), 0) AS output_tokens
                FROM drafts WHERE created_at >= ?
                """, since));
        llm.put("classifyCalls", jdbc.queryForObject("SELECT count(*) FROM predictions WHERE model_name = 'gemini_zeroshot' AND created_at >= ?", Long.class, since));
        llm.put("llmClassifiedShare", jdbc.queryForObject("SELECT COALESCE(avg((category_source = 'LLM')::int), 0) FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? AND category IS NOT NULL", Double.class, since));
        llm.put("costByDay", jdbc.queryForList("SELECT to_char(created_at, 'YYYY-MM-DD') AS day, sum(cost_usd) AS cost_usd, count(*) AS calls FROM drafts WHERE created_at >= ? AND cost_usd > 0 GROUP BY 1 ORDER BY 1", since));
        out.put("llm", llm);

        // ---- classification quality from agent corrections
        out.put("labelCorrections", jdbc.queryForMap("""
                SELECT count(*) FILTER (WHERE corrected_category IS NOT NULL) AS category_corrections,
                       count(*) FILTER (WHERE corrected_priority IS NOT NULL) AS priority_corrections FROM feedback WHERE created_at >= ?
                """, since));

        // ---- pipeline latency (seconds)
        out.put("latencySeconds", jdbc.queryForMap("""
                SELECT COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (triaged_at - ingested_at))), 0) AS triage_p50,
                       COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (triaged_at - ingested_at))), 0) AS triage_p95
                FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? AND triaged_at IS NOT NULL
                """, since));
        out.put("draftLatencySeconds", jdbc.queryForMap("""
                SELECT COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (drafted_at - ingested_at))), 0) AS draft_p50,
                       COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (drafted_at - ingested_at))), 0) AS draft_p95
                FROM tickets WHERE source <> 'SEED' AND ingested_at >= ? AND drafted_at IS NOT NULL
                """, since));
        out.put("queues", jobs.depthByStatus());
        out.put("activeIncidents", jdbc.queryForObject("SELECT count(*) FROM duplicate_clusters WHERE is_incident AND status = 'ACTIVE'", Long.class));
        out.put("windowDays", days);
        return out;
    }

    /** Compact list for charts of clusters growing over time. */
    @GetMapping("/api/metrics/incidents")
    public List<Map<String, Object>> incidents() {
        return jdbc.queryForList("SELECT id, title, product, size, first_seen_at, last_seen_at, is_incident FROM duplicate_clusters WHERE is_incident ORDER BY first_seen_at DESC LIMIT 50");
    }
}
