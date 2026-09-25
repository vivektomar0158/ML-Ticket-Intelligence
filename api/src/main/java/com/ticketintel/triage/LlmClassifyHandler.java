package com.ticketintel.triage;

import com.ticketintel.common.AppProperties;
import com.ticketintel.events.SseHub;
import com.ticketintel.jobs.JobHandler;
import com.ticketintel.jobs.JobRepository.Job;
import com.ticketintel.mlclient.MlClient;
import com.ticketintel.mlclient.MlClient.LlmUnavailableException;
import com.ticketintel.mlclient.MlDtos.ClassifyRes;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import tools.jackson.databind.json.JsonMapper;

/** Second stage of the classification cascade: only tickets the cheap model was unsure about reach the LLM. */
@Component
public class LlmClassifyHandler implements JobHandler {
    private final JdbcTemplate jdbc;
    private final MlClient ml;
    private final SseHub events;
    private final JsonMapper json;
    private final AppProperties props;

    public LlmClassifyHandler(JdbcTemplate jdbc, MlClient ml, SseHub events, JsonMapper json, AppProperties props) {
        this.jdbc = jdbc;
        this.ml = ml;
        this.events = events;
        this.json = json;
        this.props = props;
    }

    @Override public String type() { return "LLM_CLASSIFY"; }
    @Override public int batchSize() { return 1; }
    @Override public int threads() { return props.workers().classifyThreads(); }

    @Override
    public void handle(List<Job> jobs, Context ctx) {
        for (Job job : jobs) {
            List<Map<String, Object>> rows = jdbc.queryForList(
                    "SELECT subject, body, category_source, category, category_conf FROM tickets WHERE id = ?", job.ticketId());
            if (rows.isEmpty() || "AGENT".equals(rows.get(0).get("category_source"))) continue;   // agent's label always wins
            var t = rows.get(0);
            ClassifyRes res;
            try {
                res = ml.classify((String) t.get("subject"), (String) t.get("body"));
            } catch (LlmUnavailableException e) {
                if (Duration.between(job.createdAt(), Instant.now()).toHours() >= 2) continue;   // give up quietly: the model label stands
                ctx.defer(job, 120, "LLM unavailable");
                continue;
            }
            jdbc.update("""
                    INSERT INTO predictions (ticket_id, task, model_name, model_version, label, score, details, latency_ms)
                    VALUES (?, 'CATEGORY', 'gemini_zeroshot', ?, ?, ?, CAST(? AS jsonb), ?) ON CONFLICT DO NOTHING
                    """, job.ticketId(), res.meta().model(), res.category(), (float) res.confidence(),
                    json.writeValueAsString(Map.of("rationale", res.rationale(), "priority", res.priority(), "costUsd", res.meta().costUsd())),
                    res.meta().latencyMs());
            boolean changed = !res.category().equals(t.get("category"));
            jdbc.update("UPDATE tickets SET category = ?, category_conf = ?, category_source = 'LLM', version = version + 1 WHERE id = ? AND category_source <> 'AGENT'",
                    res.category(), (float) res.confidence(), job.ticketId());
            events.publish("ticket.updated", Map.of("ticketId", job.ticketId(), "category", res.category(), "changed", changed));
        }
    }
}
