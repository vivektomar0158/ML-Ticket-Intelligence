package com.ticketintel.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

/**
 * Model drift watch without labels: compares the last 7 days with the 30 days before, on (a) the share of tickets the cheap
 * classifier was unsure about and (b) how the category mix moved (total variation distance). A jump means new kinds of
 * tickets are arriving that the model was not trained on: time to look at corrections and retrain.
 */
@Service
public class DriftService {
    private static final Logger log = LoggerFactory.getLogger(DriftService.class);
    static final double LOW_CONF = 0.8;
    private final JdbcTemplate jdbc;
    private final AtomicReference<Double> lowShare = new AtomicReference<>(0.0);
    private final AtomicReference<Double> shift = new AtomicReference<>(0.0);

    public DriftService(JdbcTemplate jdbc, MeterRegistry metrics) {
        this.jdbc = jdbc;
        metrics.gauge("drift.low_confidence_share", lowShare, AtomicReference::get);
        metrics.gauge("drift.category_shift", shift, AtomicReference::get);
    }

    private Map<String, Object> window(String from, String to) {
        Map<String, Object> w = jdbc.queryForMap("""
                SELECT count(*) AS n, COALESCE(avg((p.score < ?)::int), 0) AS low_share, COALESCE(avg(p.score), 0) AS mean_conf
                FROM predictions p JOIN tickets t ON t.id = p.ticket_id
                WHERE p.task = 'CATEGORY' AND p.model_name <> 'gemini_zeroshot' AND t.source <> 'SEED'
                  AND p.created_at >= now() - CAST(? AS interval) AND p.created_at < now() - CAST(? AS interval)
                """, (float) LOW_CONF, from, to);
        Map<String, Double> mix = new LinkedHashMap<>();
        jdbc.query("""
                SELECT p.label, count(*) AS c FROM predictions p JOIN tickets t ON t.id = p.ticket_id
                WHERE p.task = 'CATEGORY' AND p.model_name <> 'gemini_zeroshot' AND t.source <> 'SEED'
                  AND p.created_at >= now() - CAST(? AS interval) AND p.created_at < now() - CAST(? AS interval) GROUP BY 1
                """, rs -> { mix.put(rs.getString(1), rs.getDouble(2)); }, from, to);
        double total = mix.values().stream().mapToDouble(Double::doubleValue).sum();
        Map<String, Double> dist = new LinkedHashMap<>();
        mix.forEach((k, v) -> dist.put(k, total == 0 ? 0 : v / total));
        var out = new LinkedHashMap<String, Object>(w);
        out.put("mix", dist);
        return out;
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> compute() {
        Map<String, Object> recent = window("7 days", "0 seconds"), base = window("37 days", "7 days");
        Map<String, Double> a = (Map<String, Double>) recent.get("mix"), b = (Map<String, Double>) base.get("mix");
        Set<String> keys = new HashSet<>(a.keySet());
        keys.addAll(b.keySet());
        double tvd = 0;
        for (String k : keys) tvd += Math.abs(a.getOrDefault(k, 0.0) - b.getOrDefault(k, 0.0)) / 2;
        long nr = ((Number) recent.get("n")).longValue(), nb = ((Number) base.get("n")).longValue();
        double lr = ((Number) recent.get("low_share")).doubleValue(), lb = ((Number) base.get("low_share")).doubleValue();

        List<String> reasons = new ArrayList<>();
        boolean enough = nr >= 30 && nb >= 30;                       // never alarm on tiny samples
        if (enough && lr > 0.05 && lr > 2 * lb) reasons.add("share of low-confidence tickets is %.0f%% vs %.0f%% baseline".formatted(lr * 100, lb * 100));
        if (enough && tvd > 0.2) reasons.add("category mix shifted (total variation %.2f)".formatted(tvd));
        lowShare.set(lr);
        shift.set(tvd);

        var out = new LinkedHashMap<String, Object>();
        out.put("recent7d", Map.of("n", nr, "lowConfidenceShare", lr, "meanConfidence", ((Number) recent.get("mean_conf")).doubleValue()));
        out.put("baseline30d", Map.of("n", nb, "lowConfidenceShare", lb, "meanConfidence", ((Number) base.get("mean_conf")).doubleValue()));
        out.put("categoryShift", tvd);
        out.put("enoughData", enough);
        out.put("alert", !reasons.isEmpty());
        out.put("reasons", reasons);
        return out;
    }

    @Scheduled(fixedDelay = 3_600_000, initialDelay = 120_000)
    void hourly() {
        var r = compute();
        if (Boolean.TRUE.equals(r.get("alert"))) log.warn("MODEL DRIFT ALERT: {}", r.get("reasons"));
    }
}
