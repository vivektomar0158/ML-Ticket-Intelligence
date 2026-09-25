package com.ticketintel.retrieval;

import com.ticketintel.common.AppProperties;
import com.ticketintel.common.Util;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/**
 * Retrieval half of RAG over RESOLVED tickets: vector kNN (pgvector) + keyword (Postgres full-text), fused with
 * Reciprocal Rank Fusion. Error codes / product names are keyword-ish, so the keyword leg rescues cases where the
 * embedding is fuzzy. Final score reported to the LLM gateway is the cosine similarity (used to gauge grounding strength).
 */
@Component
public class HybridRetriever {

    public record Hit(long id, String subject, String problem, String resolution, double score, Long clusterId) {}

    private static final Pattern TOKEN = Pattern.compile("[a-z0-9]{3,}");
    private static final Set<String> STOP = Set.of("the", "and", "for", "that", "with", "this", "have", "our", "you", "your", "are",
            "was", "not", "but", "can", "all", "any", "get", "getting", "please", "help", "need", "when", "from", "since", "been", "would",
            "could", "there", "they", "their", "what", "just", "still", "into", "about", "after", "before", "some", "has", "had", "how");
    private static final int LEG = 20;

    private final JdbcTemplate jdbc;
    private final int topK;

    public HybridRetriever(JdbcTemplate jdbc, AppProperties props) {
        this.jdbc = jdbc;
        this.topK = props.retrieval().topK();
    }

    public List<Hit> retrieve(long ticketId, float[] embedding, String subject, String body) {
        String q = Util.vec(embedding);
        List<Long> vector = jdbc.queryForList("""
                SELECT ticket_id FROM ticket_embeddings WHERE is_resolved AND ticket_id <> ?
                ORDER BY embedding <=> CAST(? AS vector) LIMIT ?
                """, Long.class, ticketId, q, LEG);
        List<Long> keyword = keywordLeg(ticketId, subject + " " + body);

        Map<Long, Double> rrf = new LinkedHashMap<>();
        for (int r = 0; r < vector.size(); r++) rrf.merge(vector.get(r), 1.0 / (60 + r + 1), Double::sum);
        for (int r = 0; r < keyword.size(); r++) rrf.merge(keyword.get(r), 1.0 / (60 + r + 1), Double::sum);
        List<Long> ranked = rrf.entrySet().stream().sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(LEG).map(Map.Entry::getKey).toList();
        if (ranked.isEmpty()) return List.of();

        String ids = ranked.stream().map(String::valueOf).collect(Collectors.joining(","));
        Map<Long, Hit> byId = new LinkedHashMap<>();
        jdbc.query("""
                SELECT t.id, t.subject, t.body, t.resolution, t.cluster_id, 1 - (e.embedding <=> CAST(? AS vector)) AS cosine
                FROM tickets t JOIN ticket_embeddings e ON e.ticket_id = t.id
                WHERE t.id IN (%s) AND t.resolution IS NOT NULL
                """.formatted(ids), rs -> {
            byId.put(rs.getLong("id"), new Hit(rs.getLong("id"), rs.getString("subject"), Util.trunc(rs.getString("body"), 500),
                    Util.trunc(rs.getString("resolution"), 1200), rs.getDouble("cosine"), (Long) rs.getObject("cluster_id")));
        }, q);

        // diversity: at most one hit per duplicate cluster so 5 near-identical incident tickets don't crowd out other evidence
        List<Hit> out = new ArrayList<>();
        Set<Long> seenClusters = new HashSet<>();
        for (Long id : ranked) {
            Hit h = byId.get(id);
            if (h == null) continue;
            if (h.clusterId() != null && !seenClusters.add(h.clusterId())) continue;
            out.add(h);
            if (out.size() == topK) break;
        }
        return out;
    }

    private List<Long> keywordLeg(long ticketId, String text) {
        Set<String> tokens = new java.util.LinkedHashSet<>();
        Matcher m = TOKEN.matcher(text.toLowerCase(Locale.ROOT));
        while (m.find() && tokens.size() < 14) {
            String t = m.group();
            if (!STOP.contains(t)) tokens.add(t.replaceAll("[^a-z0-9]+", ""));
        }
        tokens.removeIf(String::isBlank);
        if (tokens.isEmpty()) return List.of();
        String tsquery = String.join(" | ", tokens);          // OR semantics: long tickets rarely match every term
        return jdbc.queryForList("""
                SELECT id FROM tickets, to_tsquery('english', ?) query
                WHERE status IN ('RESOLVED','CLOSED') AND resolution IS NOT NULL AND id <> ? AND search_tsv @@ query
                ORDER BY ts_rank_cd(search_tsv, query) DESC LIMIT ?
                """, Long.class, tsquery, ticketId, LEG);
    }
}
