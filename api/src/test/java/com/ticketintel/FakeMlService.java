package com.ticketintel;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

/**
 * Deterministic stand-in for the Python ML service. Topic vectors: tickets mentioning the same keyword get (almost) the same
 * unit embedding, so duplicate detection / retrieval behave realistically. Failure switches simulate outages.
 */
public class FakeMlService {
    static final JsonMapper JSON = JsonMapper.builder().build();
    private static final String[] TOPICS = {"sso", "invoice", "slow", "webhook", "export"};

    public final AtomicBoolean mlDown = new AtomicBoolean();       // every endpoint returns 500
    public final AtomicBoolean llmDown = new AtomicBoolean();      // /v1/llm/* returns 503 LLM_UNAVAILABLE
    public final AtomicInteger draftCalls = new AtomicInteger();
    public final AtomicInteger classifyCalls = new AtomicInteger();
    public final AtomicInteger analyzeCalls = new AtomicInteger();
    public volatile double categoryConfidence = 0.95;              // < 0.8 makes the cascade call the LLM
    public volatile String lastDraftRequest = "";
    private final HttpServer server;

    public FakeMlService() {
        try {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        server.createContext("/v1/analyze", ex -> handle(ex, this::analyze));
        server.createContext("/v1/escalation", ex -> handle(ex, this::escalation));
        server.createContext("/v1/llm/classify", ex -> handle(ex, this::classify));
        server.createContext("/v1/llm/draft", ex -> handle(ex, this::draft));
        server.createContext("/v1/embed", ex -> handle(ex, this::embed));
        server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(4));
        server.start();
    }

    public String url() { return "http://127.0.0.1:" + server.getAddress().getPort(); }

    public void stop() { server.stop(0); }

    public void reset() {
        mlDown.set(false);
        llmDown.set(false);
        draftCalls.set(0);
        classifyCalls.set(0);
        analyzeCalls.set(0);
        categoryConfidence = 0.95;
    }

    // ---------------------------------------------------------------- helpers
    interface Handler { Object apply(JsonNode body); }

    private void handle(HttpExchange ex, Handler h) throws IOException {
        try {
            String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            if (mlDown.get()) { send(ex, 500, "{\"detail\":\"boom\"}"); return; }
            if (ex.getRequestURI().getPath().startsWith("/v1/llm") && llmDown.get()) {
                send(ex, 503, "{\"code\":\"LLM_UNAVAILABLE\",\"detail\":\"quota exhausted\"}");
                return;
            }
            send(ex, 200, JSON.writeValueAsString(h.apply(JSON.readTree(raw))));
        } catch (Exception e) {
            send(ex, 500, "{\"detail\":\"" + e + "\"}");
        }
    }

    private static void send(HttpExchange ex, int code, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(code, b.length);
        ex.getResponseBody().write(b);
        ex.close();
    }

    static int topic(String text) {
        String t = text.toLowerCase();
        for (int i = 0; i < TOPICS.length; i++) if (t.contains(TOPICS[i])) return i;
        return TOPICS.length;
    }

    static float[] embedding(String text) {
        float[] v = new float[384];
        v[topic(text)] = 1f;
        v[100 + Math.floorMod(text.hashCode(), 200)] = 0.05f;      // tiny per-text noise: similar, not identical
        double n = 0;
        for (float x : v) n += x * x;
        for (int i = 0; i < v.length; i++) v[i] /= (float) Math.sqrt(n);
        return v;
    }

    private static final String[] CATS = {"LOGIN_ACCESS", "BILLING", "PERFORMANCE", "INTEGRATION", "DATA_PRIVACY", "BUG"};

    // -------------------------------------------------------------- endpoints
    private Object analyze(JsonNode req) {
        analyzeCalls.incrementAndGet();
        List<Object> items = new ArrayList<>();
        for (JsonNode t : req.path("tickets")) {
            String text = t.path("subject").stringValue("") + " " + t.path("body").stringValue("");
            float[] e = embedding(text);
            Map<String, Object> it = new LinkedHashMap<>();
            it.put("ticketId", t.path("id").asLong());
            it.put("embedding", e);
            it.put("embeddingModel", "minilm-l6-v2@1");
            it.put("category", Map.of("label", CATS[topic(text)], "confidence", categoryConfidence, "model", "emb_lr@test", "probs", Map.of(CATS[topic(text)], categoryConfidence)));
            boolean urgent = text.toLowerCase().contains("urgent") || text.toLowerCase().contains("down");
            it.put("priority", Map.of("label", urgent ? "URGENT" : "MEDIUM", "confidence", 0.8, "model", "priority_lr@test",
                    "signals", urgent ? List.of("outage") : List.of(), "probs", Map.of("MEDIUM", 0.8)));
            it.put("needsLlm", categoryConfidence < 0.8);
            items.add(it);
        }
        return Map.of("items", items, "latencyMs", 1.0);
    }

    private Object escalation(JsonNode req) {
        List<Object> items = new ArrayList<>();
        for (JsonNode t : req.path("tickets")) {
            boolean ent = "ENTERPRISE".equals(t.path("tier").stringValue(""));
            double risk = ent ? 0.9 : Math.min(0.9, 0.1 + 0.1 * t.path("similarRecent").asInt());
            items.add(Map.of("ticketId", t.path("id").asLong(), "risk", risk, "rawScore", risk, "route", risk >= 0.4 ? "SENIOR" : "STANDARD",
                    "topFactors", List.of(Map.of("feature", "tier", "contribution", 0.5)), "model", "lgbm_platt@test"));
        }
        return Map.of("items", items, "seniorThreshold", 0.4);
    }

    private Object classify(JsonNode req) {
        classifyCalls.incrementAndGet();
        String text = req.path("subject").stringValue("") + " " + req.path("body").stringValue("");
        return Map.of("category", CATS[topic(text)], "priority", "HIGH", "confidence", 0.9, "rationale", "fake llm",
                "meta", meta());
    }

    private Object draft(JsonNode req) {
        draftCalls.incrementAndGet();
        lastDraftRequest = req.toString();
        JsonNode sources = req.path("sources");
        List<Object> cits = new ArrayList<>();
        StringBuilder reply = new StringBuilder("Thanks for reaching out. ");
        double best = 0;
        for (JsonNode s : sources) {
            best = Math.max(best, s.path("score").asDouble());
            if (cits.isEmpty()) {
                long id = s.path("id").asLong();
                reply.append(s.path("resolution").stringValue("")).append(" [T-").append(id).append("]");
                cits.add(Map.of("ticketId", id, "why", "closest resolved ticket"));
            }
        }
        if (cits.isEmpty()) reply.append("We are investigating; can you share any error message?");
        return Map.of("reply", reply.toString(), "citations", cits, "confidence", 0.8, "needsInfo", List.of(),
                "grounding", cits.isEmpty() ? "NONE" : best >= 0.75 ? "STRONG" : "WEAK", "strippedCitations", 0, "meta", meta());
    }

    private Object embed(JsonNode req) {
        List<Object> out = new ArrayList<>();
        for (JsonNode t : req.path("texts")) out.add(embedding(t.stringValue("")));
        return Map.of("embeddings", out, "model", "minilm-l6-v2@1");
    }

    private static Map<String, Object> meta() {
        return Map.of("model", "fake-llm", "latencyMs", 5, "inputTokens", 100, "outputTokens", 50, "costUsd", 0.0001, "cached", false,
                "promptVersion", "test");
    }
}
