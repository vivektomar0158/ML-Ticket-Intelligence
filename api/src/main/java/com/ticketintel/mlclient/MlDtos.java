package com.ticketintel.mlclient;

import java.util.List;
import java.util.Map;

/** Wire contract of the Python ML service (see docs/contracts/ml-service.openapi.json). */
public final class MlDtos {
    private MlDtos() {}

    public record TicketIn(long id, String subject, String body, String tier) {}
    public record AnalyzeReq(List<TicketIn> tickets, boolean includeEmbedding) {}
    public record CategoryPred(String label, double confidence, String model, Map<String, Double> probs) {}
    public record PriorityPred(String label, double confidence, String model, List<String> signals, Map<String, Double> probs) {}
    public record AnalyzeItem(long ticketId, float[] embedding, String embeddingModel, CategoryPred category,
                              PriorityPred priority, boolean needsLlm) {}
    public record AnalyzeRes(List<AnalyzeItem> items, double latencyMs) {}

    public record EscIn(long id, String subject, String body, String tier, String product, int hour, int weekday,
                        int priorTickets7d, int similarRecent, float[] embedding) {}
    public record EscalationReq(List<EscIn> tickets) {}
    public record Contribution(String feature, double contribution) {}
    public record EscItem(long ticketId, double risk, double rawScore, String route, List<Contribution> topFactors, String model) {}
    public record EscalationRes(List<EscItem> items, double seniorThreshold) {}

    public record ClassifyReq(String subject, String body) {}
    public record LlmMeta(String model, int latencyMs, int inputTokens, int outputTokens, double costUsd, boolean cached,
                          String promptVersion) {}
    public record ClassifyRes(String category, String priority, double confidence, String rationale, LlmMeta meta) {}

    public record Source(long id, String subject, String problem, String resolution, double score) {}
    public record DraftTicket(String subject, String body, String category, String priority, boolean isIncident, int clusterSize) {}
    public record DraftReq(DraftTicket ticket, List<Source> sources, String instruction) {}
    public record Citation(long ticketId, String why) {}
    public record DraftRes(String reply, List<Citation> citations, double confidence, List<String> needsInfo, String grounding,
                           int strippedCitations, LlmMeta meta) {}

    public record EmbedReq(List<String> texts) {}
    public record EmbedRes(List<float[]> embeddings, String model) {}
}
