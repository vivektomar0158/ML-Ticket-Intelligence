package com.ticketintel.common;

import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app")
public record AppProperties(
        String mlServiceUrl,
        Ml ml,
        Security security,
        Workers workers,
        Dedup dedup,
        Retrieval retrieval,
        Draft draft) {

    public record Ml(int fastTimeoutMs, int llmTimeoutMs, int breakerOpenSeconds) {}

    public record Security(String jwtSecret, int jwtTtlHours, String ingestApiKey, boolean seedUsers, List<String> corsOrigins) {}

    public record Workers(boolean enabled, int triageThreads, int triageBatch, int classifyThreads, int draftThreads,
                          int pollMinMs, int pollMaxMs, int visibilityTimeoutMinutes) {}

    public record Dedup(double tau, int windowHours, int burstMinutes, int incidentMinTickets, int incidentMinCustomers) {}

    public record Retrieval(int topK) {}

    public record Draft(int shedQueueDepth, int incidentReuseHours, int maxWaitHours) {}
}
