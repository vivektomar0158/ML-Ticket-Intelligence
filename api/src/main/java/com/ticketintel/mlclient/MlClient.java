package com.ticketintel.mlclient;

import com.ticketintel.common.AppProperties;
import com.ticketintel.mlclient.MlDtos.*;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;

/**
 * HTTP client for the Python ML service. Two circuit breakers ("fast" models, "llm") with separate timeouts, so a slow or
 * failing LLM can never starve triage. LLM quota exhaustion is reported as {@link LlmUnavailableException} (not counted
 * as a breaker failure): callers defer and retry instead of burning attempts.
 */
@Component
public class MlClient {
    private static final Logger log = LoggerFactory.getLogger(MlClient.class);

    public static class MlUnavailableException extends RuntimeException {
        public MlUnavailableException(String m, Throwable c) { super(m, c); }
    }

    public static class LlmUnavailableException extends RuntimeException {
        public LlmUnavailableException(String m) { super(m); }
    }

    public static class MlBadRequestException extends RuntimeException {
        public MlBadRequestException(String m) { super(m); }
    }

    private final RestClient fast;
    private final RestClient llm;
    private final CircuitBreaker fastCb;
    private final CircuitBreaker llmCb;

    public MlClient(AppProperties props) {
        this.fast = build(props.mlServiceUrl(), props.ml().fastTimeoutMs());
        this.llm = build(props.mlServiceUrl(), props.ml().llmTimeoutMs());
        CircuitBreakerConfig cfg = CircuitBreakerConfig.custom()
                .slidingWindowSize(20).minimumNumberOfCalls(5).failureRateThreshold(50)
                .waitDurationInOpenState(Duration.ofSeconds(props.ml().breakerOpenSeconds()))
                .ignoreExceptions(LlmUnavailableException.class, MlBadRequestException.class).build();
        this.fastCb = CircuitBreaker.of("ml-fast", cfg);
        this.llmCb = CircuitBreaker.of("ml-llm", cfg);
    }

    private static RestClient build(String baseUrl, int timeoutMs) {
        HttpClient http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(3)).build();
        var factory = new JdkClientHttpRequestFactory(http);
        factory.setReadTimeout(Duration.ofMillis(timeoutMs));
        return RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();
    }

    public CircuitBreaker.State fastState() { return fastCb.getState(); }

    public CircuitBreaker.State llmState() { return llmCb.getState(); }

    // ------------------------------------------------------------------ fast models
    public AnalyzeRes analyze(List<TicketIn> tickets, boolean includeEmbedding) {
        return call(fastCb, 3, () -> post(fast, "/v1/analyze", new AnalyzeReq(tickets, includeEmbedding), AnalyzeRes.class));
    }

    public EscalationRes escalation(List<EscIn> tickets) {
        return call(fastCb, 3, () -> post(fast, "/v1/escalation", new EscalationReq(tickets), EscalationRes.class));
    }

    public EmbedRes embed(List<String> texts) {
        return call(fastCb, 3, () -> post(fast, "/v1/embed", new EmbedReq(texts), EmbedRes.class));
    }

    // -------------------------------------------------------------------- LLM tasks
    public ClassifyRes classify(String subject, String body) {
        return call(llmCb, 2, () -> post(llm, "/v1/llm/classify", new ClassifyReq(subject, body), ClassifyRes.class));
    }

    public DraftRes draft(DraftReq req) {
        return call(llmCb, 2, () -> post(llm, "/v1/llm/draft", req, DraftRes.class));
    }

    // --------------------------------------------------------------------- internals
    private <T> T post(RestClient c, String path, Object body, Class<T> type) {
        try {
            return c.post().uri(path).contentType(MediaType.APPLICATION_JSON).body(body).retrieve().body(type);
        } catch (RestClientResponseException e) {
            String resp = e.getResponseBodyAsString();
            int s = e.getStatusCode().value();
            if (s == 503 && resp.contains("LLM_UNAVAILABLE")) throw new LlmUnavailableException(resp);
            if (s >= 400 && s < 500) throw new MlBadRequestException("ML " + s + ": " + resp);
            throw new MlUnavailableException("ML " + s + ": " + com.ticketintel.common.Util.trunc(resp, 200), e);
        } catch (ResourceAccessException e) {
            throw new MlUnavailableException("ML unreachable: " + e.getMessage(), e);
        }
    }

    private <T> T call(CircuitBreaker cb, int attempts, Supplier<T> s) {
        MlUnavailableException last = null;
        for (int i = 1; i <= attempts; i++) {
            try {
                return cb.executeSupplier(s);
            } catch (CallNotPermittedException e) {
                throw new MlUnavailableException("circuit open: " + cb.getName(), e);
            } catch (MlUnavailableException e) {
                last = e;
                log.warn("ML call failed (attempt {}/{}): {}", i, attempts, e.getMessage());
                if (i < attempts) {
                    try { Thread.sleep(200L * i); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                }
            }
        }
        throw last;
    }
}
