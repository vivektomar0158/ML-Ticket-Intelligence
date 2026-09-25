package com.ticketintel;

import static org.assertj.core.api.Assertions.assertThat;

import com.ticketintel.jobs.JobRepository;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.postgresql.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;
import tools.jackson.databind.JsonNode;

/**
 * End-to-end tests against a real pgvector Postgres (Testcontainers) and a fake ML service (JDK HTTP server), covering the
 * pipeline, failure modes (ML down, LLM down), concurrency of the queue, security and the review/feedback loop.
 */
@Testcontainers
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
        "app.workers.poll-min-ms=50", "app.workers.poll-max-ms=200", "app.ml.breaker-open-seconds=2"})
class PipelineIT {

    @Container
    @ServiceConnection
    static PostgreSQLContainer pg = new PostgreSQLContainer(DockerImageName.parse("pgvector/pgvector:pg16").asCompatibleSubstituteFor("postgres"));

    static final FakeMlService ml = new FakeMlService();

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("app.ml-service-url", ml::url);
    }

    @AfterAll
    static void stop() { ml.stop(); }

    @org.springframework.boot.test.web.server.LocalServerPort int port;
    @Autowired JdbcTemplate jdbc;
    @Autowired JobRepository jobs;

    final HttpClient http = HttpClient.newHttpClient();

    // ------------------------------------------------------------------ helpers
    record Res(int status, JsonNode body) {}

    Res call(String method, String path, Object body, String token, String... headers) throws Exception {
        var b = HttpRequest.newBuilder(URI.create("http://localhost:" + port + path)).timeout(Duration.ofSeconds(30));
        if (token != null) b.header("Authorization", "Bearer " + token);
        for (int i = 0; i < headers.length; i += 2) b.header(headers[i], headers[i + 1]);
        b.header("Content-Type", "application/json");
        var pub = body == null ? HttpRequest.BodyPublishers.noBody() : HttpRequest.BodyPublishers.ofString(body instanceof String s ? s : FakeMlService.JSON.writeValueAsString(body));
        HttpResponse<String> r = http.send(b.method(method, pub).build(), HttpResponse.BodyHandlers.ofString());
        return new Res(r.statusCode(), r.body() == null || r.body().isBlank() ? null : FakeMlService.JSON.readTree(r.body()));
    }

    String login(String user) throws Exception {
        Res r = call("POST", "/api/auth/login", Map.of("username", user, "password", user), null);
        assertThat(r.status()).isEqualTo(200);
        return r.body().path("accessToken").stringValue("");
    }

    Map<String, Object> ticket(String subject, String body, String customer, String tier) {
        return Map.of("subject", subject, "body", body, "product", "web-app", "customerId", customer, "customerTier", tier);
    }

    long ingest(String subject, String body, String customer, String tier) throws Exception {
        Res r = call("POST", "/api/tickets", ticket(subject, body, customer, tier), login("agent"));
        assertThat(r.status()).isEqualTo(202);
        return r.body().path("id").asLong();
    }

    void await(String what, BooleanSupplier cond) throws Exception {
        long end = System.currentTimeMillis() + 30_000;
        while (System.currentTimeMillis() < end) {
            if (cond.getAsBoolean()) return;
            Thread.sleep(100);
        }
        throw new AssertionError("timed out waiting for: " + what);
    }

    String col(long id, String column) {
        return jdbc.queryForObject("SELECT " + column + "::text FROM tickets WHERE id = ?", String.class, id);
    }

    void awaitDrafted(long... ids) throws Exception {
        for (long id : ids) await("ticket " + id + " drafted", () -> "READY".equals(col(id, "draft_state")));
    }

    @BeforeEach
    void clean() throws Exception {
        jdbc.update("DELETE FROM jobs");
        Thread.sleep(300);
        jdbc.execute("TRUNCATE tickets, customers, duplicate_clusters, ingest_batches RESTART IDENTITY CASCADE");
        ml.reset();
    }

    // ------------------------------------------------------------------- tests
    @Test
    void schemaAndPgvectorAreInstalled() {
        assertThat(jdbc.queryForObject("SELECT extname FROM pg_extension WHERE extname = 'vector'", String.class)).isEqualTo("vector");
        assertThat(jdbc.queryForObject("SELECT count(*) FROM information_schema.tables WHERE table_name IN ('tickets','ticket_embeddings','jobs','drafts','feedback','predictions')", Integer.class)).isEqualTo(6);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM sla_policies", Integer.class)).isEqualTo(12);
    }

    @Test
    void securityRolesTokensAndApiKey() throws Exception {
        assertThat(call("GET", "/api/tickets", null, null).status()).isEqualTo(401);
        assertThat(call("POST", "/api/auth/login", Map.of("username", "agent", "password", "wrong"), null).status()).isEqualTo(401);
        String agent = login("agent"), admin = login("admin");
        assertThat(call("GET", "/api/tickets", null, agent).status()).isEqualTo(200);
        assertThat(call("GET", "/api/admin/status", null, agent).status()).isEqualTo(403);
        assertThat(call("GET", "/api/admin/status", null, admin).status()).isEqualTo(200);
        // machine ingestion with the API key works but grants nothing else
        Res ok = call("POST", "/api/tickets", ticket("SSO down", "sso fails", "K1", "PRO"), null, "X-API-Key", "dev-ingest-key");
        assertThat(ok.status()).isEqualTo(202);
        assertThat(call("GET", "/api/tickets", null, null, "X-API-Key", "dev-ingest-key").status()).isEqualTo(403);
        assertThat(call("POST", "/api/tickets", ticket("x", "y", "K1", "PRO"), null, "X-API-Key", "wrong").status()).isEqualTo(401);
    }

    @Test
    void ingestIsIdempotentAndValidated() throws Exception {
        String t = login("agent");
        Res a = call("POST", "/api/tickets", ticket("SSO fails", "cannot sign in with sso", "C1", "PRO"), t, "Idempotency-Key", "abc-1");
        Res b = call("POST", "/api/tickets", ticket("SSO fails", "cannot sign in with sso", "C1", "PRO"), t, "Idempotency-Key", "abc-1");
        assertThat(a.body().path("id").asLong()).isEqualTo(b.body().path("id").asLong());
        assertThat(b.body().path("duplicate").asBoolean()).isTrue();
        assertThat(jdbc.queryForObject("SELECT count(*) FROM tickets", Integer.class)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM jobs WHERE type = 'TRIAGE'", Integer.class)).isLessThanOrEqualTo(1);
        // validation
        assertThat(call("POST", "/api/tickets", Map.of("subject", "", "body", "b", "product", "web-app", "customerTier", "PRO"), t).status()).isEqualTo(400);
        assertThat(call("POST", "/api/tickets", Map.of("subject", "s", "body", "b", "product", "nope", "customerTier", "PRO"), t).status()).isEqualTo(400);
        // batch: partial acceptance
        Res batch = call("POST", "/api/tickets/batch", List.of(ticket("ok one", "invoice question", "C2", "FREE"), Map.of("subject", "bad", "body", "x", "product", "zzz", "customerTier", "FREE")), t);
        assertThat(batch.body().path("created").asInt()).isEqualTo(1);
        assertThat(batch.body().path("rejected").asInt()).isEqualTo(1);
    }

    @Test
    void csvUploadReportsRowErrorsWithoutFailingTheFile() throws Exception {
        String csv = "subject,body,product,customer_tier,ticket_uid,created_at\n"
                + "SSO login broken,sso fails for me,web-app,PRO,u1,2026-06-01 10:00:00\n"
                + "Invoice wrong,invoice double charge,web-app,FREE,u2,2026-06-01T10:05:00Z\n"
                + "Bad product,body,unknown,FREE,u3,2026-06-01 10:00:00\n"
                + "SSO login broken,sso fails for me,web-app,PRO,u1,2026-06-01 10:00:00\n";
        String boundary = "----b" + System.nanoTime();
        String payload = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"t.csv\"\r\nContent-Type: text/csv\r\n\r\n" + csv + "\r\n--" + boundary + "--\r\n";
        var req = HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/api/ingest/csv")).header("Authorization", "Bearer " + login("agent"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary).POST(HttpRequest.BodyPublishers.ofString(payload)).build();
        var r = http.send(req, HttpResponse.BodyHandlers.ofString());
        JsonNode j = FakeMlService.JSON.readTree(r.body());
        assertThat(r.statusCode()).isEqualTo(202);
        assertThat(j.path("accepted").asInt()).isEqualTo(2);
        assertThat(j.path("duplicates").asInt()).isEqualTo(1);
        assertThat(j.path("rejected").asInt()).isEqualTo(1);
        assertThat(j.path("errors").get(0).path("row").asInt()).isEqualTo(4);   // line in the file, counting the header
    }

    @Test
    void queueClaimsEachJobExactlyOnceUnderConcurrency() throws Exception {
        // REEMBED has no handler, so only this test's threads claim these jobs
        for (int i = 0; i < 300; i++) {
            long tid = jdbc.queryForObject("INSERT INTO tickets (source, customer_tier, product, subject, body, content_hash, created_at) VALUES ('API','FREE','web-app','s','b','h', now()) RETURNING id", Long.class);
            jobs.enqueue("REEMBED", tid, i % 7, null);
        }
        Set<Long> seen = Collections.newSetFromMap(new ConcurrentHashMap<>());
        List<Long> all = Collections.synchronizedList(new ArrayList<>());
        ExecutorService pool = Executors.newFixedThreadPool(8);
        CountDownLatch done = new CountDownLatch(8);
        for (int w = 0; w < 8; w++) {
            String name = "w" + w;
            pool.submit(() -> {
                try {
                    List<JobRepository.Job> batch;
                    while (!(batch = jobs.claim("REEMBED", 10, name)).isEmpty()) {
                        for (var j : batch) { all.add(j.id()); seen.add(j.id()); jobs.complete(j.id()); }
                    }
                } finally { done.countDown(); }
            });
        }
        assertThat(done.await(30, TimeUnit.SECONDS)).isTrue();
        pool.shutdown();
        assertThat(all).hasSize(300);                   // every job processed...
        assertThat(seen).hasSize(300);                  // ...exactly once (no duplicates across workers)
    }

    @Test
    void stuckRunningJobsAreReaped() {
        long tid = jdbc.queryForObject("INSERT INTO tickets (source, customer_tier, product, subject, body, content_hash, created_at) VALUES ('API','FREE','web-app','s','b','h', now()) RETURNING id", Long.class);
        jdbc.update("INSERT INTO jobs (type, ticket_id, status, locked_at, locked_by, attempts) VALUES ('REEMBED', ?, 'RUNNING', now() - interval '10 minutes', 'dead-worker', 1)", tid);
        assertThat(jobs.reap(5)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT status FROM jobs WHERE ticket_id = ?", String.class, tid)).isEqualTo("PENDING");
    }

    @Test
    void fullPipelineTriagesRoutesAndDrafts() throws Exception {
        long id = ingest("Cannot login via SSO", "Getting an SSO error since the update, urgent, everyone is down", "C1", "ENTERPRISE");
        awaitDrafted(id);
        assertThat(col(id, "status")).isEqualTo("DRAFTED");
        assertThat(col(id, "category")).isEqualTo("LOGIN_ACCESS");
        assertThat(col(id, "queue")).isEqualTo("SENIOR");                      // enterprise -> high risk -> senior queue
        assertThat(col(id, "category_source")).isEqualTo("MODEL");
        assertThat(jdbc.queryForObject("SELECT count(*) FROM predictions WHERE ticket_id = ?", Integer.class, id)).isEqualTo(3);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM ticket_embeddings WHERE ticket_id = ? AND is_open AND NOT is_resolved", Integer.class, id)).isEqualTo(1);
        // SLA: ENTERPRISE / URGENT = 15 minutes after creation
        assertThat(jdbc.queryForObject("SELECT extract(epoch FROM sla_due_at - created_at)::int / 60 FROM tickets WHERE id = ?", Integer.class, id)).isEqualTo(15);
        // no history yet -> ungrounded draft that asks questions instead of inventing a fix
        JsonNode d = call("GET", "/api/tickets/" + id, null, login("agent")).body();
        assertThat(d.path("draft").path("grounding").stringValue("")).isEqualTo("NONE");
        assertThat(d.path("predictions").size()).isEqualTo(3);
    }

    @Test
    void duplicateBurstBecomesAnIncidentAndLaterDraftsAreReused() throws Exception {
        List<Long> ids = new ArrayList<>();
        for (int i = 1; i <= 6; i++) ids.add(ingest("SSO login failing " + i, "sso sign in fails with error " + i, "C" + i, "PRO"));
        for (long id : ids) await("triaged " + id, () -> col(id, "triaged_at") != null);
        Set<String> clusters = new HashSet<>();
        for (long id : ids) clusters.add(col(id, "cluster_id"));
        assertThat(clusters).hasSize(1).doesNotContain((String) null);         // one cluster for the whole burst
        assertThat(jdbc.queryForObject("SELECT is_incident FROM duplicate_clusters", Boolean.class)).isTrue();
        assertThat(jdbc.queryForObject("SELECT size FROM duplicate_clusters", Integer.class)).isEqualTo(6);
        ids.forEach(id -> { try { awaitDrafted(id); } catch (Exception e) { throw new RuntimeException(e); } });

        int callsBefore = ml.draftCalls.get();
        long late = ingest("SSO login failing again", "sso sign in fails, same error", "C99", "PRO");
        awaitDrafted(late);
        assertThat(ml.draftCalls.get()).isEqualTo(callsBefore);                // zero extra LLM calls: canonical draft reused
        assertThat(jdbc.queryForObject("SELECT reused_from_draft_id IS NOT NULL FROM drafts WHERE ticket_id = ? AND status = 'GENERATED'", Boolean.class, late)).isTrue();
    }

    @Test
    void unrelatedTicketsDoNotCluster() throws Exception {
        long a = ingest("SSO login failing", "sso fails", "C1", "PRO"), b = ingest("Invoice is wrong", "invoice shows double charge", "C2", "PRO");
        await("both triaged", () -> col(a, "triaged_at") != null && col(b, "triaged_at") != null);
        assertThat(col(a, "cluster_id")).isNull();
        assertThat(col(b, "cluster_id")).isNull();
    }

    @Test
    void ragLoopResolvedTicketsBecomeEvidenceForNewOnes() throws Exception {
        String admin = login("admin"), agent = login("agent");
        float[] emb = FakeMlService.embedding("invoice charged twice");
        var h = new java.util.LinkedHashMap<String, Object>();
        h.put("externalId", "H1"); h.put("createdAt", "2026-05-01T10:00:00Z"); h.put("customerId", "HC1"); h.put("customerTier", "PRO");
        h.put("product", "web-app"); h.put("subject", "Invoice charged twice"); h.put("body", "invoice double charge");
        h.put("resolution", "Refund the duplicate charge from Billing > Invoices."); h.put("category", "BILLING"); h.put("priority", "MEDIUM");
        h.put("embedding", emb);
        Res imp = call("POST", "/api/admin/history", List.of(h), admin);
        assertThat(imp.body().path("imported").asInt()).isEqualTo(1);

        long t1 = ingest("Invoice problem", "my invoice looks wrong, charged twice", "C1", "PRO");
        awaitDrafted(t1);
        JsonNode d = call("GET", "/api/tickets/" + t1, null, agent).body();
        assertThat(d.path("draft").path("body").stringValue("")).contains("Refund the duplicate charge");   // grounded in the resolved ticket
        assertThat(d.path("draft").path("citations").get(0).path("ticketId").asLong()).isPositive();
        assertThat(d.path("sourceTickets").size()).isGreaterThanOrEqualTo(1);

        // agent edits + approves -> ticket resolved, becomes retrievable evidence right away
        long draftId = d.path("draft").path("id").asLong();
        Res rev = call("POST", "/api/drafts/" + draftId + "/review", Map.of("action", "EDIT_APPROVE", "finalBody", "Hi! We refunded the duplicate charge to your card; expect it in 3-5 days.", "rating", 4, "timeToReviewMs", 8000), agent);
        assertThat(rev.status()).isEqualTo(200);
        assertThat(col(t1, "status")).isEqualTo("RESOLVED");
        assertThat(jdbc.queryForObject("SELECT is_resolved AND NOT is_open FROM ticket_embeddings WHERE ticket_id = ?", Boolean.class, t1)).isTrue();
        assertThat(jdbc.queryForObject("SELECT edit_ratio FROM feedback WHERE ticket_id = ?", Double.class, t1)).isGreaterThan(0.2);
        assertThat(jdbc.queryForObject("SELECT drafts.status FROM drafts WHERE id = ?", String.class, draftId)).isEqualTo("EDITED");

        long t2 = ingest("Billing charge duplicated", "invoice duplicate charge again", "C2", "FREE");
        awaitDrafted(t2);
        JsonNode d2 = call("GET", "/api/tickets/" + t2, null, agent).body();
        List<Long> srcIds = new ArrayList<>();
        d2.path("sourceTickets").forEach(n -> srcIds.add(n.path("id").asLong()));
        assertThat(srcIds).contains(t1);                                     // the loop closed: t1's approved reply is now evidence
    }

    @Test
    void reviewRulesRejectDoubleReviewAndManualResolve() throws Exception {
        String agent = login("agent");
        long id = ingest("Webhook not firing", "webhook never fires", "C1", "PRO");
        awaitDrafted(id);
        long draftId = call("GET", "/api/tickets/" + id, null, agent).body().path("draft").path("id").asLong();
        assertThat(call("POST", "/api/drafts/" + draftId + "/review", Map.of("action", "REJECT"), agent).status()).isEqualTo(400);   // reason required
        assertThat(call("POST", "/api/drafts/" + draftId + "/review", Map.of("action", "EDIT_APPROVE", "finalBody", "short"), agent).status()).isEqualTo(400);
        assertThat(call("POST", "/api/drafts/" + draftId + "/review", Map.of("action", "REJECT", "rejectReason", "WRONG_INFO", "correctedCategory", "BUG"), agent).status()).isEqualTo(200);
        assertThat(call("POST", "/api/drafts/" + draftId + "/review", Map.of("action", "APPROVE"), agent).status()).isEqualTo(409);    // already reviewed
        assertThat(col(id, "status")).isEqualTo("IN_REVIEW");
        assertThat(col(id, "category")).isEqualTo("BUG");
        assertThat(col(id, "category_source")).isEqualTo("AGENT");           // agent correction wins over the model
        assertThat(call("POST", "/api/tickets/" + id + "/resolve", Map.of("body", "We fixed the webhook signing secret; please retry."), agent).status()).isEqualTo(200);
        assertThat(col(id, "status")).isEqualTo("RESOLVED");
        assertThat(jdbc.queryForObject("SELECT count(*) FROM feedback WHERE ticket_id = ?", Integer.class, id)).isEqualTo(2);
    }

    @Test
    void llmOutageDegradesGracefullyThenRecovers() throws Exception {
        ml.llmDown.set(true);
        long id = ingest("Export is failing", "export fails every time", "C1", "PRO");
        await("triaged despite LLM outage", () -> col(id, "triaged_at") != null);
        await("draft waiting", () -> "WAITING_LLM".equals(col(id, "draft_state")));
        assertThat(col(id, "status")).isEqualTo("TRIAGED");                    // fast path unaffected: predictions + routing are there
        assertThat(col(id, "category")).isNotNull();
        assertThat(jdbc.queryForObject("SELECT attempts FROM jobs WHERE ticket_id = ? AND type = 'DRAFT'", Integer.class, id)).isZero();   // deferred, not burned
        ml.llmDown.set(false);
        jdbc.update("UPDATE jobs SET run_after = now() WHERE ticket_id = ? AND type = 'DRAFT'", id);
        awaitDrafted(id);
    }

    @Test
    void mlServiceOutageDefersTriageWithoutLosingTickets() throws Exception {
        ml.mlDown.set(true);
        long id = ingest("SSO broken", "sso login broken", "C1", "PRO");
        Thread.sleep(3000);
        assertThat(col(id, "status")).isEqualTo("NEW");                        // ingestion stayed fast and accepted it
        assertThat(jdbc.queryForObject("SELECT status FROM jobs WHERE ticket_id = ? AND type = 'TRIAGE'", String.class, id)).isEqualTo("PENDING");
        assertThat(jdbc.queryForObject("SELECT attempts FROM jobs WHERE ticket_id = ? AND type = 'TRIAGE'", Integer.class, id)).isLessThanOrEqualTo(1);
        ml.mlDown.set(false);
        Thread.sleep(2500);                                                    // circuit breaker half-opens
        jdbc.update("UPDATE jobs SET run_after = now() WHERE ticket_id = ?", id);
        await("triaged after recovery", () -> col(id, "triaged_at") != null);
    }

    @Test
    void cascadeCallsTheLlmOnlyWhenTheCheapModelIsUnsure() throws Exception {
        ml.categoryConfidence = 0.95;
        long sure = ingest("SSO broken", "sso login broken", "C1", "PRO");
        awaitDrafted(sure);
        assertThat(ml.classifyCalls.get()).isZero();
        assertThat(col(sure, "category_source")).isEqualTo("MODEL");

        ml.categoryConfidence = 0.5;
        long unsure = ingest("Invoice weird", "invoice odd thing", "C2", "PRO");
        await("llm classified", () -> "LLM".equals(col(unsure, "category_source")));
        assertThat(ml.classifyCalls.get()).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM predictions WHERE ticket_id = ? AND task = 'CATEGORY'", Integer.class, unsure)).isEqualTo(2);   // both opinions kept
    }

    @Test
    void listFiltersAndKeysetPaginationCoverEveryTicketOnce() throws Exception {
        String agent = login("agent");
        List<Long> ids = new ArrayList<>();
        for (int i = 0; i < 25; i++) ids.add(ingest("Slow dashboard " + i, "dashboard is slow number " + i, "C" + (i % 5), i % 2 == 0 ? "ENTERPRISE" : "FREE"));
        for (long id : ids) await("triaged", () -> col(id, "triaged_at") != null);
        Set<Long> seen = new HashSet<>();
        String cursor = null;
        double lastRisk = 2;
        int pages = 0;
        do {
            Res r = call("GET", "/api/tickets?limit=10&sort=risk" + (cursor == null ? "" : "&cursor=" + cursor), null, agent);
            for (JsonNode t : r.body().path("items")) {
                assertThat(seen.add(t.path("id").asLong())).isTrue();          // no duplicates across pages
                assertThat(t.path("escalation_risk").isNumber()).isTrue();
                assertThat(t.path("escalation_risk").asDouble()).isLessThanOrEqualTo(lastRisk + 1e-9);   // strictly ordered by risk across pages
                lastRisk = t.path("escalation_risk").asDouble();
            }
            cursor = r.body().path("nextCursor").isNull() ? null : r.body().path("nextCursor").stringValue(null);
            pages++;
        } while (cursor != null && pages < 10);
        assertThat(seen).hasSize(25);
        Res senior = call("GET", "/api/tickets?queue=SENIOR&limit=100", null, agent);
        assertThat(senior.body().path("items").size()).isBetween(1, 25);
        for (JsonNode t : senior.body().path("items")) assertThat(t.path("queue").stringValue("")).isEqualTo("SENIOR");
        assertThat(call("GET", "/api/tickets?sort=bogus", null, agent).status()).isEqualTo(400);
        assertThat(call("GET", "/api/tickets/counts", null, agent).status()).isEqualTo(200);
    }

    @Test
    void metricsOverviewAndAdminEndpointsRespond() throws Exception {
        var h = new java.util.LinkedHashMap<String, Object>();       // seeded history must NOT count as live traffic
        h.put("externalId", "H-M1"); h.put("createdAt", "2026-05-01T10:00:00Z"); h.put("customerId", "HC9"); h.put("customerTier", "PRO");
        h.put("product", "web-app"); h.put("subject", "old"); h.put("body", "old ticket"); h.put("resolution", "old fix applied");
        h.put("category", "BUG"); h.put("priority", "LOW"); h.put("embedding", FakeMlService.embedding("old"));
        assertThat(call("POST", "/api/admin/history", List.of(h), login("admin")).status()).isEqualTo(200);
        long id = ingest("Webhook fails", "webhook failing", "C1", "PRO");
        awaitDrafted(id);
        Res m = call("GET", "/api/metrics/overview?days=1", null, login("agent"));
        assertThat(m.status()).isEqualTo(200);
        assertThat(m.body().path("totals").path("tickets").asInt()).isEqualTo(1);
        assertThat(m.body().has("drafts")).isTrue();
        assertThat(m.body().path("llm").path("draft_calls").asInt()).isEqualTo(1);
        assertThat(call("GET", "/api/admin/jobs", null, login("admin")).status()).isEqualTo(200);
    }

    @Test
    void driftMonitorFlagsAJumpInLowConfidenceTickets() throws Exception {
        String agent = login("agent");
        // baseline: 35 confident tickets, aged into the 7-37 day window
        for (int i = 0; i < 35; i++) ingest("SSO baseline " + i, "sso baseline ticket " + i, "C" + i, "FREE");
        await("baseline triaged", () -> jdbc.queryForObject("SELECT count(*) FROM tickets WHERE triaged_at IS NOT NULL", Integer.class) == 35);
        jdbc.update("UPDATE predictions SET created_at = now() - interval '20 days'");
        // recent week: the cheap model becomes unsure about most new tickets (new kind of traffic)
        ml.categoryConfidence = 0.55;
        for (int i = 0; i < 35; i++) ingest("Invoice odd " + i, "invoice strange thing " + i, "D" + i, "FREE");
        await("recent triaged", () -> jdbc.queryForObject("SELECT count(*) FROM tickets WHERE triaged_at IS NOT NULL", Integer.class) == 70);
        JsonNode d = call("GET", "/api/metrics/drift", null, agent).body();
        assertThat(d.path("enoughData").asBoolean()).isTrue();
        assertThat(d.path("alert").asBoolean()).isTrue();
        assertThat(d.path("recent7d").path("lowConfidenceShare").asDouble()).isGreaterThan(0.9);
        assertThat(d.path("baseline30d").path("lowConfidenceShare").asDouble()).isLessThan(0.1);
        assertThat(d.path("reasons").size()).isGreaterThanOrEqualTo(1);
    }

    @Test
    void driftMonitorStaysQuietOnSmallSamples() throws Exception {
        ingest("SSO one", "sso one", "C1", "FREE");
        await("triaged", () -> jdbc.queryForObject("SELECT count(*) FROM tickets WHERE triaged_at IS NOT NULL", Integer.class) == 1);
        JsonNode d = call("GET", "/api/metrics/drift", null, login("agent")).body();
        assertThat(d.path("enoughData").asBoolean()).isFalse();
        assertThat(d.path("alert").asBoolean()).isFalse();
    }

    @Test
    void serverSentEventsAnnounceTriage() throws Exception {
        String token = login("agent");
        var req = HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/api/events?access_token=" + token)).build();
        var resp = http.send(req, HttpResponse.BodyHandlers.ofInputStream());
        assertThat(resp.statusCode()).isEqualTo(200);
        var seen = new java.util.concurrent.atomic.AtomicBoolean();
        Thread reader = new Thread(() -> {
            try (var br = new BufferedReader(new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) if (line.contains("ticket.triaged")) { seen.set(true); return; }
            } catch (Exception ignored) { }
        });
        reader.setDaemon(true);
        reader.start();
        ingest("SSO broken", "sso broken", "C1", "PRO");
        await("SSE ticket.triaged", seen::get);
    }
}
