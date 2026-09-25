package com.ticketintel.ticket;

import com.ticketintel.common.ApiException;
import com.ticketintel.ticket.IngestService.IngestResult;
import com.ticketintel.ticket.IngestService.TicketRequest;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import tools.jackson.databind.json.JsonMapper;

@RestController
public class IngestController {
    static final int MAX_BATCH = 500;
    static final int MAX_CSV_ROWS = 50_000;

    private final IngestService ingest;
    private final JdbcTemplate jdbc;
    private final JsonMapper json;

    public IngestController(IngestService ingest, JdbcTemplate jdbc, JsonMapper json) {
        this.ingest = ingest;
        this.jdbc = jdbc;
        this.json = json;
    }

    /** Idempotent: the same Idempotency-Key (or externalId) returns the original ticket instead of creating a new one. */
    @PostMapping("/api/tickets")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, Object> create(@RequestBody TicketRequest req,
                                      @RequestHeader(value = "Idempotency-Key", required = false) String idemKey) {
        if (req.externalId() == null && idemKey != null && !idemKey.isBlank()) {
            req = new TicketRequest("idem:" + idemKey, req.subject(), req.body(), req.product(), req.customerId(),
                    req.customerTier(), req.createdAt());
        }
        IngestResult r = ingest.ingest(req, "API", null, null);
        if (r.error() != null) throw ApiException.badRequest(r.error());
        return Map.of("id", r.id(), "status", r.status().equals("CREATED") ? "NEW" : "EXISTING", "duplicate", r.status().equals("DUPLICATE"));
    }

    @PostMapping("/api/tickets/batch")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, Object> batch(@RequestBody List<TicketRequest> reqs) {
        if (reqs.isEmpty() || reqs.size() > MAX_BATCH) throw ApiException.badRequest("batch must contain 1.." + MAX_BATCH + " tickets");
        Map<String, Long> cache = new HashMap<>();
        List<IngestResult> results = new ArrayList<>();
        for (TicketRequest r : reqs) results.add(ingest.ingest(r, "API", null, cache));
        return Map.of("results", results,
                "created", results.stream().filter(x -> x.status().equals("CREATED")).count(),
                "duplicates", results.stream().filter(x -> x.status().equals("DUPLICATE")).count(),
                "rejected", results.stream().filter(x -> x.status().equals("REJECTED")).count());
    }

    /**
     * CSV columns (case-insensitive): subject, body, product, customer_tier [, external_id|ticket_uid, created_at, customer_id].
     * Rows are validated and inserted individually, so one bad row never fails the file; errors are reported per row.
     */
    @PostMapping("/api/ingest/csv")
    @ResponseStatus(HttpStatus.ACCEPTED)
    public Map<String, Object> csv(@RequestParam("file") MultipartFile file, Authentication auth) throws IOException {
        if (file.isEmpty()) throw ApiException.badRequest("empty file");
        Long batchId = jdbc.queryForObject("INSERT INTO ingest_batches (filename, status, created_by) VALUES (?, 'PROCESSING', ?) RETURNING id",
                Long.class, file.getOriginalFilename(), auth.getName());
        int total = 0, accepted = 0, dups = 0, rejected = 0;
        List<Map<String, Object>> errors = new ArrayList<>();
        Map<String, Long> cache = new HashMap<>();
        var format = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).setIgnoreHeaderCase(true).setTrim(true).build();
        try (var reader = new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8);
             CSVParser parser = CSVParser.parse(reader, format)) {
            var headers = parser.getHeaderMap().keySet().stream().map(String::toLowerCase).toList();
            for (String required : List.of("subject", "body", "product", "customer_tier")) {
                if (!headers.contains(required)) throw ApiException.badRequest("missing required column: " + required);
            }
            for (CSVRecord rec : parser) {
                if (++total > MAX_CSV_ROWS) throw ApiException.badRequest("file exceeds " + MAX_CSV_ROWS + " rows");
                IngestResult r;
                try {
                    String ext = opt(rec, "external_id");
                    if (ext == null) ext = opt(rec, "ticket_uid");
                    String ts = opt(rec, "created_at");
                    Instant created = ts == null ? null : OffsetDateTime.parse(ts.contains("+") || ts.endsWith("Z") ? ts.replace(' ', 'T') : ts.replace(' ', 'T') + "Z").toInstant();
                    r = ingest.ingest(new TicketRequest(ext, rec.get("subject"), rec.get("body"), rec.get("product"),
                            opt(rec, "customer_id"), rec.get("customer_tier"), created), "CSV", batchId, cache);
                } catch (Exception e) {
                    r = new IngestResult(null, "REJECTED", "row unparseable: " + e.getMessage());
                }
                switch (r.status()) {
                    case "CREATED" -> accepted++;
                    case "DUPLICATE" -> dups++;
                    default -> {
                        rejected++;
                        if (errors.size() < 100) errors.add(Map.of("row", rec.getRecordNumber() + 1, "error", r.error()));
                    }
                }
            }
        }
        jdbc.update("UPDATE ingest_batches SET status = 'DONE', total = ?, accepted = ?, duplicates = ?, rejected = ?, errors = CAST(? AS jsonb) WHERE id = ?",
                total, accepted, dups, rejected, json.writeValueAsString(errors), batchId);
        var out = new LinkedHashMap<String, Object>();
        out.put("batchId", batchId);
        out.put("total", total);
        out.put("accepted", accepted);
        out.put("duplicates", dups);
        out.put("rejected", rejected);
        out.put("errors", errors);
        return out;
    }

    @GetMapping("/api/ingest/batches/{id}")
    public Map<String, Object> batchStatus(@PathVariable long id) {
        return ingest.batchStatus(id);
    }

    private static String opt(CSVRecord r, String name) {
        if (!r.isMapped(name)) return null;
        String v = r.get(name);
        return v == null || v.isBlank() ? null : v;
    }
}
