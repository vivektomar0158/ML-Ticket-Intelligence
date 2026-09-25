package com.ticketintel;

import static org.assertj.core.api.Assertions.assertThat;

import com.ticketintel.common.Util;
import com.ticketintel.review.EditDistance;
import com.ticketintel.ticket.IngestService;
import com.ticketintel.ticket.IngestService.TicketRequest;
import org.junit.jupiter.api.Test;

class UnitTests {
    @Test
    void levenshtein() {
        assertThat(EditDistance.levenshtein("kitten", "sitting")).isEqualTo(3);
        assertThat(EditDistance.levenshtein("", "abc")).isEqualTo(3);
        assertThat(EditDistance.levenshtein("same", "same")).isZero();
        assertThat(EditDistance.ratio("same", "same")).isZero();
        assertThat(EditDistance.ratio("abc", "xyz")).isEqualTo(1.0);
        assertThat(EditDistance.ratio(null, null)).isZero();
    }

    @Test
    void vectorLiteralRoundTrips() {
        float[] v = {0.5f, -1.25f, 3f};
        assertThat(Util.parseVec(Util.vec(v))).containsExactly(v);
    }

    @Test
    void ingestValidation() {
        var ok = new TicketRequest("e", "s", "b", "web-app", "c", "PRO", null);
        assertThat(IngestService.validate(ok)).isNull();
        assertThat(IngestService.validate(new TicketRequest(null, "s", "b", "nope", "c", "PRO", null))).contains("product");
        assertThat(IngestService.validate(new TicketRequest(null, "s", "b", "api", "c", "GOLD", null))).contains("customerTier");
        assertThat(IngestService.validate(new TicketRequest(null, "x".repeat(301), "b", "api", "c", "PRO", null))).contains("300");
        assertThat(IngestService.validate(new TicketRequest(null, "s", " ", "api", "c", "PRO", null))).contains("body");
    }
}
