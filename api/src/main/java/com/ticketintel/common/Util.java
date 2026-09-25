package com.ticketintel.common;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;

public final class Util {
    private Util() {}

    public static final Set<String> TIERS = Set.of("FREE", "PRO", "ENTERPRISE");
    public static final Set<String> PRODUCTS = Set.of("web-app", "mobile-app", "api", "desktop-sync", "admin-console");
    public static final List<String> PRIORITIES = List.of("LOW", "MEDIUM", "HIGH", "URGENT");
    public static final Set<String> CATEGORIES = Set.of("BILLING", "LOGIN_ACCESS", "BUG", "FEATURE_REQUEST", "PERFORMANCE",
            "INTEGRATION", "ACCOUNT_MANAGEMENT", "DATA_PRIVACY");

    public static String sha256(String s) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    /** pgvector text literal, e.g. "[0.1,0.2]" - bind with CAST(? AS vector). */
    public static String vec(float[] v) {
        StringBuilder sb = new StringBuilder(v.length * 9).append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(v[i]);
        }
        return sb.append(']').toString();
    }

    /** Parses a pgvector text value such as "[0.1,0.2]". */
    public static float[] parseVec(String s) {
        String[] parts = s.substring(1, s.length() - 1).split(",");
        float[] v = new float[parts.length];
        for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i]);
        return v;
    }

    public static OffsetDateTime utc(Instant i) {
        return OffsetDateTime.ofInstant(i, ZoneOffset.UTC);
    }

    public static Instant instant(ResultSet rs, String col) throws SQLException {
        OffsetDateTime o = rs.getObject(col, OffsetDateTime.class);
        return o == null ? null : o.toInstant();
    }

    public static Double dbl(ResultSet rs, String col) throws SQLException {
        double d = rs.getDouble(col);
        return rs.wasNull() ? null : d;
    }

    public static String trunc(String s, int n) {
        return s == null || s.length() <= n ? s : s.substring(0, n);
    }
}
