package com.ticketintel.review;

/** Levenshtein distance with two rolling rows (O(min(n,m)) memory). */
public final class EditDistance {
    private EditDistance() {}

    public static int levenshtein(String a, String b) {
        if (a == null) a = "";
        if (b == null) b = "";
        if (a.length() < b.length()) { String t = a; a = b; b = t; }     // b is the shorter one
        int[] prev = new int[b.length() + 1], cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] t = prev; prev = cur; cur = t;
        }
        return prev[b.length()];
    }

    /** 0 = identical, 1 = completely rewritten. */
    public static double ratio(String a, String b) {
        int max = Math.max(a == null ? 0 : a.length(), b == null ? 0 : b.length());
        return max == 0 ? 0 : (double) levenshtein(a, b) / max;
    }
}
