package com.ticketintel.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

/** Dev convenience: seed agent / senior / admin users (password = username). Disable with SEED_USERS=false. */
@Component
public class SeedUsers implements ApplicationRunner {
    private static final Logger log = LoggerFactory.getLogger(SeedUsers.class);
    private final JdbcTemplate jdbc;
    private final PasswordEncoder encoder;
    private final AppProperties props;

    public SeedUsers(JdbcTemplate jdbc, PasswordEncoder encoder, AppProperties props) {
        this.jdbc = jdbc;
        this.encoder = encoder;
        this.props = props;
    }

    @Override
    public void run(ApplicationArguments args) {
        if (!props.security().seedUsers()) return;
        String[][] users = {{"agent", "Alex Agent", "AGENT"}, {"agent2", "Sam Support", "AGENT"},
                {"senior", "Sasha Senior", "SENIOR"}, {"admin", "Ada Admin", "ADMIN"}};
        for (String[] u : users) {
            int n = jdbc.update("INSERT INTO agents (username, password_hash, display_name, role) VALUES (?,?,?,?) ON CONFLICT (username) DO NOTHING",
                    u[0], encoder.encode(u[0]), u[1], u[2]);
            if (n > 0) log.info("seeded user {} ({})", u[0], u[2]);
        }
    }
}
