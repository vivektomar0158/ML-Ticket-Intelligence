package com.ticketintel.auth;

import com.ticketintel.common.AppProperties;
import com.ticketintel.common.ApiException;
import jakarta.validation.constraints.NotBlank;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.security.core.Authentication;

@RestController
@Validated
public class AuthController {
    public record LoginRequest(@NotBlank String username, @NotBlank String password) {}

    private final JdbcTemplate jdbc;
    private final PasswordEncoder encoder;
    private final JwtEncoder jwt;
    private final AppProperties props;

    public AuthController(JdbcTemplate jdbc, PasswordEncoder encoder, JwtEncoder jwt, AppProperties props) {
        this.jdbc = jdbc;
        this.encoder = encoder;
        this.jwt = jwt;
        this.props = props;
    }

    @PostMapping("/api/auth/login")
    public Map<String, Object> login(@RequestBody @jakarta.validation.Valid LoginRequest req) {
        List<Map<String, Object>> rows = jdbc.queryForList(
                "SELECT id, password_hash, display_name, role FROM agents WHERE username = ?", req.username());
        // same error for unknown user and wrong password (no user enumeration)
        if (rows.isEmpty() || !encoder.matches(req.password(), (String) rows.get(0).get("password_hash"))) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Invalid credentials");
        }
        var row = rows.get(0);
        Instant now = Instant.now();
        Duration ttl = Duration.ofHours(props.security().jwtTtlHours());
        var claims = JwtClaimsSet.builder().subject(req.username()).issuedAt(now).expiresAt(now.plus(ttl))
                .claim("role", row.get("role")).claim("name", row.get("display_name")).claim("uid", row.get("id")).build();
        String token = jwt.encode(JwtEncoderParameters.from(JwsHeader.with(MacAlgorithm.HS256).build(), claims)).getTokenValue();
        return Map.of("accessToken", token, "expiresIn", ttl.toSeconds(), "role", row.get("role"),
                "name", row.get("display_name"), "username", req.username());
    }

    @GetMapping("/api/auth/me")
    public Map<String, Object> me(Authentication auth) {
        return Map.of("username", auth.getName(), "authorities", auth.getAuthorities().stream().map(Object::toString).toList());
    }
}
