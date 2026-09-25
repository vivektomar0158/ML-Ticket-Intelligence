package com.ticketintel.common;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/** Machine-to-machine ingestion: header {@code X-API-Key} grants role INGEST (ticket ingestion endpoints only). */
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    private final AppProperties props;

    public ApiKeyFilter(AppProperties props) {
        this.props = props;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest req, HttpServletResponse res, FilterChain chain)
            throws ServletException, IOException {
        String key = req.getHeader("X-API-Key");
        if (key != null && SecurityConfig.constantTimeEquals(key, props.security().ingestApiKey())) {
            var auth = new UsernamePasswordAuthenticationToken("ingest-api", null, List.of(new SimpleGrantedAuthority("ROLE_INGEST")));
            SecurityContextHolder.getContext().setAuthentication(auth);
        }
        chain.doFilter(req, res);
    }
}
