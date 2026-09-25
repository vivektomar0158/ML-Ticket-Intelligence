package com.ticketintel.review;

import com.ticketintel.review.ReviewService.ReviewRequest;
import java.util.Map;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

@RestController
public class ReviewController {
    public record ResolveRequest(String body, String correctedCategory, String correctedPriority, Integer rating) {}

    private final ReviewService service;

    public ReviewController(ReviewService service) {
        this.service = service;
    }

    @PostMapping("/api/drafts/{id}/review")
    public Map<String, Object> review(@PathVariable long id, @RequestBody ReviewRequest req, Authentication auth) {
        return service.review(id, req, auth.getName());
    }

    @PostMapping("/api/tickets/{id}/resolve")
    public Map<String, Object> resolve(@PathVariable long id, @RequestBody ResolveRequest req, Authentication auth) {
        return service.resolveManually(id, req.body(), req.correctedCategory(), req.correctedPriority(), req.rating(), auth.getName());
    }
}
