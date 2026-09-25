package com.ticketintel.events;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
public class EventsController {
    private final SseHub hub;

    public EventsController(SseHub hub) {
        this.hub = hub;
    }

    @GetMapping("/api/events")
    public SseEmitter events() {
        return hub.subscribe();
    }
}
