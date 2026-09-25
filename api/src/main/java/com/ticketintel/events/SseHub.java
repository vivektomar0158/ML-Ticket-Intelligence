package com.ticketintel.events;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/** Fan-out of domain events (ticket.created / triaged / drafted / resolved, incident.detected, sla.breached) to dashboards. */
@Component
public class SseHub {
    private static final Logger log = LoggerFactory.getLogger(SseHub.class);
    private final CopyOnWriteArrayList<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    public SseEmitter subscribe() {
        SseEmitter e = new SseEmitter(0L);
        emitters.add(e);
        e.onCompletion(() -> emitters.remove(e));
        e.onTimeout(() -> emitters.remove(e));
        e.onError(t -> emitters.remove(e));
        try {
            e.send(SseEmitter.event().name("hello").data(Map.of("ok", true)));
        } catch (IOException ex) {
            emitters.remove(e);
        }
        return e;
    }

    /** Publishes after the surrounding transaction commits (so clients never see state that gets rolled back). */
    public void publish(String type, Map<String, Object> data) {
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    send(type, data);
                }
            });
        } else {
            send(type, data);
        }
    }

    private void send(String type, Map<String, Object> data) {
        for (SseEmitter e : emitters) {
            try {
                e.send(SseEmitter.event().name(type).data(data));
            } catch (Exception ex) {
                emitters.remove(e);
            }
        }
    }

    @Scheduled(fixedDelay = 25_000)
    void heartbeat() {
        for (SseEmitter e : emitters) {
            try {
                e.send(SseEmitter.event().comment("keepalive"));
            } catch (Exception ex) {
                emitters.remove(e);
            }
        }
    }

    public int subscribers() {
        return emitters.size();
    }
}
