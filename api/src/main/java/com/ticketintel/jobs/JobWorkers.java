package com.ticketintel.jobs;

import com.ticketintel.common.AppProperties;
import com.ticketintel.jobs.JobRepository.Job;
import com.ticketintel.mlclient.MlClient.LlmUnavailableException;
import com.ticketintel.mlclient.MlClient.MlBadRequestException;
import com.ticketintel.mlclient.MlClient.MlUnavailableException;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PreDestroy;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/** Starts N polling threads per job type. Adaptive polling: immediately re-poll after work, back off when idle. */
@Component
public class JobWorkers implements ApplicationRunner {
    private static final Logger log = LoggerFactory.getLogger(JobWorkers.class);

    private final JobRepository repo;
    private final List<JobHandler> handlers;
    private final AppProperties props;
    private final MeterRegistry metrics;
    private final List<ExecutorService> pools = new ArrayList<>();
    private volatile boolean running = true;
    private final String host;

    public JobWorkers(JobRepository repo, List<JobHandler> handlers, AppProperties props, MeterRegistry metrics) {
        this.repo = repo;
        this.handlers = handlers;
        this.props = props;
        this.metrics = metrics;
        String h;
        try { h = InetAddress.getLocalHost().getHostName(); } catch (Exception e) { h = "host"; }
        this.host = h;
    }

    @Override
    public void run(ApplicationArguments args) {
        if (!props.workers().enabled()) {
            log.info("job workers disabled (app.workers.enabled=false)");
            return;
        }
        for (JobHandler h : handlers) {
            ExecutorService pool = Executors.newFixedThreadPool(h.threads(), r -> {
                Thread t = new Thread(r);
                t.setDaemon(true);
                return t;
            });
            pools.add(pool);
            for (int i = 0; i < h.threads(); i++) {
                String name = host + "/" + h.type().toLowerCase() + "-" + i;
                pool.submit(() -> loop(h, name));
            }
            log.info("started {} worker(s) for {} (batch {})", h.threads(), h.type(), h.batchSize());
        }
    }

    private void loop(JobHandler h, String worker) {
        long idle = props.workers().pollMinMs();
        Timer timer = Timer.builder("jobs.processing").tag("type", h.type()).register(metrics);
        while (running) {
            try {
                List<Job> jobs = repo.claim(h.type(), h.batchSize(), worker);
                if (jobs.isEmpty()) {
                    Thread.sleep(idle);
                    idle = Math.min(props.workers().pollMaxMs(), idle * 2);
                    continue;
                }
                idle = props.workers().pollMinMs();
                timer.record(() -> process(h, jobs));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } catch (Exception e) {                       // DB hiccup etc.: never let the thread die
                log.error("{} loop error: {}", worker, e.toString());
                try { Thread.sleep(2000); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); return; }
            }
        }
    }

    void process(JobHandler h, List<Job> jobs) {
        var ctx = new JobHandler.Context();
        try {
            h.handle(jobs, ctx);
        } catch (MlUnavailableException | LlmUnavailableException e) {
            // dependency outage: not the job's fault -> defer without consuming an attempt
            for (Job j : jobs) repo.defer(j.id(), 30, e.getMessage());
            metrics.counter("jobs.deferred", "type", h.type()).increment(jobs.size());
            return;
        } catch (Exception e) {
            log.warn("{} batch of {} failed: {}", h.type(), jobs.size(), e.toString());
            boolean retry = !(e instanceof MlBadRequestException) && !(e instanceof IllegalArgumentException);
            for (Job j : jobs) {
                if (!ctx.deferred.containsKey(j.id()) && !ctx.failed.containsKey(j.id())) ctx.fail(j, e.toString(), retry);
            }
        }
        for (Job j : jobs) {
            if (ctx.deferred.containsKey(j.id())) {
                Object[] d = ctx.deferred.get(j.id());
                repo.defer(j.id(), (Integer) d[0], (String) d[1]);
                metrics.counter("jobs.deferred", "type", h.type()).increment();
            } else if (ctx.failed.containsKey(j.id())) {
                repo.fail(j, ctx.failed.get(j.id()), ctx.failedRetryable.get(j.id()));
                metrics.counter("jobs.failed", "type", h.type()).increment();
            } else {
                repo.complete(j.id());
                metrics.counter("jobs.done", "type", h.type()).increment();
            }
        }
    }

    @Scheduled(fixedDelay = 60_000)
    void reap() {
        if (!props.workers().enabled()) return;
        int n = repo.reap(props.workers().visibilityTimeoutMinutes());
        if (n > 0) log.warn("reaper returned {} stuck job(s) to PENDING", n);
    }

    @Scheduled(cron = "0 30 3 * * *")
    void cleanup() {
        repo.cleanup(7);
    }

    @PreDestroy
    void stop() {
        running = false;
        for (ExecutorService p : pools) {
            p.shutdownNow();
            try { p.awaitTermination(3, TimeUnit.SECONDS); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
        }
    }
}
