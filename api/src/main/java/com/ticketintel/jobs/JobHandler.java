package com.ticketintel.jobs;

import com.ticketintel.jobs.JobRepository.Job;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A unit of background work. The worker marks every job DONE unless the handler failed/deferred it via the context. */
public interface JobHandler {
    String type();

    int batchSize();

    int threads();

    /** Must be idempotent: a job can be delivered more than once (crash after side effects, before completion). */
    void handle(List<Job> jobs, Context ctx) throws Exception;

    final class Context {
        final Map<Long, String> failed = new HashMap<>();
        final Map<Long, Boolean> failedRetryable = new HashMap<>();
        final Map<Long, Object[]> deferred = new HashMap<>();

        public void fail(Job j, String error, boolean retryable) {
            failed.put(j.id(), error);
            failedRetryable.put(j.id(), retryable);
        }

        public void defer(Job j, int seconds, String reason) {
            deferred.put(j.id(), new Object[] {seconds, reason});
        }
    }
}
