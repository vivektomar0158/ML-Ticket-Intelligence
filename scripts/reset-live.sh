#!/usr/bin/env bash
# Dev helper: delete live (non-seed) tickets and everything hanging off them; keeps the seeded RAG history.
# usage: scripts/reset-live.sh [postgres-container]
C="${1:-ticketintelligence-postgres-1}"
docker exec "$C" psql -U ticketintel -d ticketintel -q \
  -c "DELETE FROM jobs" \
  -c "UPDATE duplicate_clusters SET canonical_ticket_id = NULL" \
  -c "UPDATE tickets SET cluster_id = NULL WHERE source <> 'SEED'" \
  -c "DELETE FROM feedback WHERE ticket_id IN (SELECT id FROM tickets WHERE source <> 'SEED')" \
  -c "DELETE FROM tickets WHERE source <> 'SEED'" \
  -c "DELETE FROM duplicate_clusters" \
  -c "DELETE FROM ingest_batches"
docker exec "$C" psql -U ticketintel -d ticketintel -tAc "SELECT 'live tickets left: ' || count(*) FROM tickets WHERE source <> 'SEED'"
