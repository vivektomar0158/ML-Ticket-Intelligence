#!/usr/bin/env bash
# Chaos checks against the RUNNING local stack (Git Bash on Windows; uses taskkill / netstat).
#   1. ML service dies: ingestion must stay fast + lossless; tickets get triaged after it comes back.
#   2. API is hard-killed mid-processing: no ticket lost, none double-processed after restart.
# usage: scripts/chaos.sh      (needs postgres, ml-service (port 8000) and api (port 8080) running; run scripts/reset-live.sh first)
set -u
cd "$(dirname "$0")/.."
PG=ticketintelligence-postgres-1
q() { docker exec $PG psql -U ticketintel -d ticketintel -tAc "$1"; }
ok=0; fail=0
check() { if [ "$2" = "1" ]; then echo "PASS  $1"; ok=$((ok+1)); else echo "FAIL  $1"; fail=$((fail+1)); fi; }
kill_port() { pid=$(netstat -ano | grep ":$1 .*LISTENING" | awk '{print $5}' | head -1); [ -n "$pid" ] && taskkill //PID "$pid" //F >/dev/null 2>&1; }
wait_http() { for _ in $(seq 1 60); do curl -s -m 2 "$1" | grep -q "$2" && return 0; sleep 2; done; return 1; }
start_ml()  { (cd ml-service && LLM_PROVIDER=mock uv run uvicorn app.main:app --port 8000 > "${TEMP:-/tmp}/ml_chaos.log" 2>&1 &); wait_http localhost:8000/health/ready '"ok"'; }
start_api() { (cd api && ./mvnw -q spring-boot:run > "${TEMP:-/tmp}/api_chaos.log" 2>&1 &); wait_http localhost:8080/actuator/health UP; }
seed() { uv run --project ml-service python scripts/seed.py replay --n "$1" --no-incidents --batch 10 --speed 0 2>&1 | tail -1; }

echo "== 1. ML service outage"
kill_port 8000; sleep 2
t0=$(date +%s%N); seed 40 >/dev/null; ms=$(( ($(date +%s%N) - t0) / 1000000 ))
n=$(q "select count(*) from tickets where source<>'SEED'")
check "ingestion accepts all tickets while ML is down ($n/40 stored, ${ms} ms total)" $([ "$n" = 40 ] && echo 1 || echo 0)
sleep 8
new=$(q "select count(*) from tickets where source<>'SEED' and status='NEW'")
check "tickets wait as NEW (nothing lost, nothing failed): $new NEW" $([ "$new" = 40 ] && echo 1 || echo 0)
dead=$(q "select count(*) from jobs where status='DEAD'")
check "outage does not kill jobs (DEAD=$dead)" $([ "$dead" = 0 ] && echo 1 || echo 0)
start_ml
q "update jobs set run_after = now() where status='PENDING'" >/dev/null
for _ in $(seq 1 60); do left=$(q "select count(*) from jobs where status in ('PENDING','RUNNING')"); [ "$left" = 0 ] && break; sleep 2; done
tri=$(q "select count(*) from tickets where source<>'SEED' and status='DRAFTED'")
check "all 40 tickets triaged + drafted after recovery ($tri/40)" $([ "$tri" = 40 ] && echo 1 || echo 0)

echo "== 2. API hard-kill mid-processing"
seed 150 >/dev/null; sleep 1
kill_port 8080; taskkill //F //IM java.exe >/dev/null 2>&1; sleep 2
stuck=$(q "select count(*) from jobs where status='RUNNING'")
pend=$(q "select count(*) from jobs where status='PENDING'")
echo "      at kill time: RUNNING=$stuck PENDING=$pend"
start_api
q "update jobs set locked_at = now() - interval '10 minutes' where status='RUNNING'" >/dev/null   # simulate the 5-min visibility timeout elapsing
for _ in $(seq 1 90); do left=$(q "select count(*) from jobs where status in ('PENDING','RUNNING')"); [ "$left" = 0 ] && break; sleep 2; done
total=$(q "select count(*) from tickets where source<>'SEED'")
dr=$(q "select count(*) from tickets where source<>'SEED' and status='DRAFTED'")
check "no ticket lost across the crash ($total/190 stored)" $([ "$total" = 190 ] && echo 1 || echo 0)
check "every ticket reached DRAFTED ($dr/190)" $([ "$dr" = 190 ] && echo 1 || echo 0)
dup_pred=$(q "select count(*) from (select ticket_id, task, model_name, model_version from predictions group by 1,2,3,4 having count(*)>1) x")
dup_draft=$(q "select count(*) from (select ticket_id from drafts where status='GENERATED' group by 1 having count(*)>1) x")
check "no duplicate predictions after redelivery ($dup_pred)" $([ "$dup_pred" = 0 ] && echo 1 || echo 0)
check "exactly one active draft per ticket ($dup_draft with >1)" $([ "$dup_draft" = 0 ] && echo 1 || echo 0)
dead=$(q "select count(*) from jobs where status='DEAD'")
check "no dead jobs ($dead)" $([ "$dead" = 0 ] && echo 1 || echo 0)

echo; echo "chaos: $ok passed, $fail failed"
[ "$fail" = 0 ]
