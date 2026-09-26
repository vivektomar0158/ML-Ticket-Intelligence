# ADR 0004: JdbcTemplate instead of JPA; Spring Boot 4

**Decision.** All persistence is explicit SQL via `JdbcTemplate`. Spring Boot 4.1 (Initializr no longer offers 3.x).

**Why.** The hot paths are pgvector kNN, `SKIP LOCKED` claiming, advisory locks, partial indexes, `ON CONFLICT` idempotency and keyset pagination; an ORM adds nothing there and hides the queries that matter. Dropping JPA also removes the vector-type mapping problem.

**Consequences.** Result rows are maps, so the dashboard consumes snake_case column names; typed records are used at the ML boundary.
