PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS blobs (
  blob_hash   TEXT PRIMARY KEY,
  compression TEXT NOT NULL DEFAULT 'zstd',
  byte_size   INTEGER NOT NULL,
  payload     BLOB NOT NULL,
  created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id         TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  parent_run_id  TEXT REFERENCES runs(run_id),
  experiment_id  TEXT REFERENCES experiments(experiment_id),
  variant_name   TEXT,
  forked_at_step INTEGER,
  mode           TEXT NOT NULL,
  run_seed       INTEGER NOT NULL,
  git_commit_sha TEXT,
  status         TEXT NOT NULL DEFAULT 'RUNNING',
  total_tokens   INTEGER NOT NULL DEFAULT 0,
  total_cost_usd REAL NOT NULL DEFAULT 0,
  total_latency_ms REAL NOT NULL DEFAULT 0,
  divergence_step INTEGER,
  divergence_diff_hash TEXT REFERENCES blobs(blob_hash),
  created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
  experiment_id   TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  parent_run_id   TEXT NOT NULL REFERENCES runs(run_id),
  fork_step_index INTEGER NOT NULL,
  variants_hash   TEXT NOT NULL REFERENCES blobs(blob_hash),
  checks_hash     TEXT REFERENCES blobs(blob_hash),
  created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
  step_id        TEXT PRIMARY KEY,
  run_id         TEXT NOT NULL REFERENCES runs(run_id),
  step_index     INTEGER NOT NULL,
  step_type      TEXT NOT NULL,
  name           TEXT NOT NULL,
  occurrence     INTEGER NOT NULL,
  match_key      TEXT NOT NULL,
  input_hash     TEXT NOT NULL REFERENCES blobs(blob_hash),
  output_hash    TEXT REFERENCES blobs(blob_hash),
  pre_state_hash  TEXT REFERENCES blobs(blob_hash),
  post_state_hash TEXT REFERENCES blobs(blob_hash),
  frozen_timestamp TEXT NOT NULL,
  rng_seed       INTEGER NOT NULL,
  tool_mode      TEXT,
  source         TEXT NOT NULL,
  error          TEXT,
  latency_ms     REAL NOT NULL,
  UNIQUE(run_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_steps_lookup ON steps(run_id, name, occurrence);

CREATE TABLE IF NOT EXISTS verdicts (
  verdict_id   TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES runs(run_id),
  compared_run_id TEXT,
  step_index   INTEGER,
  tier         TEXT NOT NULL,
  metric_name  TEXT NOT NULL,
  passed       INTEGER NOT NULL,
  score        REAL,
  position_consistent INTEGER,
  reasoning_json TEXT NOT NULL,
  judge_model  TEXT,
  created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS judge_cache (
  cache_key    TEXT PRIMARY KEY,
  model        TEXT NOT NULL,
  prompt_hash  TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at   REAL NOT NULL
);
