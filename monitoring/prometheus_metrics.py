from prometheus_client import Counter, Histogram, Gauge

# ---- Request Metrics ----
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of inference requests"
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Latency of inference requests",
    buckets=(5, 10, 20, 40, 60, 120, 180)
)

# ---- Model Selection ----
MODEL_SELECTION = Counter(
    "llm_model_selected_total",
    "Count of selected model tiers",
    ["tier"]
)

# ---- Cache ----
CACHE_HIT = Counter(
    "llm_cache_hits_total",
    "Number of cache hits"
)

CACHE_MISS = Counter(
    "llm_cache_miss_total",
    "Number of cache misses"
)

# ---- Load Score ----
LOAD_SCORE_GAUGE = Gauge(
    "llm_current_load_score",
    "Current computed load score"
)

# ---- Escalation ----
ESCALATION_COUNT = Counter(
    "llm_escalations_total",
    "Number of tier escalations"
)