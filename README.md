<div align="center">

# LLM_Adapter

### Adaptive Multi-LLM Inference Gateway with Self-Learning Routing, Semantic Caching & Cost Optimization

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-000000?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF?style=flat-square&logo=meta&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20router-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?style=flat-square&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?style=flat-square&logo=grafana&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**A production-grade adaptive inference gateway that routes queries across multiple Ollama models (TinyLlama / Phi-3 / Mistral) using a 3-layer decision cascade — UCB1 bandit exploration, a self-learning DecisionTree classifier, and Pareto-optimized utility scoring — with FAISS-powered semantic caching, real-time quality feedback loops, GPU-aware routing, streaming inference, and full Prometheus + Grafana observability.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [API](#-api-reference) • [Roadmap](#-roadmap)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Request Lifecycle](#-request-lifecycle)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
  - [The Routing Brain](#the-routing-brain)
  - [Semantic Cache](#semantic-cache)
  - [Quality Feedback Loop](#quality-feedback-loop)
  - [Self-Learning Router](#self-learning-router)
  - [Bandit Exploration](#bandit-exploration)
  - [GPU-Aware Routing](#gpu-aware-routing)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Monitoring & Observability](#-monitoring--observability)
- [Performance & Benchmarks](#-performance--benchmarks)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Overview

**LLM_Adapter** is an **intelligent inference gateway** that solves a real-world problem: sending every query to the most powerful LLM is slow, expensive, and wasteful. Most queries don't need a 7B-parameter model — a 1.1B model handles them just fine. But a static rule like *"always use small for short queries"* breaks the moment query patterns shift.

This project builds a **self-improving system** that learns the optimal model for each query type while balancing latency, cost, system load, and response quality — mirroring the inference orchestration patterns used internally at companies like **Fireworks AI**, **Together AI**, **Anyscale**, and major cloud LLM providers.

### Why this exists

| Problem | Naive Approach | LLM_Adapter Approach |
|---------|---------------|----------------------|
| Cost explosion | Send all queries to GPT-4-class model | Route by complexity, save 60–70% |
| Latency on simple queries | Wait 30s for "What is 2+2?" | Tiny model answers in 1–2s |
| Cache misses on rephrased queries | Exact-match dict cache | FAISS semantic similarity (0.9 threshold) |
| Static routing rules | Hardcoded if/else | Self-learning DecisionTree retrained every 50 queries |
| No exploration | Always pick "best" model | UCB1 bandit forces periodic exploration |
| Repeated questions | Re-run inference every time | Sub-millisecond cache hits with cost-aware LRU |
| No quality signal | Trust the model blindly | Real-time semantic similarity + length scoring feedback |
| GPU memory pressure | Crash or OOM | Graceful tier downgrade based on VRAM/utilization |

### Who is this for

- **AI Infrastructure Engineers** — production-grade routing patterns
- **ML Platform Teams** — multi-model orchestration reference architecture
- **LLM Ops Engineers** — observability, caching, and feedback loop design
- **Researchers** — experimental ground for bandit algorithms and adaptive routing
- **Interview candidates** — a portfolio project demonstrating systems-level ML engineering

---

## Features

### Intelligent Routing
- **3-layer decision cascade** — UCB bandit → ML classifier → Pareto heuristic
- **Multi-objective optimization** — `Utility = Quality − α·Latency − β·Cost`
- **Load-adaptive penalty scaling** — `α = 0.15 + load²·0.8` for graceful degradation
- **Complexity-aware adjustments** — boost medium tier for reasoning, penalize for trivia
- **Safety latency guard** — automatic downgrade if latency exceeds threshold

### Self-Learning
- **DecisionTree classifier** trained on `router_logs.csv` (auto-retrains every 50 queries)
- **UCB1 bandit** with exploration coefficient `c = 1.5`
- **Thompson Sampling** as drop-in alternative
- **Auto-benchmarking** of all tiers every 100 queries (background thread)
- **EMA-updated model metrics** for latency, cost, quality

### Semantic Caching
- **FAISS IndexFlatIP** with 384-dim MiniLM embeddings
- **Adaptive similarity threshold** — auto-tunes between 0.80–0.95 by hit rate
- **Cost-aware LRU eviction** — preserves expensive responses longer
- **zlib compression** for cached payloads
- **TTL-based expiration** (default 1 hour)
- **Tier-aware keys** — prevents cross-tier contamination

### Quality Feedback Loop
- **Semantic similarity scoring** via cosine on MiniLM embeddings
- **Length adequacy scoring** — penalizes truncated/empty responses
- **Combined score** — `0.7·relevance + 0.3·length_score`
- **Bandit reward updates** — quality flows back to exploration policy
- **Model metric updates** — running quality average per tier

### Production Hardening
- **GPU-aware routing** via `nvidia-smi` (graceful CPU fallback)
- **Streaming inference** via chunked HTTP responses
- **Session-based conversation memory** with sliding window + summarization
- **Escalation logic** — small → medium when response is short on complex queries
- **Async FastAPI** with concurrent request handling
- **Prometheus metrics** — counters, histograms, gauges
- **Grafana dashboards** — pre-wired data sources
- **Docker Compose** deployment

### Frontend Dashboard
- **Real-time monitoring** — tier distribution, latency, quality, cost charts
- **Live chat interface** — toggle between normal and streaming modes
- **Session persistence** — multi-turn conversations
- **Bandit/Cache/GPU panels** — instant system insights

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                  CLIENT LAYER                                  │
│              ( Frontend Dashboard │ Swagger UI │ REST Clients )                │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │  HTTP / WebSocket
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI INFERENCE GATEWAY                             │
│                                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ FeatureExtractor │  │  LoadController  │  │   HybridMemory (sessions)    │ │
│  │ (token/word/type)│  │ (CPU + requests) │  │ (sliding window + summarize) │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────┬───────────────┘ │
│           │                     │                            │                 │
│           └──────────┬──────────┴────────────────────────────┘                 │
│                      ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                          ROUTER (3-layer cascade)                        │  │
│  │                                                                          │  │
│  │   ┌────────────────┐    ┌────────────────┐    ┌────────────────────┐   │  │
│  │   │  UCB1 Bandit   │ →  │  ML Classifier │ →  │ Pareto + Utility   │   │  │
│  │   │ (exploration)  │    │ (DecisionTree) │    │   (heuristic)      │   │  │
│  │   └────────────────┘    └────────────────┘    └────────────────────┘   │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │                   ┌────────────────────┐                                │  │
│  │                   │   GPU-Aware Cap    │                                │  │
│  │                   │  (VRAM-based)      │                                │  │
│  │                   └────────────────────┘                                │  │
│  └────────────────────────────────┬─────────────────────────────────────────┘  │
│                                   │                                            │
│                                   ▼  selected_tier                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         SEMANTIC CACHE (FAISS)                           │  │
│  │      MiniLM embeddings → IndexFlatIP → adaptive threshold (0.8–0.95)     │  │
│  │      Cost-aware LRU eviction · zlib compression · TTL expiration         │  │
│  └────────────────┬──────────────────────────────────────┬──────────────────┘  │
│      cache hit    │                                       │  cache miss        │
│                   ▼                                       ▼                    │
│            return cached                      ┌──────────────────────┐        │
│                                                │  Ollama Inference    │        │
│                                                │  TinyLlama / Phi-3   │        │
│                                                │      / Mistral        │        │
│                                                └──────────┬───────────┘        │
│                                                           │                    │
│                                                           ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                       FEEDBACK & LEARNING LAYER                          │  │
│  │                                                                          │  │
│  │  QualityEvaluator  →  Bandit reward  →  Metrics update (EMA)             │  │
│  │  Escalation check  →  Auto-benchmark (every 100q)  →  Retrain (every 50q)│  │
│  │  Training log → router_logs.csv → DecisionTree → trained_router.pkl      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                            OBSERVABILITY LAYER                                 │
│      Prometheus  →  Grafana  →  Frontend Dashboard (real-time polling)         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Receive query              POST /generate                        │
│ 2. Increment load counter     LoadController.increment_requests()   │
│ 3. Compute load score         load = (active/10 + cpu/100) / 2      │
│ 4. Extract features           {tokens, words, type, complexity}     │
│ 5. Route                      Bandit → ML → Heuristic → GPU cap     │
│ 6. Cache lookup               FAISS top-k search (k=1, sim ≥ 0.9)   │
│    └─ HIT  → return cached response                                 │
│ 7. Build prompt with memory   context + user query                  │
│ 8. Inference                  Ollama HTTP call (timeout 300s)       │
│ 9. Escalation check           small + complex + short → retry medium│
│ 10. Quality scoring           0.7·cosine + 0.3·length_score         │
│ 11. Update bandit reward      bandit.update(tier, quality)          │
│ 12. Update model metrics      EMA on latency / cost / quality       │
│ 13. Cache store               FAISS.add(embedding, response)        │
│ 14. Memory append             session[user] + session[assistant]    │
│ 15. Training log              CSV row: features + selected_tier     │
│ 16. Trigger retrain (mod 50)  DecisionTree.fit() + reload           │
│ 17. Trigger benchmark (mod 100) Run all tiers in background thread  │
│ 18. Prometheus metrics        Counter / Histogram / Gauge updates   │
│ 19. Decrement load counter    LoadController.decrement_requests()   │
│ 20. Return response           {response, tokens, model_used, cached}│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and accessible from PATH
- **Docker Desktop** *(optional — only for Prometheus + Grafana)*

### One-Command Setup

```bash
# Windows
start.bat

# Linux/Mac (manual equivalent below)
```

The `start.bat` script:
1. Starts Ollama server (skips if running)
2. Pulls `tinyllama`, `phi3`, `mistral` (skips if present)
3. Starts Prometheus + Grafana via `docker compose`
4. Boots FastAPI with `--reload`

### Manual Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Pull models (one-time)
ollama pull tinyllama
ollama pull phi3
ollama pull mistral

# 4. Start the gateway
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. (Optional) Start Prometheus + Grafana
docker compose up -d prometheus grafana
```

### Access Points

| Service | URL | Notes |
|---------|-----|-------|
| **Dashboard** | http://localhost:8000 | Frontend SPA |
| **Swagger UI** | http://localhost:8000/docs | Interactive API explorer |
| **Health** | http://localhost:8000/health | JSON status |
| **Metrics** | http://localhost:8000/metrics | Prometheus scrape target |
| **Prometheus** | http://localhost:9090 | Metric queries |
| **Grafana** | http://localhost:3000 | Default `admin/admin` |

### First Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of India?", "session_id": "test1"}'
```

```json
{
  "response": "The capital of India is New Delhi...",
  "tokens": 18,
  "model_used": "tiny",
  "cached": false,
  "load_score": 0.08
}
```

---

## How It Works

### The Routing Brain

The router is a **3-layer cascade** — each layer tries to make a decision; if it can't, control passes to the next layer.

```
                ┌──────────────────────────────────┐
                │     Query + features + load      │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │  Layer 1: UCB1 Bandit            │
                │                                  │
                │  if any tier tried < 3 times:    │
                │      return that tier (explore)  │
                │  else:                           │
                │      pass to next layer          │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │  Layer 2: ML Classifier          │
                │                                  │
                │  if trained_router.pkl exists:   │
                │      return DecisionTree.predict │
                │  else:                           │
                │      pass to next layer          │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │  Layer 3: Pareto + Utility       │
                │                                  │
                │  Apply complexity adjustments    │
                │  Compute Pareto front            │
                │  Score: Q - α·L - β·C            │
                │  return argmax(utility)          │
                └────────────────┬─────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │  Final: GPU-Aware Cap            │
                │                                  │
                │  max_tier = gpu_monitor.hint()   │
                │  return min(selected, max_tier)  │
                └──────────────────────────────────┘
```

#### Utility Function

```
Utility(tier) = Quality(tier) - α(load) · normalized_latency(tier)
                              - β(load) · normalized_cost(tier)

where:
    α(load) = 0.15 + load² · 0.8
    β(load) = 0.10 + load² · 0.5
    normalized_latency = latency / max(latencies)
    normalized_cost    = cost / max(costs)
```

#### Complexity-Aware Quality Adjustments

| Question Type | Detection | tiny | small | medium |
|--------------|-----------|------|-------|--------|
| Reasoning (why/how) | starts with `why`/`how` | 0 | +0.15 | **+0.30** |
| Descriptive | contains `explain`/`describe` | 0 | +0.10 | **+0.25** |
| Long factual (>8 words) | starts with `what`/`who`/`where`/`when` | 0 | +0.05 | +0.15 |
| Short factual (≤8 words) | same as above, short | **+0.20** | 0 | **−0.20** |
| Short other (≤6 words) | otherwise | **+0.15** | 0 | **−0.15** |

This prevents the router from sending *"What's 2+2?"* to Mistral while still using Mistral for *"Explain how transformers work."*

---

### Semantic Cache

```
Query: "what's the capital of india"
                    │
                    ▼
         ┌──────────────────────┐
         │  MiniLM-L6-v2        │
         │  encoder             │
         └──────────┬───────────┘
                    │  384-dim vector
                    ▼
         ┌──────────────────────┐
         │  L2 Normalize        │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  FAISS IndexFlatIP   │
         │  search(k=1)         │
         └──────────┬───────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
   sim ≥ 0.9                sim < 0.9
   AND same tier               OR different tier
          │                    │
          ▼                    ▼
   Return cached         Cache MISS
   response              → Run inference
   (zlib decompress)     → Add to FAISS
                         → Adapt threshold
```

#### Adaptive Threshold Logic

```python
total = hits + misses
if total >= 20:
    hit_rate = hits / total
    if hit_rate < 0.10:   threshold -= 0.01   # relax (allow more hits)
    elif hit_rate > 0.50: threshold += 0.01   # tighten (ensure quality)
    threshold = clamp(threshold, 0.80, 0.95)
```

#### Cost-Aware Eviction

When the cache is full, eviction prefers the **cheapest** entry (lowest tier cost). This means a cached Mistral response (cost=6) survives longer than a TinyLlama response (cost=1) — because it represents more saved compute.

---

### Quality Feedback Loop

Every response is scored and fed back into the system:

```
Response from model
        │
        ▼
┌────────────────────────────┐
│  QualityEvaluator          │
│                            │
│  relevance =               │
│   cosine(emb(q), emb(r))   │
│                            │
│  length_score:             │
│   words < 5  → 0.2         │
│   words < 20 → 0.6         │
│   words ≥ 20 → 1.0         │
│                            │
│  quality =                 │
│   0.7·relevance +          │
│   0.3·length_score         │
└─────────┬──────────────────┘
          │ quality ∈ [0, 1]
          │
   ┌──────┼─────────────────┐
   │      │                 │
   ▼      ▼                 ▼
Bandit   ModelMetrics    (logged)
update   update_quality
(reward) (EMA)
```

---

### Self-Learning Router

```
Every query →  router_logs.csv
              ┌──────────────────────────────────┐
              │ token_length,word_count,         │
              │ question_type,load_score,        │
              │ selected_tier                    │
              └──────────────────────────────────┘
                              │
                              │  (every 50 queries)
                              ▼
              ┌──────────────────────────────────┐
              │  RetrainingScheduler.on_query()  │
              │                                  │
              │  if query_count % 50 == 0:       │
              │      train()                     │
              │      router.reload_ml_model()    │
              └──────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │  train_router.py                 │
              │                                  │
              │  • Read CSV (min 20 rows)        │
              │  • Class weight balancing        │
              │  • DecisionTreeClassifier        │
              │    (max_depth=5,                 │
              │     min_samples_leaf=2)          │
              │  • Cross-val if ≥30 rows         │
              │  • joblib.dump → trained_router  │
              └──────────────────────────────────┘
```

#### Why DecisionTree?

| Choice | Reason |
|--------|--------|
| **DecisionTree** ✅ | Sub-microsecond inference, interpretable, works on small data |
| Neural network ❌ | Overkill for 4 features × 3 classes, GPU-hungry, opaque |
| Logistic regression ❌ | Cannot capture non-linear interactions (load × complexity) |
| Random forest ❌ | Marginal gains, slower, more memory |

---

### Bandit Exploration

#### UCB1 (default)

```
                                ln(N)
   UCB(tier) = μ(tier) + c · √─────────
                                n(tier)

where:
   μ(tier) = average reward for this tier
   N        = total selections across all tiers
   n(tier)  = number of times this tier was selected
   c        = 1.5 (exploration coefficient)

Tier with highest UCB wins.
Untried tiers (n=0) get forced exploration first 3 times.
```

#### Thompson Sampling (alternative)

```
Each tier maintains a Beta(α, β) distribution.
On selection: sample from each tier's Beta, pick highest.
On reward (quality q ∈ [0,1]):
   α_tier += q
   β_tier += (1 - q)

Naturally Bayesian. No manual exploration coefficient.
```

---

### GPU-Aware Routing

```
                ┌──────────────────────────────┐
                │  GPUMonitor (nvidia-smi)     │
                └──────────┬───────────────────┘
                           │
                           ▼
              ┌──────────────────────────────┐
              │  No GPU?                     │
              │  → return "medium"           │
              │  (Ollama runs all on CPU)    │
              └──────────────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────┐
              │  Refresh GPU stats           │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │  pressure > 0.85 OR          │
              │  utilization > 0.90          │
              │     → max tier = "tiny"      │
              │                              │
              │  pressure > 0.60 OR          │
              │  utilization > 0.70          │
              │     → max tier = "small"     │
              │                              │
              │  else                        │
              │     → max tier = "medium"    │
              └──────────────────────────────┘
```

VRAM thresholds:

| Tier | Model | Approx VRAM |
|------|-------|-------------|
| tiny | TinyLlama 1.1B | ~1.5 GB |
| small | Phi-3 3.8B | ~4.0 GB |
| medium | Mistral 7B | ~6.0 GB |

---

## Project Structure

```
LLM_Adapter/
├── api/
│   ├── main.py                  # FastAPI app — endpoints, request pipeline, lifecycle
│   ├── routes.py                # (placeholder for future route splitting)
│   └── dependencies.py          # (placeholder for DI)
├── models/
│   ├── base_model.py            # Abstract base class (generate / generate_stream)
│   ├── ollama_client.py         # Async HTTP client for Ollama (timeout 300s)
│   ├── model_registry.py        # Tier → OllamaClient mapping (Strategy Pattern)
│   ├── model_metrics.py         # Per-tier latency / quality / cost (EMA)
│   └── hf_client.py             # (placeholder for HuggingFace direct inference)
├── routing/
│   ├── router.py                # 3-layer cascade — Bandit → ML → Heuristic → GPU cap
│   ├── feature_extractor.py     # Query → structured features
│   ├── pareto_engine.py         # Pareto front computation
│   ├── bandit.py                # UCB1 + Thompson Sampling
│   ├── gpu_monitor.py           # nvidia-smi based GPU detection
│   ├── load_controller.py       # CPU + active request load score
│   ├── train_router.py          # DecisionTree training from CSV
│   ├── retraining.py            # Query-count-based auto retraining
│   └── trained_router.pkl       # Serialized ML model (auto-generated)
├── caching/
│   ├── semantic_cache.py        # FAISS + MiniLM with adaptive threshold
│   ├── cache_policy.py          # LRU + zlib + cost-aware eviction
│   └── cache_store.py           # (base cache primitives)
├── evaluation/
│   ├── quality_metrics.py       # Semantic similarity + length scoring
│   ├── auto_benchmark.py        # Background benchmarking every 100 queries
│   ├── benchmark_runner.py      # Manual benchmark CLI
│   ├── cost_estimator.py        # Token → cost
│   └── latency.py               # Latency utilities
├── memory/
│   ├── hybrid_memory.py         # Sliding window + token compression
│   ├── session_store.py         # In-memory session map
│   └── summarizer.py            # Text summarization helpers
├── monitoring/
│   ├── prometheus_metrics.py    # Counters / Histograms / Gauges
│   ├── prometheus.yml           # Scrape config
│   └── metrics_collector.py     # Helper utilities
├── frontend/
│   ├── index.html               # Full SPA dashboard
│   └── static/
├── training/
│   ├── router_logs.csv          # Training data (auto-appended per query)
│   └── synthetic_queries.json   # Query templates for bootstrapping
├── load_testing/
│   ├── load_generator.py        # Async concurrent load tester
│   └── scenarios.yaml           # Load test scenarios
├── core/
│   ├── config.py                # Pydantic settings
│   ├── constants.py             # Global constants
│   └── lifecycle.py             # App lifecycle hooks
├── utils/
│   ├── logger.py                # Structured logging
│   ├── helpers.py               # Decorators / utilities
│   └── token_counter.py         # Token estimation
├── preprocessing/               # (text / document / image preprocessing)
├── configs/                     # YAML configs (model / routing / memory)
├── Dockerfile                   # Slim Python 3.11 image
├── docker-compose.yml           # FastAPI + Prometheus + Grafana
├── requirements.txt             # Python dependencies
├── start.bat                    # One-click startup
├── stop.bat                     # One-click shutdown
├── BEGINNER_GUIDE.md            # Deep-dive learning guide
├── Implementation.md            # Technical progress report
└── README.md                    # You are here
```

---

## API Reference

### `POST /generate`

Standard inference endpoint. Returns the full response in a single JSON payload.

**Request body:**
```json
{
  "query": "Explain transformer attention",
  "session_id": "user-42"
}
```

**Response:**
```json
{
  "response": "Attention in transformers is...",
  "tokens": 187,
  "model_used": "medium",
  "cached": false,
  "load_score": 0.14
}
```

### `POST /generate/stream`

Streaming inference. Returns tokens as they are generated via chunked HTTP. Quality scoring, caching, and memory updates happen after the stream completes.

**Response:** `text/plain` stream of tokens.

### `GET /health`

```json
{ "app": "LLM_Adapter", "version": "0.1.0", "status": "running" }
```

### `GET /metrics`

Prometheus exposition format. Scraped by Prometheus every 15s.

### `GET /cache/stats`

```json
{
  "hits": 47,
  "misses": 132,
  "hit_rate": 0.2625,
  "threshold": 0.89,
  "cache_size": 132
}
```

### `GET /bandit/stats`

```json
{
  "tiny":   { "count": 45, "avg_reward": 0.4812 },
  "small":  { "count": 38, "avg_reward": 0.6291 },
  "medium": { "count": 51, "avg_reward": 0.8174 },
  "total": 134
}
```

### `GET /gpu`

```json
{
  "gpu_available": false,
  "gpu_info": {},
  "memory_pressure": 1.0,
  "gpu_utilization": 1.0,
  "routing_hint": "medium"
}
```

---

## Configuration

### Environment / Settings

`core/config.py` (Pydantic settings — override via environment variables):

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `DEFAULT_MODEL_TIER` | `tiny` | Fallback tier if all else fails |
| `BENCHMARK_MODE` | `True` | Enable auto-benchmarking |
| `BENCHMARK_SAMPLE_RATE` | `0.2` | Sampling rate for benchmark mode |

### Tunable Constants

| File | Constant | Default | Effect |
|------|----------|---------|--------|
| `routing/retraining.py` | `RETRAIN_EVERY_N` | `50` | Retrain interval (queries) |
| `evaluation/auto_benchmark.py` | `interval_queries` | `100` | Benchmark interval |
| `caching/semantic_cache.py` | `similarity_threshold` | `0.9` | Initial cache threshold |
| `caching/cache_policy.py` | `max_size` | `500` | Cache capacity |
| `caching/cache_policy.py` | `ttl_seconds` | `3600` | Cache TTL |
| `memory/hybrid_memory.py` | `turn_limit` | `8` | Conversation window size |
| `memory/hybrid_memory.py` | `token_limit` | `1000` | Compression threshold |
| `routing/bandit.py` | `exploration_weight` | `1.5` | UCB1 c parameter |
| `models/ollama_client.py` | `timeout` | `300.0` | Ollama HTTP timeout (s) |

### Model Tiers

`models/model_registry.py` — change models by editing one dict:

```python
{
    "tiny":   OllamaClient("tinyllama"),  # 1.1B
    "small":  OllamaClient("phi3"),       # 3.8B
    "medium": OllamaClient("mistral"),    # 7B
}
```

---

## Monitoring & Observability

### Prometheus Metrics

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `llm_requests_total` | Counter | — | Total inference requests |
| `llm_request_latency_seconds` | Histogram | — | Latency distribution |
| `llm_model_selected_total` | Counter | `tier` | Per-tier selection count |
| `llm_cache_hits_total` | Counter | — | Semantic cache hits |
| `llm_cache_miss_total` | Counter | — | Semantic cache misses |
| `llm_current_load_score` | Gauge | — | Current system load (0–1) |
| `llm_escalations_total` | Counter | — | Tier escalations triggered |

### Grafana Dashboards

Pre-wired data source connects to `http://prometheus:9090`. Suggested panels:
- Request rate (req/s)
- p50/p95/p99 latency
- Tier distribution pie
- Cache hit ratio
- Load score time series
- Escalation rate

---

## Performance & Benchmarks

> Reference numbers measured on a CPU-only environment (Intel i7, 16GB RAM, no GPU).

| Metric | TinyLlama | Phi-3 | Mistral |
|--------|-----------|-------|---------|
| Avg cold latency | ~6s | ~14s | ~32s |
| Avg warm latency | ~3s | ~8s | ~22s |
| Quality (semantic) | 0.45 | 0.65 | 0.85 |
| Cost (relative units) | 1 | 3 | 6 |
| VRAM (when GPU avail.) | 1.5 GB | 4 GB | 6 GB |

### Routing Effectiveness (qualitative)

After ~100 queries with feedback enabled:
- Simple factual queries → **>90%** routed to tiny
- Reasoning / explanatory queries → **>75%** routed to medium
- Cache hit rate stabilizes around **20–35%** for repeated/paraphrased queries
- Avg cost per query reduced **60–70%** vs naive *always-medium* baseline

---

## Roadmap

- [x] Multi-LLM adapter layer (Ollama)
- [x] Pareto-optimized heuristic router
- [x] Semantic cache (FAISS + MiniLM)
- [x] Conversation memory (sliding + summarization)
- [x] Prometheus + Grafana observability
- [x] Self-learning ML router (DecisionTree)
- [x] UCB1 + Thompson Sampling bandits
- [x] Quality feedback loop
- [x] Auto-benchmarking
- [x] GPU-aware routing
- [x] Streaming inference
- [x] Docker Compose deployment
- [x] Frontend dashboard
- [ ] RAG integration (document retrieval before routing)
- [ ] Persistent FAISS index (Milvus / Qdrant)
- [ ] Redis-backed session store
- [ ] Online learning router (contextual bandits)
- [ ] Multi-tenant API keys + rate limiting
- [ ] A/B testing framework for routing strategies
- [ ] OpenAI / Anthropic API adapters

---

## Contributing

Contributions welcome. To add a feature:

1. Fork the repo and create a feature branch
2. Run the test queries through `start.bat` and verify the dashboard works
3. Add observability hooks if introducing a new component
4. Ensure backward compatibility with the heuristic fallback path
5. Open a pull request with a clear description and reasoning

For larger changes — new routing strategies, alternative models, retrieval layers — open an issue first to discuss design.

---

## License

MIT License — free to use, modify, and distribute.

---

## Acknowledgments

- **[Ollama](https://ollama.com)** — local LLM runtime that makes this entire project possible
- **[FAISS](https://github.com/facebookresearch/faiss)** — Facebook's vector similarity library
- **[sentence-transformers](https://www.sbert.net/)** — MiniLM-L6-v2 embeddings
- **[FastAPI](https://fastapi.tiangolo.com/)** — async web framework
- **[Prometheus](https://prometheus.io/)** & **[Grafana](https://grafana.com/)** — observability stack
- Inference orchestration patterns inspired by **Fireworks AI**, **Together AI**, and **Anyscale**

---

<div align="center">

**Built with the goal of demonstrating real-world AI infrastructure engineering.**

If this project helped you, consider giving it a ⭐.

</div>
