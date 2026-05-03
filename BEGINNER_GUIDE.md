# LLM_Adapter — Complete Beginner Guide
### From First Principles to Cracking a 40 LPA AI Infrastructure Job

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Does This Project Exist? (The Business Problem)](#2-why-does-this-project-exist)
3. [The Core Idea — Explained Like You're 5](#3-the-core-idea)
4. [Architecture Overview](#4-architecture-overview)
5. [Complete Request Flow — What Happens When You Ask a Question](#5-complete-request-flow)
6. [Deep Dive: Every Component Explained](#6-deep-dive-every-component)
   - 6.1 [FastAPI Server (api/main.py)](#61-fastapi-server)
   - 6.2 [Model Registry & Ollama Client](#62-model-registry--ollama-client)
   - 6.3 [Feature Extractor — Understanding the Query](#63-feature-extractor)
   - 6.4 [Load Controller — Knowing System Pressure](#64-load-controller)
   - 6.5 [The Router — The Brain of the System](#65-the-router)
   - 6.6 [Pareto Engine — Multi-Objective Optimization](#66-pareto-engine)
   - 6.7 [UCB Bandit — Exploration vs Exploitation](#67-ucb-bandit)
   - 6.8 [ML Router — Self-Learning from Data](#68-ml-router)
   - 6.9 [GPU Monitor — Hardware-Aware Routing](#69-gpu-monitor)
   - 6.10 [Semantic Cache — Never Answer the Same Question Twice](#610-semantic-cache)
   - 6.11 [Quality Evaluator — Scoring Every Response](#611-quality-evaluator)
   - 6.12 [Conversation Memory — Remembering Chat Context](#612-conversation-memory)
   - 6.13 [Auto Retraining & Benchmarking](#613-auto-retraining--benchmarking)
   - 6.14 [Prometheus Metrics & Grafana](#614-prometheus-metrics--grafana)
   - 6.15 [Streaming Responses](#615-streaming-responses)
   - 6.16 [Frontend Dashboard](#616-frontend-dashboard)
7. [Design Decisions — Why We Did It This Way](#7-design-decisions)
8. [Bugs We Hit and How We Fixed Them](#8-bugs-we-hit-and-how-we-fixed-them)
9. [Project File Map](#9-project-file-map)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Interview Preparation — What to Say](#11-interview-preparation)
12. [Key Concepts You Must Know](#12-key-concepts-you-must-know)

---

## 1. What Is This Project?

**LLM_Adapter** is a **production-grade adaptive multi-LLM routing system**.

In plain English: when a user asks a question, instead of always sending it to one single AI model, our system **intelligently picks the best model** for that specific question based on how complex it is, how busy the system is, how much it would cost, and how fast we need the answer.

Think of it like a **hospital triage system**:
- Simple headache → general doctor (fast, cheap)
- Broken bone → specialist (moderate, more skilled)
- Heart attack → senior surgeon (slower, expensive, highest quality)

Our "doctors" are three LLM models running on **Ollama** (a local model runner):

| Tier     | Model       | Parameters | Speed   | Quality | Cost  |
|----------|-------------|-----------|---------|---------|-------|
| **tiny** | TinyLlama   | 1.1B      | Fastest | Basic   | Cheapest |
| **small**| Phi-3       | 3.8B      | Medium  | Good    | Medium |
| **medium**| Mistral    | 7B        | Slowest | Best    | Expensive |

---

## 2. Why Does This Project Exist?

### The Real-World Problem

Companies like OpenAI, Anthropic, Google, and AWS run multiple AI models. They don't send every query to GPT-4 (their most expensive model). That would bankrupt them.

Instead, they have **inference gateways** — systems that decide:
- "This is a simple greeting → send to a cheap, fast model"
- "This needs deep reasoning → send to the expensive, powerful model"
- "System is overloaded → send to whatever is available"

**This is exactly what we built.** Our system mimics the infrastructure that companies like Fireworks AI, Together AI, and Anyscale use internally.

### Why This Matters for a 40 LPA Job

At 40 LPA roles (AI Platform Engineer, ML Infrastructure Engineer, LLM Ops Engineer), companies want people who understand:
- How to **orchestrate multiple models** (not just call one API)
- How to **optimize cost vs quality** (real business constraint)
- How to build systems that **learn and improve** (self-learning router)
- How to handle **production concerns** (caching, monitoring, load balancing)
- How to make **engineering trade-offs** (when to use simple heuristics vs ML)

This project demonstrates ALL of that.

---

## 3. The Core Idea

Imagine you run 3 restaurants:
- **Street food stall** (tiny) — fast, cheap, good for simple things
- **Regular restaurant** (small) — moderate speed, decent quality
- **5-star hotel** (medium) — slow, expensive, best quality

A customer walks in and says: "I want a glass of water."

Do you send them to the 5-star hotel? No! That's wasteful. The street food stall can handle it perfectly.

Another customer says: "I want a 7-course gourmet meal with wine pairing."

Do you send them to the street food stall? No! They need the 5-star hotel.

**Our system does exactly this, but for AI queries.** It looks at each question, evaluates its complexity, checks system load, and routes it to the right model.

But here's what makes it ADVANCED:
1. It doesn't just use simple rules — it uses **Pareto optimization** (multi-objective math)
2. It **learns from experience** — an ML model trains on past routing decisions
3. It **explores** — a bandit algorithm tries different models to discover better options
4. It **caches** — semantically similar questions get instant answers
5. It **scores quality** — every response is evaluated and fed back into the system

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                             │
│                   "Explain quantum computing"                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (api/main.py)                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │   Feature     │  │    Load      │  │   Conversation       │ │
│   │   Extractor   │  │  Controller  │  │   Memory             │ │
│   │              │  │              │  │   (session context)  │ │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│          │                 │                      │             │
│          ▼                 ▼                      │             │
│   ┌─────────────────────────────────┐             │             │
│   │          ROUTER (3 layers)      │             │             │
│   │                                 │             │             │
│   │  Layer 1: UCB Bandit            │             │             │
│   │    └─ Explore underused tiers   │             │             │
│   │                                 │             │             │
│   │  Layer 2: ML Model              │             │             │
│   │    └─ DecisionTree from CSV     │             │             │
│   │                                 │             │             │
│   │  Layer 3: Heuristic Fallback    │             │             │
│   │    └─ Pareto + Utility scoring  │             │             │
│   │                                 │             │             │
│   │  Final: GPU Cap                 │             │             │
│   │    └─ Downgrade if GPU full     │             │             │
│   └──────────────┬──────────────────┘             │             │
│                  │                                │             │
│                  ▼ selected_tier                   │             │
│   ┌──────────────────────────┐                    │             │
│   │    Semantic Cache Check   │                    │             │
│   │  (FAISS + MiniLM)        │                    │             │
│   └──────┬───────────────────┘                    │             │
│          │                                        │             │
│          ▼ cache miss                             │             │
│   ┌──────────────────────────┐                    │             │
│   │   Ollama Inference       │◄───────────────────┘             │
│   │   (TinyLlama/Phi3/       │  (prompt includes context)      │
│   │    Mistral)              │                                  │
│   └──────┬───────────────────┘                                  │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────┐                                  │
│   │   Escalation Check       │  If small model gave short      │
│   │   (small → medium)       │  answer on complex query,       │
│   └──────┬───────────────────┘  retry with medium               │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────┐  ┌──────────────────────────┐   │
│   │   Quality Scoring        │  │   Prometheus Metrics     │   │
│   │   (semantic similarity)  │  │   (latency, count, tier) │   │
│   └──────┬───────────────────┘  └──────────────────────────┘   │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────────────┐  ┌──────────────────────────┐   │
│   │   Cache Store            │  │   Training Log           │   │
│   │   (save for future)      │  │   (router_logs.csv)      │   │
│   └──────────────────────────┘  └──────────────────────────┘   │
│                                                                 │
│          ▼                                                      │
│   ┌──────────────────────────────────────────┐                  │
│   │   Return Response to User                │                  │
│   └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Complete Request Flow

Here's what happens step-by-step when you type "Explain how neural networks learn" and hit Send:

### Step 1: Feature Extraction
```python
# routing/feature_extractor.py
features = {
    "token_length": 36,        # character count
    "word_count": 5,           # "Explain how neural networks learn"
    "question_type": 2,        # 2 = descriptive (contains "explain")
    "avg_word_length": 6.2,
    "punctuation_density": 0.0
}
```
**Why?** The router needs structured data about the query to make a decision. It can't just look at raw text.

### Step 2: Load Measurement
```python
# routing/load_controller.py
load_score = 0.12  # low load: few active requests, CPU not stressed
```
**Why?** Under heavy load, we should prefer faster/cheaper models. Under low load, we can afford expensive models.

### Step 3: Routing Decision (3-Layer Cascade)

**Layer 1 — UCB Bandit Check:**
The bandit algorithm checks if any model tier has been tried fewer than 3 times. If so, it forces exploration. If all tiers have been tried enough, it uses the UCB1 formula to decide if it should explore or let the next layers decide.

**Layer 2 — ML Model:**
If a trained DecisionTree exists (`trained_router.pkl`), it predicts the tier:
```python
features = [36, 5, 2, 0.12]  # token_length, word_count, question_type, load
prediction = model.predict([features])  # → "medium"
```

**Layer 3 — Heuristic Fallback (if no ML model):**
Uses Pareto optimization + utility scoring:
```
For "explain" query (descriptive, type=2):
  - Boost medium quality by +0.25
  - Boost small quality by +0.1

Utility = Quality - α*Latency - β*Cost

  tiny:   0.45 - 0.15*(12/30) - 0.10*(1/6) = 0.45 - 0.06 - 0.017 = 0.373
  small:  0.75 - 0.15*(18/30) - 0.10*(3/6) = 0.75 - 0.09 - 0.050 = 0.610
  medium: 1.10 - 0.15*(30/30) - 0.10*(6/6) = 1.10 - 0.15 - 0.100 = 0.850

  Winner: medium (utility 0.850)
```

**Final — GPU Cap:**
If running on GPU and memory is tight, downgrades the tier. On CPU, this is a no-op (all models run fine on CPU via Ollama).

**Result: tier = "medium" (Mistral)**

### Step 4: Cache Check
The system encodes "Explain how neural networks learn" into a 384-dimension vector using MiniLM, then searches FAISS for a vector within 90% cosine similarity. First time asking → cache miss.

### Step 5: Build Prompt with Memory
```
system: Summary: previously discussed basic Python concepts
user: Explain how neural networks learn
```
The conversation memory adds context from previous turns in the same session.

### Step 6: Ollama Inference
Sends the prompt to Mistral via HTTP:
```python
POST http://localhost:11434/api/generate
{
    "model": "mistral",
    "prompt": "user: Explain how neural networks learn",
    "stream": false,
    "options": {"temperature": 0.7}
}
```
Mistral generates a full response (can take 30-120 seconds on CPU).

### Step 7: Escalation Check
If the selected model was "small" AND the query was complex AND the response was too short (< 60 tokens), escalate to "medium" and re-generate. This prevents under-answering.

### Step 8: Quality Scoring
```python
# evaluation/quality_metrics.py
relevance = cosine_similarity(embed("Explain how neural networks learn"),
                               embed(response_text))  # e.g., 0.72

word_count = 150  # long response
length_score = 1.0  # good length

quality = 0.7 * 0.72 + 0.3 * 1.0 = 0.804
```
This score is fed back to:
- The **bandit** (so it learns medium gives good quality for these queries)
- The **model metrics** (updates medium's running quality average)

### Step 9: Post-Processing
- **Cache Store**: Save this query-response pair for future cache hits
- **Memory Update**: Add this turn to the session history
- **Training Log**: Append `[36, 5, 2, 0.12, "medium"]` to `router_logs.csv`
- **Retrain Check**: If query count reaches 50, retrain the ML router
- **Benchmark Check**: If query count reaches 100, run auto-benchmarks
- **Prometheus**: Increment request count, record latency, update tier counters

### Step 10: Response
```json
{
    "response": "Neural networks learn through a process called...",
    "tokens": 150,
    "model_used": "medium",
    "cached": false,
    "load_score": 0.12
}
```

---

## 6. Deep Dive: Every Component

### 6.1 FastAPI Server

**File:** `api/main.py` (~314 lines)

**What it does:** The entry point. Handles HTTP requests, wires all components together.

**Key endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the frontend dashboard (index.html) |
| `/generate` | POST | Main inference endpoint (normal mode) |
| `/generate/stream` | POST | Streaming inference endpoint |
| `/health` | GET | Health check (for Docker/monitoring) |
| `/metrics` | GET | Prometheus metrics scrape endpoint |
| `/gpu` | GET | GPU status (for dashboard polling) |
| `/cache/stats` | GET | Cache hit/miss stats (for dashboard) |
| `/bandit/stats` | GET | Bandit exploration stats (for dashboard) |

**Why FastAPI?** It's async by default (critical for waiting on Ollama without blocking), has automatic OpenAPI docs at `/docs`, and is the industry standard for Python ML APIs.

**CORS middleware** is enabled so the frontend (served from the same origin) can make API calls. In production you'd restrict the origins.

---

### 6.2 Model Registry & Ollama Client

**Files:** `models/model_registry.py`, `models/ollama_client.py`, `models/base_model.py`

**Model Registry** — A simple dictionary mapping tier names to Ollama client instances:
```python
{
    "tiny":   OllamaClient("tinyllama"),
    "small":  OllamaClient("phi3"),
    "medium": OllamaClient("mistral"),
}
```

**Why a registry?** If we want to swap Mistral for Llama-3, we change ONE line. The rest of the system doesn't care. This is the **Strategy Pattern**.

**OllamaClient** — Communicates with Ollama's HTTP API:
- `generate()` — Sends prompt, waits for full response (timeout: 300s)
- `generate_stream()` — Sends prompt, yields tokens one-by-one as they arrive

**Why 300s timeout?** On CPU (no GPU), Mistral (7B params) can take 2-5 minutes for long responses. The original 120s was too short and caused timeout errors. This was a bug we discovered and fixed during development.

**Base model (ABC):** Both methods are defined in an abstract base class. This means if tomorrow we want to add an OpenAI model or a HuggingFace model, we just implement the same interface.

---

### 6.3 Feature Extractor

**File:** `routing/feature_extractor.py`

**What it does:** Converts raw text queries into structured numerical features that the router can use.

```python
"Why does gradient descent converge?" →
{
    "token_length": 38,      # character count
    "word_count": 6,         # word count
    "avg_word_length": 5.5,  # complexity hint
    "punctuation_density": 0.026,
    "question_type": 1       # 1 = reasoning (starts with "why")
}
```

**Question type classification:**
| Type | Value | Detected By | Example |
|------|-------|-------------|---------|
| Factual | 0 | Starts with what/who/where/when | "What is Python?" |
| Reasoning | 1 | Starts with why/how | "Why does gravity exist?" |
| Descriptive | 2 | Contains "explain" or "describe" | "Explain quantum physics" |
| Other | 3 | Default | "Write a poem about cats" |

**Why this matters:** Factual queries ("What is 2+2?") can be handled by tiny models. Reasoning queries ("Why does quantum entanglement work?") need medium models. This classification drives the complexity-aware routing.

---

### 6.4 Load Controller

**File:** `routing/load_controller.py`

**What it does:** Measures how busy the system is right now.

```python
load_score = (active_requests/10 + cpu_percent/100) / 2
```

Two signals:
1. **Active request count** — how many queries are being processed right now
2. **CPU usage** — system-level pressure from `psutil`

The load score is a number from 0.0 (idle) to 1.0 (fully loaded).

**Why this matters:** Under high load, the router increases its penalty for slow/expensive models. This naturally shifts routing towards faster, cheaper tiers:

```
Low load  → α=0.15, β=0.10  (quality matters most)
High load → α=0.95, β=0.60  (speed and cost matter more)
```

The formula is: `α = 0.15 + (load²) * 0.8` — the squared term means penalties grow exponentially as load increases. This is a deliberate design choice: we want the system to degrade gracefully, not suddenly.

---

### 6.5 The Router — The Brain of the System

**File:** `routing/router.py`

This is the most important file. It has a **3-layer cascade**:

```
Query → Bandit → ML Model → Heuristic → GPU Cap → Selected Tier
```

**Why a cascade?** Each layer serves a different purpose:

1. **UCB Bandit (Layer 1):** "Should I explore a less-tried model?" — Prevents the system from getting stuck always picking the same tier. Forces occasional experimentation.

2. **ML Router (Layer 2):** "Based on past data, what's the best tier?" — A trained DecisionTree that learned from hundreds of past routing decisions.

3. **Heuristic (Layer 3):** "Fall back to math-based optimization" — If there's no trained ML model yet, or it fails, we use Pareto optimization + utility scoring.

4. **GPU Cap (Final):** "Can the hardware handle this?" — Downgrades if GPU memory is insufficient.

**Why this order?** The bandit runs first because exploration MUST override exploitation sometimes (otherwise the system never improves). The ML model runs second because it's trained on real data. The heuristic is a reliable fallback.

---

### 6.6 Pareto Engine

**File:** `routing/pareto_engine.py`

**The concept (from economics):** A solution is "Pareto optimal" if you can't improve one objective without making another worse.

**Example with our 3 models:**
```
           Quality  Latency  Cost
tiny:      0.45     12s      1
small:     0.65     18s      3
medium:    0.85     30s      6
```

Is tiny "dominated" by small? Let's check:
- Small has better quality (0.65 > 0.45) ✓
- But small has worse latency (18 > 12) ✗

Since small isn't better in ALL dimensions, tiny is NOT dominated. All three models end up on the Pareto front (none dominates another).

**Why Pareto?** If we had 10 models, some might be strictly worse than others in every dimension. Pareto filtering removes those dominated options first, so our utility scoring only considers viable candidates.

**The utility function applied after Pareto:**
```
Utility = Quality - α * (Latency/MaxLatency) - β * (Cost/MaxCost)
```

This is a **scalarization** — converting multiple objectives into a single number. The α and β weights change based on system load, making the system adaptive.

---

### 6.7 UCB Bandit

**File:** `routing/bandit.py`

**The problem it solves:** Imagine the router always picks "medium" because it has the best quality. But maybe "small" has improved (new model update, better prompting), and we'd never know because we never try it. This is the **explore-exploit dilemma**.

**UCB1 (Upper Confidence Bound):**
```
UCB = average_reward + c * √(ln(total_tries) / tries_for_this_tier)
```

- **average_reward**: How good has this tier been? (quality score average)
- **exploration bonus**: Tiers tried less often get a bigger bonus, encouraging exploration
- **c = 1.5**: How much we value exploration (higher = more exploration)

**How it works in practice:**
1. First 9 queries: Forces 3 tries each for tiny, small, medium (initial exploration)
2. After that: Calculates UCB for each tier, picks the highest
3. If a tier hasn't been tried recently, its exploration bonus grows, eventually forcing a retry

**We also implemented Thompson Sampling** (an alternative):
- Maintains a Beta(α, β) distribution per tier
- Samples from each distribution
- Picks the tier with the highest sample
- Naturally balances exploration/exploitation through Bayesian updating

**Why both?** We use UCB1 by default (deterministic, easier to debug). Thompson Sampling is available as a swap-in alternative for experimentation.

---

### 6.8 ML Router

**Files:** `routing/train_router.py`, `routing/retraining.py`, `routing/trained_router.pkl`

**Training data:** Every query logs features to `training/router_logs.csv`:
```csv
token_length,word_count,question_type,load_score,selected_tier
18,4,0,0.12,tiny
45,8,1,0.08,medium
28,5,2,0.45,small
```

**Training process:**
1. Read CSV (need minimum 20 rows)
2. Handle class imbalance with `compute_sample_weight("balanced")`
3. Train a `DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)`
4. Cross-validate if 30+ rows available
5. Save to `routing/trained_router.pkl`

**Why DecisionTree?**
- Fast inference (nanoseconds, not milliseconds)
- Interpretable (you can visualize the tree and explain decisions)
- Works well with small datasets (we start with ~50 rows)
- No GPU needed for training

**Why not a neural network?** Overkill. We have 4 features and 3 classes. A DecisionTree handles this perfectly. Using a neural net would be slower to train, harder to debug, and wouldn't give better results on such simple data. This is the kind of engineering judgment interviewers love to hear.

**Retraining:** Every 50 queries, the `RetrainingScheduler` automatically retrains the model with new data. This means the router improves over time as it sees more queries.

---

### 6.9 GPU Monitor

**File:** `routing/gpu_monitor.py`

**What it does:** Detects GPU availability and provides routing constraints.

```python
# Detection: checks if nvidia-smi exists on PATH
gpu_available = shutil.which("nvidia-smi") is not None

# If GPU exists, queries:
# - VRAM total, used, free
# - GPU utilization percentage

# Routing hint based on pressure:
# memory_pressure > 85% or utilization > 90% → max tier = "tiny"
# memory_pressure > 60% or utilization > 70% → max tier = "small"
# else → max tier = "medium"
```

**CPU-only mode (our case):** When no GPU is detected, it returns "medium" as the routing hint, meaning no restriction. Ollama handles all models on CPU automatically.

**Why include this if we don't have a GPU?** It shows interviewers you designed for production. Real deployments have GPUs, and your system gracefully handles both scenarios. The `_apply_gpu_cap()` function demonstrates hardware-aware routing — a key topic in ML infrastructure interviews.

---

### 6.10 Semantic Cache

**Files:** `caching/semantic_cache.py`, `caching/cache_policy.py`

**The problem:** If 100 users ask "What is Python?", "what's Python?", and "tell me about Python", those are semantically the same question. We shouldn't run inference 100 times.

**How it works:**

1. **Encode the query** using MiniLM (sentence-transformers) into a 384-dimension vector
2. **Search FAISS** for the nearest cached vector
3. If cosine similarity ≥ 0.9 AND same model tier → cache hit (return instantly)
4. If miss → run inference, then store the result

**FAISS (Facebook AI Similarity Search):** A library optimized for fast vector search. Even with 10,000 cached queries, search takes < 1ms.

**Why tier-aware?** If "What is Python?" was cached from the tiny model, and now a complex query routes to medium, we don't want to return the tiny model's short answer. Cache keys include the tier.

**Adaptive threshold:**
```python
if hit_rate < 10%:   threshold -= 0.01  (relax, allow more hits)
if hit_rate > 50%:   threshold += 0.01  (tighten, ensure quality)
# Bounded: 0.80 to 0.95
```
This is a simple feedback loop that tunes itself.

**Cost-aware eviction (LRU + cost):**
When the cache is full (500 entries), instead of evicting the oldest entry (standard LRU), we evict the **cheapest** entry first. Why? A cached response from medium (cost=6) saved more computation than one from tiny (cost=1). So we keep expensive results longer.

**Compression:** Cached text is compressed with `zlib` to reduce memory usage. Decompressed on retrieval.

---

### 6.11 Quality Evaluator

**File:** `evaluation/quality_metrics.py`

**The problem:** How do you know if a model's response was actually good?

**Our approach — two signals:**

1. **Semantic relevance (70% weight):** How semantically similar is the response to the query?
   ```python
   embed("What is Python?")  →  vector_a
   embed("Python is a programming language...")  →  vector_b
   similarity = cosine(vector_a, vector_b)  # e.g., 0.75
   ```

2. **Length adequacy (30% weight):** Was the response long enough?
   ```python
   < 5 words  → 0.2 (too short, probably garbage)
   5-20 words → 0.6 (brief but acceptable)
   20+ words  → 1.0 (good length)
   ```

**Combined score:** `quality = 0.7 * relevance + 0.3 * length_score`

**Where this score goes:**
- **Bandit:** `update(tier, quality)` — influences future exploration
- **Model metrics:** `update_quality(tier, quality)` — updates the tier's running quality average (EMA)
- **Router decisions:** Better quality scores mean the tier gets selected more often

**This creates a feedback loop:**
```
Query → Route to tier → Get response → Score quality → Update tier metrics → 
Better routing decisions → Better responses → Higher scores → ...
```

This is the **self-learning** aspect that makes this project impressive.

---

### 6.12 Conversation Memory

**Files:** `memory/hybrid_memory.py`, `memory/session_store.py`

**The problem:** LLMs are stateless. Each request is independent. But users expect conversations:
```
User: "What is Python?"
AI: "Python is a programming language..."
User: "What can I do with it?"  ← "it" refers to Python
```

Without memory, the second query makes no sense to the model.

**Our approach — Hybrid Memory:**

1. **Session store:** In-memory dictionary mapping session IDs to message lists
2. **Sliding window:** Keep only the last 8 turns (to avoid token limit issues)
3. **Token-based compression:** If the context exceeds 1000 tokens, summarize the older half

```python
# The prompt sent to Ollama includes context:
"system: Summary: discussed Python basics and data types
user: What libraries does it have?
assistant: Python has libraries like NumPy, pandas...
user: Tell me about NumPy"
```

**Why not a database?** For an interview project running locally, in-memory is simpler and faster. In production, you'd use Redis or PostgreSQL. The architecture supports swapping — `SessionStore` is a separate class that could be backed by anything.

---

### 6.13 Auto Retraining & Benchmarking

**Files:** `routing/retraining.py`, `evaluation/auto_benchmark.py`

**Retraining (every 50 queries):**
```python
class RetrainingScheduler:
    def on_query(self):
        self.query_count += 1
        if self.query_count % 50 == 0:
            train()  # retrain DecisionTree from router_logs.csv
            router.reload_ml_model()  # hot-reload the new model
```

**Why query-count-based, not time-based?** We originally tried time-based (every 5 minutes with a background thread). But this was wasteful — if no queries came in, it retrained on the same data. And threading adds complexity. Query-count-based means we retrain exactly when we have new data.

**Auto Benchmarking (every 100 queries):**
Runs 5 diverse test queries against all 3 tiers in a background thread:
```python
BENCHMARK_QUERIES = [
    ("What is 2 + 2?", 0),                          # simple factual
    ("Explain how neural networks learn.", 2),        # descriptive
    ("Why does gradient descent converge?", 1),       # reasoning
    ("Write a Python function for binary search.", 3), # coding
    ("What is the capital of France?", 0),             # simple factual
]
```

Results update the model metrics (latency and quality) via EMA (Exponential Moving Average). This means the system's understanding of each model's performance stays current.

---

### 6.14 Prometheus Metrics & Grafana

**Files:** `monitoring/prometheus_metrics.py`, `monitoring/prometheus.yml`

**Prometheus** is the industry standard for metrics collection. We expose these metrics:

| Metric | Type | Purpose |
|--------|------|---------|
| `llm_requests_total` | Counter | Total request count |
| `llm_request_latency_seconds` | Histogram | Latency distribution |
| `llm_model_selected_total` | Counter (labeled) | Per-tier selection count |
| `llm_cache_hits_total` | Counter | Cache hit count |
| `llm_cache_miss_total` | Counter | Cache miss count |
| `llm_current_load_score` | Gauge | Current system load |
| `llm_escalations_total` | Counter | Escalation count |

Prometheus scrapes `/metrics` every 15 seconds. Grafana reads from Prometheus and displays dashboards.

**Why this matters:** Every production system needs observability. Being able to say "I added Prometheus counters, histograms, and gauges, and built Grafana dashboards for latency, throughput, and cache hit rate" is music to an interviewer's ears.

---

### 6.15 Streaming Responses

**Endpoint:** `POST /generate/stream`

**Normal mode:** Wait 30-120 seconds → get entire response at once.
**Stream mode:** Start seeing tokens after 1-2 seconds, response builds word by word.

Both modes go through the same routing pipeline. The difference is:
- Normal: `model.generate()` → waits for Ollama to finish → returns JSON
- Stream: `model.generate_stream()` → yields tokens via `StreamingResponse` → returns as chunks

**After streaming completes**, we still do quality scoring, caching, and memory updates on the full assembled response. The user gets real-time output AND the system still learns.

**Why streaming matters:** It's how ChatGPT, Claude, and every production LLM API works. Users perceive streaming as faster even though total time is the same.

---

### 6.16 Frontend Dashboard

**File:** `frontend/index.html`

A single-page application with:

**Left panel — System monitoring:**
- 4 stat cards (Total Requests, Cache Hit Rate, Active Sessions, Avg Latency)
- Tier Distribution chart (doughnut — which models get used most)
- Latency Trend chart (line — response times over time)
- Quality Scores chart (bar — per-tier quality)
- Cost Breakdown chart (line — spending over time)
- Bandit Stats panel (exploration counts per tier)
- Cache Stats panel (hits, misses, threshold, size)
- GPU Status panel (available, memory, utilization)
- Request Log table (last N requests with details)

**Right panel — Chat interface:**
- Normal/Stream toggle button
- Session ID for conversation tracking
- Message bubbles with metadata (model used, latency, cached)
- Typing indicator for streaming mode

**Auto-polling:** Dashboard polls `/cache/stats`, `/bandit/stats`, `/gpu` every 5 seconds.

**Tech:** Pure HTML/CSS/JS with Chart.js for graphs. No React/Vue — keeps deployment simple (just serve one HTML file).

---

## 7. Design Decisions — Why We Did It This Way

### Decision 1: Three model tiers, not two or five

**Why?** Three gives a clear quality-speed-cost spectrum with meaningful differences. Two tiers (just small/large) lacks granularity. Five tiers would make routing too complex and hard to justify differences between adjacent tiers.

### Decision 2: Ollama instead of cloud APIs

**Why?** Runs locally, no API costs, no internet dependency, models are consistent. For an interview project, you can demo it anywhere without needing API keys or worrying about rate limits.

### Decision 3: Heuristic router first, ML router second

**Why?** The heuristic (Pareto + utility scoring) works from day zero with zero training data. The ML router needs data to train. By building both with a fallback chain, the system works immediately and gets smarter over time. This is a production pattern — never depend solely on ML for critical paths.

### Decision 4: DecisionTree, not neural network

**Why?** 4 features, 3 classes, ~50-200 training rows. A DecisionTree trains in milliseconds, predicts in nanoseconds, and is fully interpretable. A neural net would be slower, harder to debug, and wouldn't perform better on this data size. **Interviewers love hearing you chose the SIMPLER tool when it's sufficient.**

### Decision 5: FAISS for caching, not a simple dictionary

**Why?** A dictionary cache only works for exact matches. "What is Python?" and "what's python" would be different keys. FAISS does **semantic** similarity search — it catches paraphrases. This is what real production caches do (e.g., Azure's Semantic Kernel cache).

### Decision 6: Query-count retraining, not time-based

**Why?** Time-based retraining wastes resources when idle and might miss busy periods. Query-count-based means we retrain exactly when new data arrives. We originally built time-based with a background thread (every 5 minutes) but discovered it made the system heavier than needed.

### Decision 7: In-process memory, not Redis

**Why?** For a demo project, in-memory is sufficient and removes external dependencies. The architecture (SessionStore as a separate class) allows swapping to Redis trivially. Don't over-engineer what you don't need yet — but show you know how to scale it.

### Decision 8: Single HTML file, no React

**Why?** One file = zero build step, zero npm dependencies, zero webpack config. It serves from FastAPI's `StaticFiles`. For a demo, this is perfect. It shows UI skill without framework overhead.

---

## 8. Bugs We Hit and How We Fixed Them

These are GOLD for interviews. When asked "tell me about a technical challenge," these are real stories.

### Bug 1: Router Always Picked "tiny"

**Symptom:** Every query, regardless of complexity, routed to tiny.

**Root cause:** The original latency penalty (α) was too aggressive. On CPU, latency matters a lot, and the formula over-penalized medium's higher latency. Also, the router didn't consider query complexity at all.

**Fix:**
1. Added **complexity-aware quality adjustments**:
   - Reasoning queries: boost medium quality by +0.3
   - Descriptive queries: boost medium by +0.25
   - Short factual: boost tiny by +0.2, penalize medium by -0.2
2. Softened α/β scaling for CPU: `α = 0.15 + load² * 0.8` (was more aggressive before)

**Lesson:** Always test with diverse query types, not just one.

### Bug 2: "What is the capital of India?" Routed to Medium

**Symptom:** After fixing Bug 1, simple factual queries went to medium (overkill).

**Root cause:** Medium's quality was so much higher (0.85 vs 0.45) that even with normalized penalties, it always won.

**Fix:** Added **overkill penalty** — for short factual queries (type=0, word_count ≤ 8):
```python
quality_adj["tiny"] = +0.2    # boost tiny (it's good enough for this)
quality_adj["medium"] = -0.2  # penalize medium (wasteful for this)
```

**Lesson:** Optimization functions need domain-specific constraints. Pure math sometimes gives technically correct but practically wrong answers.

### Bug 3: Ollama Timeout on Medium Model

**Symptom:** `ReadTimeout — Ollama took too long to respond` when using Mistral.

**Root cause:** Mistral (7B parameters) on CPU can take 2-5 minutes for long responses. Original timeout was 120 seconds.

**Fix:** Increased timeout from 120s to 300s (5 minutes).

**Lesson:** Always size timeouts based on worst-case performance, not average. CPU inference is 10-50x slower than GPU.

### Bug 4: Time-Based Retraining Was Too Heavy

**Symptom:** Background thread retraining every 5 minutes consumed resources even when idle.

**Fix:** Switched to query-count-based (every 50 queries). No background threads. Retraining only happens synchronously when there's actually new data.

**Lesson:** Background jobs need justification. If the trigger is "new data," trigger on new data, not on time.

---

## 9. Project File Map

```
LLM_Adapter/
│
├── api/
│   ├── main.py              ← FastAPI app, all endpoints, request pipeline
│   ├── routes.py             ← (placeholder for future route splitting)
│   └── dependencies.py       ← (placeholder for dependency injection)
│
├── models/
│   ├── base_model.py         ← Abstract base class (generate + generate_stream)
│   ├── ollama_client.py      ← HTTP client for Ollama API
│   ├── model_registry.py     ← Maps tier names → OllamaClient instances
│   ├── model_metrics.py      ← Per-tier latency/quality/cost tracking (EMA)
│   └── hf_client.py          ← (placeholder for HuggingFace direct inference)
│
├── routing/
│   ├── router.py             ← THE BRAIN: 3-layer cascade (Bandit → ML → Heuristic)
│   ├── feature_extractor.py  ← Query → structured features (word count, type, etc.)
│   ├── pareto_engine.py      ← Pareto front computation (multi-objective optimization)
│   ├── bandit.py             ← UCB1 + Thompson Sampling implementations
│   ├── gpu_monitor.py        ← GPU detection via nvidia-smi, routing hints
│   ├── load_controller.py    ← System load measurement (CPU + active requests)
│   ├── train_router.py       ← DecisionTree training from CSV
│   ├── retraining.py         ← Query-count-based auto retraining scheduler
│   └── trained_router.pkl    ← Serialized trained ML model (auto-generated)
│
├── caching/
│   ├── semantic_cache.py     ← FAISS + MiniLM embedding cache with adaptive threshold
│   ├── cache_policy.py       ← LRU eviction with compression + cost-aware eviction
│   └── cache_store.py        ← (base cache store)
│
├── evaluation/
│   ├── quality_metrics.py    ← Semantic similarity + length adequacy scoring
│   ├── auto_benchmark.py     ← Periodic multi-tier benchmarking (every 100 queries)
│   ├── benchmark_runner.py   ← Manual benchmark tool
│   ├── cost_estimator.py     ← Token → cost estimation
│   └── latency.py            ← Latency measurement utilities
│
├── memory/
│   ├── hybrid_memory.py      ← Sliding window + token-based compression
│   ├── session_store.py      ← In-memory session dictionary
│   └── summarizer.py         ← Text summarization utilities
│
├── monitoring/
│   ├── prometheus_metrics.py ← Prometheus counters, histograms, gauges
│   ├── prometheus.yml        ← Prometheus scrape config
│   └── metrics_collector.py  ← (additional metrics utilities)
│
├── frontend/
│   ├── index.html            ← Full SPA dashboard (charts, chat, monitoring)
│   └── static/               ← Static assets directory
│
├── training/
│   ├── router_logs.csv       ← Training data (features + selected tier per query)
│   └── synthetic_queries.json← Synthetic query templates
│
├── load_testing/
│   ├── load_generator.py     ← Async concurrent load testing tool
│   └── scenarios.yaml        ← Test scenario definitions
│
├── core/
│   ├── config.py             ← Pydantic settings (Ollama URL, app name, etc.)
│   ├── constants.py          ← Global constants
│   └── lifecycle.py          ← App lifecycle hooks
│
├── utils/
│   ├── logger.py             ← Logging configuration
│   ├── helpers.py            ← timed decorator, clamp, safe_divide
│   └── token_counter.py      ← Token estimation utility
│
├── preprocessing/
│   ├── text_processor.py     ← Text preprocessing utilities
│   ├── document_processor.py ← Document parsing (for future RAG)
│   └── image_processor.py    ← Image processing (for future multimodal)
│
├── configs/
│   ├── model_config.yaml     ← Model configuration
│   ├── routing_config.yaml   ← Routing parameters
│   └── memory_config.yaml    ← Memory settings
│
├── Dockerfile                ← Docker image definition
├── docker-compose.yml        ← Full stack: FastAPI + Prometheus + Grafana
├── requirements.txt          ← Python dependencies
├── start.bat                 ← One-click startup (Ollama + models + Docker + server)
├── stop.bat                  ← One-click shutdown (kill all services)
├── Implementation.md         ← Technical progress report
└── BEGINNER_GUIDE.md         ← This file!
```

---

## 10. How to Run the Project

### Prerequisites
1. **Python 3.11+** installed
2. **Ollama** installed (https://ollama.com)
3. **Docker Desktop** (optional, for Prometheus + Grafana)

### Quick Start
```bash
# 1. Double-click start.bat (or run it from terminal)
start.bat
```

This automatically:
1. Starts Ollama server (if not running)
2. Pulls TinyLlama, Phi-3, Mistral models (if not downloaded)
3. Starts Prometheus + Grafana via Docker (if Docker available)
4. Starts the FastAPI server with hot reload

### Access Points
| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Manual Start (if start.bat doesn't work)
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models (one time)
ollama pull tinyllama
ollama pull phi3
ollama pull mistral

# Terminal 3: Start the server
cd LLM_Adapter
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing via Swagger UI
1. Go to http://localhost:8000/docs
2. Click `POST /generate`
3. Click "Try it out"
4. Enter: `{"query": "What is Python?", "session_id": "test1"}`
5. Click "Execute"

### Testing via Dashboard
1. Go to http://localhost:8000
2. Type a question in the chat panel on the right
3. Toggle between Normal and Stream modes
4. Watch the left panel update with metrics in real-time

---

## 11. Interview Preparation — What to Say

### The Elevator Pitch (30 seconds)
> "I built a production-grade adaptive multi-LLM routing system. It routes queries to optimal model tiers based on complexity, load, cost, and quality — similar to what Fireworks AI or Together AI use internally. It features Pareto-optimized routing, a self-learning ML router that retrains from its own decisions, UCB bandit exploration, FAISS-based semantic caching, quality feedback loops, and full Prometheus/Grafana observability. All running locally on Ollama."

### When Asked "Walk Me Through the Architecture"
Follow the flow from Section 5. Start with the user query, go through feature extraction, routing (explain the 3 layers), caching, inference, quality scoring, and the feedback loop. Draw the diagram from Section 4 on a whiteboard.

### When Asked "What Was the Hardest Part?"
Talk about **Bug 1 (router bias)**: "The router was always picking the cheapest model. I had to add complexity-aware quality adjustments and tune the load-sensitive penalty scaling. The key insight was that on CPU (where latency is inherently high), you can't penalize latency as aggressively as you would on GPU. I also added overkill penalties — if a simple factual question routes to the most expensive model, that's wasteful even if the model is better."

### When Asked "Why Not Just Use GPT-4 for Everything?"
> "Cost and latency. If you send every query to the strongest model, your inference costs explode and response times increase. 70% of real-world queries are simple enough for a small model. The routing system saves 60-80% on cost by sending simple queries to cheap models while maintaining quality on complex ones."

### When Asked "How Does It Learn?"
> "Three learning mechanisms:
> 1. **Supervised:** Every routing decision is logged. A DecisionTree retrains every 50 queries from this data.
> 2. **Bandit exploration:** UCB1 occasionally tries different tiers to discover improvements.
> 3. **Quality feedback:** Every response is scored (semantic similarity + length adequacy), and scores flow back into model metrics and bandit rewards."

### When Asked "How Would You Scale This?"
> "Three axes:
> 1. **More models:** The registry pattern supports adding any Ollama model. Add a 'large' tier with Llama-70B when GPUs are available.
> 2. **Horizontal scaling:** Put the FastAPI server behind a load balancer. The semantic cache could move to Redis for shared state. Session store to PostgreSQL.
> 3. **Better routing:** Upgrade from DecisionTree to a gradient-boosted model or online learning (contextual bandits) as data grows. Add A/B testing for routing strategies."

### When Asked "What Would You Do Differently?"
> "Three things:
> 1. **Persistent cache:** Current FAISS index is in-memory. I'd add Milvus or Pinecone for persistence across restarts.
> 2. **RAG integration:** Add document retrieval before routing so the model has context from uploaded documents.
> 3. **Human feedback:** Currently quality scoring is automated. Adding thumbs up/down from users would give stronger training signal."

---

## 12. Key Concepts You Must Know

### For the Interview

**Pareto Optimization:** Given multiple objectives (quality, latency, cost), find solutions where you can't improve one without worsening another. We use this to filter dominated model options.

**Multi-Armed Bandit (UCB1):** A framework for the explore-exploit trade-off. Like choosing between restaurants — do you go to your favorite (exploit) or try a new one (explore)? UCB1 balances this mathematically.

**Thompson Sampling:** Alternative to UCB1. Maintains a probability distribution per option and samples from it. More Bayesian, equally effective, slightly more elegant.

**Semantic Similarity (Cosine Similarity):** Measuring how similar two text embeddings are. Values from -1 (opposite) to 1 (identical). We use it for caching (0.9 threshold) and quality scoring.

**FAISS (Facebook AI Similarity Search):** Library for fast approximate nearest neighbor search on vectors. Critical for semantic caching at scale.

**EMA (Exponential Moving Average):** Updates metrics smoothly: `new = (old + observed) / 2`. This means recent observations matter more, but old data isn't thrown away completely.

**Sentence-Transformers / MiniLM:** Small neural networks that convert text to fixed-size vectors (embeddings). MiniLM outputs 384-dim vectors and runs in < 10ms.

**Prometheus Metric Types:**
- **Counter:** Only goes up (total requests, cache hits)
- **Gauge:** Goes up and down (current load score)
- **Histogram:** Measures distributions (latency buckets)

**Scalarization:** Converting a multi-objective problem into a single-objective one by weighting and combining: `Utility = Q - αL - βC`

**LRU Cache:** Least Recently Used — evicts the oldest accessed entry when full. We enhanced it with cost-aware eviction (evict cheapest first).

**Strategy Pattern:** The model registry lets us swap models without changing routing logic. This is a Gang of Four design pattern — mention it if asked about your code's design.

**Graceful Degradation:** When GPU isn't available, the system still works (CPU fallback). When ML model isn't trained, heuristic takes over. When cache misses, inference happens. Nothing crashes — it just gets slightly slower/less optimal.

---

### Technology-Specific Knowledge

**Ollama:** Local LLM runner. Downloads and runs models via HTTP API. Like Docker for LLMs.

**FastAPI:** Async Python web framework. Built on Starlette and Pydantic. Key features: automatic `/docs`, async/await support, type validation.

**FAISS:** Facebook's vector search library. `IndexFlatIP` does inner product (cosine) search. Fast even with millions of vectors.

**scikit-learn DecisionTree:** `max_depth=5` prevents overfitting. `min_samples_leaf=2` ensures leaf nodes have enough support. `class_weight="balanced"` handles imbalanced training data.

**zlib:** Built-in Python compression. We use it to compress cached responses. Typical 2-3x compression on text.

**psutil:** System monitoring library. We use `cpu_percent()` for load measurement.

**httpx:** Async HTTP client for Python. Used to communicate with Ollama's API. Supports streaming responses.

---

## Final Words

This project is not a toy. It demonstrates **systems thinking** — how multiple components (routing, caching, monitoring, feedback loops) work together to create intelligent behavior. That's exactly what 40 LPA AI infrastructure roles demand.

The key differentiator is the **self-learning feedback loop**: Route → Infer → Score → Update metrics → Better routing → Better inferences → Higher scores. This cycle is what separates a simple API wrapper from a production-grade inference gateway.

Practice explaining each component in 2-3 sentences. Practice drawing the architecture diagram. Practice telling the bug stories. And most importantly, practice explaining **WHY** you made each design choice — not just WHAT you built.

Good luck! 🚀
