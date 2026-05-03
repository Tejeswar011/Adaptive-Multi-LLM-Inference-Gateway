import asyncio
import httpx
import time
import pandas as pd
import random


API_URL = "http://127.0.0.1:8000/generate"

# Test query pool
QUERY_POOL = [

    # Deep Learning / Math Heavy
    "Derive the backpropagation equations for LSTM.",
    "Provide full mathematical formulation of GRU.",
    "Explain multi-head attention with equations.",
    "Compare Transformer encoder and decoder mathematically.",
    "Prove why vanishing gradient occurs in RNNs.",

    # Coding / Logic
    "Write Python code to implement a basic LSTM from scratch.",
    "Explain time complexity of quicksort and merge sort.",
    "Design a REST API for scalable AI inference system.",
    "How would you optimize a large-scale distributed system?",
    "Explain CAP theorem with practical examples.",

    # System Design
    "Design YouTube recommendation system.",
    "Design Uber's real-time location tracking system.",
    "How would you build a fault-tolerant microservice architecture?",
    "Explain horizontal vs vertical scaling in cloud systems.",

    # Physics / Math
    "Derive Maxwell's equations.",
    "Explain Einstein's field equations in general relativity.",
    "Prove the central limit theorem.",
    "Derive gradient descent update rule mathematically.",

    # Business / Finance
    "Explain how hedge funds use quantitative models.",
    "How does high frequency trading work?",
    "What is the Black-Scholes equation?",

    # Philosophy / Abstract
    "What is the difference between epistemology and ontology?",
    "Explain Gödel's incompleteness theorem.",
    "Discuss ethical implications of AGI.",

    # Biology
    "Explain CRISPR gene editing mechanism.",
    "Describe how neural synapses transmit signals.",

    # Very Short
    "What is AI?",
    "Define entropy.",
]


async def send_request(client, query, session_id):
    start = time.time()

    try:
        response = await client.post(
            API_URL,
            json={
                "query": query,
                "session_id": session_id
            },
            timeout=600  # 10 min — Ollama queues requests locally so 2nd request can wait long
        )
    except Exception as e:
        print(f"[TIMEOUT/ERROR] Query skipped: {query[:50]!r} | Reason: {e}")
        return {
            "query": query,
            "latency": time.time() - start,
            "tokens": 0,
            "model_used": "error",
            "cached": False,
            "load_score": None,
        }

    latency = time.time() - start

    try:
        data = response.json()
    except Exception:
        print(f"[ERROR] Status: {response.status_code} | Raw response: {response.text!r}")
        data = {}

    return {
        "query": query,
        "latency": latency,
        "tokens": data.get("tokens", 0),
        "model_used": data.get("model_used"),
        "cached": data.get("cached"),
        "load_score": data.get("load_score"),
    }


async def run_benchmark(total_requests=20, concurrency=5):
    results = []

    async with httpx.AsyncClient() as client:

        tasks = []

        for i in range(total_requests):
            query = random.choice(QUERY_POOL)
            session_id = f"bench_{i%3}"  # simulate few sessions

            task = send_request(client, query, session_id)
            tasks.append(task)

            if len(tasks) >= concurrency:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []

        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv("benchmark_results.csv", index=False)

    print("\n===== Benchmark Summary =====")
    print(df.groupby("model_used")["latency"].mean())
    print("\nCache Hit Rate:", df["cached"].mean())
    print("\nOverall Avg Latency:", df["latency"].mean())


if __name__ == "__main__":
    asyncio.run(run_benchmark(total_requests=50, concurrency=2))