import asyncio
import httpx
import time
import random

API_URL = "http://127.0.0.1:8000/generate"

QUERIES = [
    "What is AI?",
    "Define entropy.",
    "Explain how neural networks learn.",
    "Why does gradient descent converge?",
    "Write a binary search in Python.",
    "Design a scalable microservice architecture.",
    "Derive the backpropagation equations for LSTM.",
    "What is the capital of Japan?",
    "Explain the CAP theorem with examples.",
    "How does high frequency trading work?",
]


async def send_request(client, query):
    start = time.time()
    try:
        resp = await client.post(
            API_URL,
            json={"query": query, "session_id": f"load_{random.randint(0, 5)}"},
            timeout=300,
        )
        data = resp.json()
        latency = time.time() - start
        print(f"[{latency:.1f}s] {data.get('model_used', '?'):6s} | {query[:50]}")
        return data
    except Exception as e:
        print(f"[ERROR] {e} | {query[:50]}")
        return None


async def run_load_test(total=30, concurrency=3):
    """Simulates concurrent load against the API."""
    print(f"Starting load test: {total} requests, concurrency={concurrency}\n")

    async with httpx.AsyncClient() as client:
        tasks = []
        results = []

        for i in range(total):
            query = random.choice(QUERIES)
            tasks.append(send_request(client, query))

            if len(tasks) >= concurrency:
                batch = await asyncio.gather(*tasks)
                results.extend([r for r in batch if r])
                tasks = []

        if tasks:
            batch = await asyncio.gather(*tasks)
            results.extend([r for r in batch if r])

    tiers = {}
    for r in results:
        t = r.get("model_used", "unknown")
        tiers[t] = tiers.get(t, 0) + 1

    print(f"\n===== Load Test Summary =====")
    print(f"Total requests: {len(results)}")
    print(f"Tier distribution: {tiers}")
    cached = sum(1 for r in results if r.get("cached"))
    print(f"Cache hits: {cached}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(run_load_test(total=30, concurrency=3))
