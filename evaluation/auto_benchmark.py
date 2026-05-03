import asyncio
import time
import threading
from models.model_registry import ModelRegistry
from evaluation.quality_metrics import QualityEvaluator

BENCHMARK_QUERIES = [
    ("What is 2 + 2?", 0),                          # simple factual
    ("Explain how neural networks learn.", 2),        # descriptive
    ("Why does gradient descent converge?", 1),       # reasoning
    ("Write a Python function for binary search.", 3), # coding
    ("What is the capital of France?", 0),             # simple factual
]


class AutoBenchmarker:

    def __init__(self, router, interval_queries=100):
        self.router = router
        self.interval_queries = interval_queries
        self.query_count = 0
        self.registry = ModelRegistry()
        self.evaluator = QualityEvaluator()

    def on_query(self):
        self.query_count += 1

        if self.query_count % self.interval_queries == 0:
            # Run benchmark in a background thread to not block requests
            thread = threading.Thread(target=self._run_benchmark, daemon=True)
            thread.start()

    def _run_benchmark(self):
        print(f"Auto-benchmark starting after {self.query_count} queries...")
        loop = asyncio.new_event_loop()

        try:
            results = loop.run_until_complete(self._benchmark_all_tiers())
            self._update_metrics(results)
            print("Auto-benchmark complete")
        except Exception as e:
            print(f"Auto-benchmark error: {e}")
        finally:
            loop.close()

    async def _benchmark_all_tiers(self):
        results = {}

        for tier in ["tiny", "small", "medium"]:
            model = self.registry.get(tier)
            tier_results = []

            for query, q_type in BENCHMARK_QUERIES:
                start = time.time()

                try:
                    result = await model.generate(query)
                    latency = time.time() - start
                    text = result.get("text", "")

                    quality = self.evaluator.score_response(query, text)

                    tier_results.append({
                        "latency": latency,
                        "quality": quality,
                    })
                except Exception as e:
                    print(f"Benchmark error for {tier}: {e}")
                    continue

            if tier_results:
                avg_latency = sum(r["latency"] for r in tier_results) / len(tier_results)
                avg_quality = sum(r["quality"] for r in tier_results) / len(tier_results)

                results[tier] = {
                    "latency": avg_latency,
                    "quality": avg_quality,
                }

                print(f"  {tier}: latency={avg_latency:.2f}s, quality={avg_quality:.4f}")

        return results

    def _update_metrics(self, results):
        for tier, data in results.items():
            self.router.metrics.update_latency(tier, data["latency"])
            self.router.metrics.update_quality(tier, data["quality"])

        print("Model metrics updated from benchmark")
