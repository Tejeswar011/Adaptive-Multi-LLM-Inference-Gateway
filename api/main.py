from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from utils.logger import get_logger
from models.model_registry import ModelRegistry
import time
from routing.feature_extractor import FeatureExtractor

from routing.pareto_engine import ParetoEngine

from routing.router import Router
from routing.retraining import RetrainingScheduler

from evaluation.cost_estimator import estimate_cost
from evaluation.auto_benchmark import AutoBenchmarker

from routing.load_controller import LoadController

from evaluation.quality_metrics import QualityEvaluator
import random

from caching.semantic_cache import SemanticCache

from memory.hybrid_memory import HybridMemory

from prometheus_client import generate_latest
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from monitoring.prometheus_metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    MODEL_SELECTION,
    CACHE_HIT,
    CACHE_MISS,
    LOAD_SCORE_GAUGE,
    ESCALATION_COUNT
)

import csv
import os

log_path = "training/router_logs.csv"
file_exists = os.path.isfile(log_path)

memory = HybridMemory()

semantic_cache = SemanticCache()

quality_evaluator = QualityEvaluator()

load_controller = LoadController()

router = Router()

pareto_engine = ParetoEngine()

extractor = FeatureExtractor()

registry = ModelRegistry()

logger = get_logger("LLM_Adapter")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retrainer = RetrainingScheduler(router)
benchmarker = AutoBenchmarker(router)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("LLM_Adapter is starting...")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running"
    }


from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/gpu")
def gpu_status():
    return router.gpu_monitor.get_status()

@app.get("/cache/stats")
def cache_stats():
    return semantic_cache.get_stats()

@app.get("/bandit/stats")
def bandit_stats():
    return router.bandit.get_stats()

@app.post("/generate")
async def generate_text(request: QueryRequest):

    import csv
    import os
    import time

    REQUEST_COUNT.inc()
    load_controller.increment_requests()

    current_load = load_controller.compute_load_score()
    LOAD_SCORE_GAUGE.set(current_load)

    logger.info(f"Current Load Score: {current_load:.3f}")

    try:
        # ---- Feature extraction (needed for logging + future ML router)
        features = extractor.extract(request.query)

        # ---- Routing (ML model with heuristic fallback)
        selected_tier = router.select_model(
            current_load,
            question_type=features["question_type"],
            word_count=features["word_count"],
            token_length=features["token_length"]
        )
        MODEL_SELECTION.labels(tier=selected_tier).inc()

        model = registry.get(selected_tier)

        # ---- Semantic Cache ----
        cached_response = semantic_cache.search(request.query, selected_tier)

        if cached_response is not None:
            logger.info("Cache HIT")
            CACHE_HIT.inc()

            return {
                "response": cached_response,
                "tokens": 0,
                "model_used": selected_tier,
                "cached": True,
                "load_score": current_load
            }

        CACHE_MISS.inc()

        # ---- Conversation Context ----
        context = memory.get_context(request.session_id)

        context_text = ""
        for msg in context:
            context_text += f"{msg['role']}: {msg['content']}\n"

        final_prompt = context_text + f"user: {request.query}"

        # ---- Inference ----
        start=time.time()
        with REQUEST_LATENCY.time():
            result = await model.generate(final_prompt)

        latency= time.time() - start

        # ---- Escalation ----
        complex_query = len(request.query.split()) > 5

        if selected_tier == "small" and complex_query and result["tokens"] < 60:
            logger.info("Escalating to medium")
            ESCALATION_COUNT.inc()

            medium_model = registry.get("medium")

            with REQUEST_LATENCY.time():
                result = await medium_model.generate(final_prompt)

            selected_tier = "medium"
            MODEL_SELECTION.labels(tier="medium").inc()

        cost = estimate_cost(result["tokens"])

        # ---- Quality Feedback Loop ----
        quality_score = quality_evaluator.score_response(request.query, result["text"])
        router.metrics.update_quality(selected_tier, quality_score)
        router.update_bandit(selected_tier, quality_score)
        logger.info(f"Quality score for {selected_tier}: {quality_score:.4f}")

        router.metrics.update_latency(selected_tier, latency)
        router.metrics.update_cost(selected_tier, cost)

        semantic_cache.add(request.query, selected_tier, result["text"])

        # ---- Update Memory ----
        memory.add_turn(request.session_id, "user", request.query)
        memory.add_turn(request.session_id, "assistant", result["text"])

        # ============================
        #  TRAINING DATA LOG BLOCK
        # ============================
        log_path = "training/router_logs.csv"
        file_exists = os.path.isfile(log_path)

        os.makedirs("training", exist_ok=True)

        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow([
                    "token_length",
                    "word_count",
                    "question_type",
                    "load_score",
                    "selected_tier"
                ])

            writer.writerow([
                features["token_length"],
                features["word_count"],
                features["question_type"],
                current_load,
                selected_tier
            ])

        # ---- Retrain & benchmark check ----
        retrainer.on_query()
        benchmarker.on_query()

        # ---- Final Response ----
        return {
            "response": result["text"],
            "tokens": result["tokens"],
            "model_used": selected_tier,
            "cached": False,
            "load_score": current_load
        }

    finally:
        load_controller.decrement_requests()


@app.post("/generate/stream")
async def generate_text_stream(request: QueryRequest):

    REQUEST_COUNT.inc()
    load_controller.increment_requests()

    current_load = load_controller.compute_load_score()
    LOAD_SCORE_GAUGE.set(current_load)

    features = extractor.extract(request.query)

    selected_tier = router.select_model(
        current_load,
        question_type=features["question_type"],
        word_count=features["word_count"],
        token_length=features["token_length"]
    )
    MODEL_SELECTION.labels(tier=selected_tier).inc()

    model = registry.get(selected_tier)

    context = memory.get_context(request.session_id)
    context_text = ""
    for msg in context:
        context_text += f"{msg['role']}: {msg['content']}\n"
    final_prompt = context_text + f"user: {request.query}"

    async def stream_generator():
        full_response = []
        try:
            async for token in model.generate_stream(final_prompt):
                full_response.append(token)
                yield token

            # Post-stream: update metrics and memory
            complete_text = "".join(full_response)
            tokens = len(complete_text.split())

            quality_score = quality_evaluator.score_response(request.query, complete_text)
            router.metrics.update_quality(selected_tier, quality_score)

            cost = estimate_cost(tokens)
            router.metrics.update_cost(selected_tier, cost)

            semantic_cache.add(request.query, selected_tier, complete_text)
            memory.add_turn(request.session_id, "user", request.query)
            memory.add_turn(request.session_id, "assistant", complete_text)

            retrainer.on_query()
        finally:
            load_controller.decrement_requests()

    return StreamingResponse(stream_generator(), media_type="text/plain")