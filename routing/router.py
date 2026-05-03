import os
import joblib
import numpy as np
from routing.pareto_engine import ParetoEngine
from routing.bandit import UCBBandit
from routing.gpu_monitor import GPUMonitor
from models.model_metrics import ModelMetrics

TRAINED_MODEL_PATH = "routing/trained_router.pkl"
TIERS = ["tiny", "small", "medium"]
TIER_RANK = {"tiny": 0, "small": 1, "medium": 2}


class Router:

    def __init__(self):
        self.pareto_engine = ParetoEngine()
        self.metrics = ModelMetrics()
        self.ml_model = None
        self.bandit = UCBBandit(exploration_weight=1.5)
        self.gpu_monitor = GPUMonitor()
        self._load_ml_model()

    def _load_ml_model(self):
        if os.path.isfile(TRAINED_MODEL_PATH):
            try:
                self.ml_model = joblib.load(TRAINED_MODEL_PATH)
                print(f"ML router loaded from {TRAINED_MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load ML router: {e}")
                self.ml_model = None
        else:
            print("No trained router found, using heuristic fallback")

    def reload_ml_model(self):
        self._load_ml_model()

    def select_model(self, load_score: float, question_type: int = 3, word_count: int = 5, token_length: int = 30):

        # ---- UCB Bandit exploration ----
        bandit_tier, is_exploration = self.bandit.select()

        if is_exploration:
            tier = self._apply_gpu_cap(bandit_tier)
            print(f"BANDIT EXPLORATION: selected {tier}")
            return tier

        # ---- Try ML router first ----
        if self.ml_model is not None:
            tier = self._ml_select(token_length, word_count, question_type, load_score)
            if tier is not None:
                tier = self._apply_gpu_cap(tier)
                print(f"ML router selected: {tier}")
                return tier

        # ---- Heuristic fallback ----
        print("Using heuristic fallback")
        tier = self._heuristic_select(load_score, question_type, word_count)
        return self._apply_gpu_cap(tier)

    def update_bandit(self, tier, quality_score):
        self.bandit.update(tier, quality_score)

    def _apply_gpu_cap(self, tier):
        """Downgrade tier if GPU can't handle it."""
        max_tier = self.gpu_monitor.get_routing_hint()

        if TIER_RANK[tier] > TIER_RANK[max_tier]:
            print(f"GPU cap: {tier} -> {max_tier} (memory/utilization pressure)")
            return max_tier

        return tier

    def _ml_select(self, token_length, word_count, question_type, load_score):
        try:
            features = np.array([[token_length, word_count, question_type, load_score]])
            prediction = self.ml_model.predict(features)[0]

            if prediction in TIERS:
                return prediction

            return None
        except Exception as e:
            print(f"ML router error: {e}")
            return None

    def _heuristic_select(self, load_score: float, question_type: int, word_count: int):

        # ---- Complexity-based quality adjustment ----
        quality_adj = {
            "tiny": 0.0,
            "small": 0.0,
            "medium": 0.0,
        }

        if question_type == 1:  # reasoning (why/how)
            quality_adj["medium"] = 0.3
            quality_adj["small"] = 0.15
        elif question_type == 2:  # descriptive (explain/describe)
            quality_adj["medium"] = 0.25
            quality_adj["small"] = 0.1
        elif question_type == 0 and word_count > 8:  # long factual
            quality_adj["medium"] = 0.15
            quality_adj["small"] = 0.05
        elif question_type == 0 and word_count <= 8:  # short factual
            quality_adj["tiny"] = 0.2
            quality_adj["medium"] = -0.2
        elif question_type == 3 and word_count <= 6:  # short other
            quality_adj["tiny"] = 0.15
            quality_adj["medium"] = -0.15

        candidates = []

        for tier in TIERS:
            m = self.metrics.get_metrics(tier)

            candidates.append({
                "tier": tier,
                "quality": m["quality"] + quality_adj[tier],
                "latency": m["latency"],
                "cost": m["cost"]
            })

        pareto_front = self.pareto_engine.compute_pareto_front(candidates)

        # ---- Load-sensitive penalty scaling (softened for CPU) ----
        alpha = 0.15 + (load_score ** 2) * 0.8
        beta = 0.1 + (load_score ** 2) * 0.5

        # ---- Normalize latency and cost ----
        max_latency = max(m["latency"] for m in candidates)
        max_cost = max(m["cost"] for m in candidates)

        if max_latency == 0:
            max_latency = 1

        if max_cost == 0:
            max_cost = 1

        best = None
        best_utility = float("-inf")

        utilities_dict = {}

        for model in pareto_front:

            normalized_latency = model["latency"] / max_latency
            normalized_cost = model["cost"] / max_cost

            utility = (
                model["quality"]
                - alpha * normalized_latency
                - beta * normalized_cost
            )

            utilities_dict[model["tier"]] = utility

            if utility > best_utility:
                best_utility = utility
                best = model

        print("Alpha:", alpha, "Beta:", beta)
        print("Utilities:", utilities_dict)
        print("Selected tier:", best["tier"])

        # ---- Safety latency guard ----
        if best["latency"] > 80:
            print("Latency too high, downgrading tier")

            if best["tier"] == "medium":
                return "small"

            elif best["tier"] == "small":
                return "tiny"

        return best["tier"]
