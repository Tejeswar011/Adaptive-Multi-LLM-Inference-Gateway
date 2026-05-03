class ModelMetrics:

    def __init__(self):
        self.metrics = {
            "tiny": {"latency": 12, "quality": 0.45, "cost": 1},
            "small": {"latency": 18, "quality": 0.65, "cost": 3},
            "medium": {"latency": 30, "quality": 0.85, "cost": 6},}

    def get_metrics(self, tier):
        return self.metrics.get(tier)

    def update_latency(self, tier, latency):
        current = self.metrics[tier]["latency"]
        self.metrics[tier]["latency"] = (current + latency) / 2

    def update_cost(self, tier, cost):
        current = self.metrics[tier]["cost"]
        self.metrics[tier]["cost"] = (current + cost) / 2
    
    def update_quality(self, tier, quality):
        current = self.metrics[tier]["quality"]
        self.metrics[tier]["quality"] = (current + quality) / 2 