class ParetoEngine:

    def compute_pareto_front(self, candidates):
        """
        candidates: list of dicts
        Each dict must contain:
        {
            "tier": str,
            "quality": float (higher better),
            "latency": float (lower better),
            "cost": float (lower better)
        }
        """

        pareto_front = []

        for candidate in candidates:
            dominated = False

            for other in candidates:
                if other == candidate:
                    continue

                if (
                    other["quality"] >= candidate["quality"] and
                    other["latency"] <= candidate["latency"] and
                    other["cost"] <= candidate["cost"] and
                    (
                        other["quality"] > candidate["quality"] or
                        other["latency"] < candidate["latency"] or
                        other["cost"] < candidate["cost"]
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(candidate)

        return pareto_front