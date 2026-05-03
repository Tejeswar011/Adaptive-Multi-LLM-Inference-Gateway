from core.constants import DEFAULT_COST_PER_TOKEN


def estimate_cost(tokens: int) -> float:
    return tokens * DEFAULT_COST_PER_TOKEN