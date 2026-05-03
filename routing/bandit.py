import math
import random

TIERS = ["tiny", "small", "medium"]


class UCBBandit:
    """Upper Confidence Bound bandit for tier exploration.

    Balances exploitation (pick the best known tier) with exploration
    (try under-explored tiers) using the UCB1 formula:

        UCB = avg_reward + c * sqrt(ln(total) / count)

    Reward is the quality score from the quality feedback loop.
    """

    def __init__(self, exploration_weight=1.5):
        self.c = exploration_weight
        self.counts = {tier: 0 for tier in TIERS}
        self.rewards = {tier: 0.0 for tier in TIERS}
        self.total = 0

    def select(self):
        self.total += 1

        # Force exploration: try each tier at least 3 times
        for tier in TIERS:
            if self.counts[tier] < 3:
                return tier, True  # (tier, is_exploration)

        # UCB1 selection
        best_tier = None
        best_ucb = float("-inf")

        for tier in TIERS:
            avg_reward = self.rewards[tier] / self.counts[tier]
            exploration_bonus = self.c * math.sqrt(
                math.log(self.total) / self.counts[tier]
            )
            ucb = avg_reward + exploration_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_tier = tier

        return best_tier, False

    def update(self, tier, reward):
        self.counts[tier] += 1
        self.rewards[tier] += reward

    def get_stats(self):
        stats = {}
        for tier in TIERS:
            count = self.counts[tier]
            avg = self.rewards[tier] / count if count > 0 else 0
            stats[tier] = {"count": count, "avg_reward": round(avg, 4)}
        stats["total"] = self.total
        return stats


class ThompsonSampler:
    """Thompson Sampling bandit for tier exploration.

    Maintains a Beta distribution per tier (alpha=successes, beta=failures).
    Samples from each distribution and picks the tier with highest sample.
    Naturally balances exploration/exploitation.
    """

    def __init__(self):
        # Beta(1,1) = uniform prior
        self.alpha = {tier: 1.0 for tier in TIERS}
        self.beta = {tier: 1.0 for tier in TIERS}
        self.total = 0

    def select(self):
        self.total += 1

        samples = {}
        for tier in TIERS:
            samples[tier] = random.betavariate(self.alpha[tier], self.beta[tier])

        best_tier = max(samples, key=samples.get)
        is_exploration = self.total <= len(TIERS) * 3

        return best_tier, is_exploration

    def update(self, tier, reward):
        # reward is 0-1 quality score; treat as probability of success
        self.alpha[tier] += reward
        self.beta[tier] += (1.0 - reward)

    def get_stats(self):
        stats = {}
        for tier in TIERS:
            mean = self.alpha[tier] / (self.alpha[tier] + self.beta[tier])
            stats[tier] = {
                "alpha": round(self.alpha[tier], 2),
                "beta": round(self.beta[tier], 2),
                "mean": round(mean, 4),
            }
        stats["total"] = self.total
        return stats
