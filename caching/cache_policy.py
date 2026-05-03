import time
import zlib
from collections import OrderedDict


class LRUCachePolicy:

    def __init__(self, max_size=500, ttl_seconds=3600, enable_compression=True):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.store = OrderedDict()
        self.enable_compression = enable_compression
        self.cost_map = {}  # key -> tier cost for cost-aware eviction

    def get(self, key):
        if key not in self.store:
            return None

        value, timestamp = self.store[key]

        # TTL check
        if time.time() - timestamp > self.ttl:
            del self.store[key]
            self.cost_map.pop(key, None)
            return None

        # Move to end (LRU refresh)
        self.store.move_to_end(key)

        if self.enable_compression and isinstance(value, bytes):
            return zlib.decompress(value).decode("utf-8")

        return value

    def set(self, key, value, tier_cost=1):
        if key in self.store:
            self.store.move_to_end(key)
        elif len(self.store) >= self.max_size:
            self._evict()

        if self.enable_compression and isinstance(value, str):
            value = zlib.compress(value.encode("utf-8"))

        self.store[key] = (value, time.time())
        self.cost_map[key] = tier_cost

    def _evict(self):
        # Cost-aware eviction: evict the cheapest (lowest tier cost) entry first
        # Among entries with same cost, evict the oldest (LRU)
        if not self.store:
            return

        min_cost = min(self.cost_map.get(k, 1) for k in self.store)
        for key in self.store:
            if self.cost_map.get(key, 1) == min_cost:
                del self.store[key]
                self.cost_map.pop(key, None)
                return

        # Fallback: evict oldest
        evicted_key, _ = self.store.popitem(last=False)
        self.cost_map.pop(evicted_key, None)

    def size(self):
        return len(self.store)
