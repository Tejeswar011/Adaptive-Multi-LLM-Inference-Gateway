import subprocess
import shutil


class GPUMonitor:
    """Detects GPU availability, memory usage, and provides routing hints.

    Uses nvidia-smi when available, otherwise falls back to CPU-only mode.
    """

    def __init__(self):
        self.gpu_available = self._detect_gpu()
        self.gpu_info = {}

        if self.gpu_available:
            self.gpu_info = self._query_gpu()
            print(f"GPU detected: {self.gpu_info.get('name', 'unknown')}")
        else:
            print("No GPU detected, running in CPU-only mode")

    def _detect_gpu(self):
        return shutil.which("nvidia-smi") is not None

    def _query_gpu(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return {}

            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                return {
                    "name": parts[0],
                    "memory_total_mb": int(parts[1]),
                    "memory_used_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "gpu_utilization": int(parts[4]),
                }
        except Exception as e:
            print(f"GPU query error: {e}")

        return {}

    def refresh(self):
        if self.gpu_available:
            self.gpu_info = self._query_gpu()

    def get_memory_pressure(self):
        """Returns memory pressure as 0.0 (free) to 1.0 (full)."""
        if not self.gpu_available or not self.gpu_info:
            return 1.0  # No GPU = max pressure (forces CPU-friendly routing)

        total = self.gpu_info.get("memory_total_mb", 1)
        used = self.gpu_info.get("memory_used_mb", 0)
        return min(used / max(total, 1), 1.0)

    def get_gpu_utilization(self):
        """Returns GPU utilization as 0.0 to 1.0."""
        if not self.gpu_available or not self.gpu_info:
            return 1.0

        return self.gpu_info.get("gpu_utilization", 100) / 100.0

    def can_run_tier(self, tier):
        """Check if GPU has enough memory for a given tier.

        Approximate VRAM requirements:
        - tiny (TinyLlama 1.1B): ~1.5 GB
        - small (Phi-3 3.8B): ~4 GB
        - medium (Mistral 7B): ~6 GB
        """
        if not self.gpu_available:
            return True  # CPU mode, all tiers run on CPU

        vram_needed = {"tiny": 1500, "small": 4000, "medium": 6000}
        free = self.gpu_info.get("memory_free_mb", 0)

        return free >= vram_needed.get(tier, 0)

    def get_routing_hint(self):
        """Suggests max tier based on GPU state.

        Returns the highest tier the GPU can comfortably handle.
        If no GPU, returns 'medium' (all tiers run on CPU via Ollama).
        """
        if not self.gpu_available:
            return "medium"  # CPU mode, Ollama handles everything

        self.refresh()

        memory_pressure = self.get_memory_pressure()
        utilization = self.get_gpu_utilization()

        # Under heavy GPU load, suggest lighter models
        if memory_pressure > 0.85 or utilization > 0.9:
            return "tiny"
        elif memory_pressure > 0.6 or utilization > 0.7:
            return "small"
        else:
            return "medium"

    def get_status(self):
        return {
            "gpu_available": self.gpu_available,
            "gpu_info": self.gpu_info,
            "memory_pressure": round(self.get_memory_pressure(), 4),
            "gpu_utilization": round(self.get_gpu_utilization(), 4),
            "routing_hint": self.get_routing_hint(),
        }
