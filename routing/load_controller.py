import psutil
import threading


class LoadController:

    def __init__(self):
        self.active_requests = 0
        self.lock = threading.Lock()

    def increment_requests(self):
        with self.lock:
            self.active_requests += 1

    def decrement_requests(self):
        with self.lock:
            self.active_requests -= 1

    def get_active_requests(self):
        return self.active_requests

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=0.1)

    def compute_load_score(self):
        active = self.get_active_requests()
        cpu = self.get_cpu_usage()

        active_score = min(active / 10, 1.0)
        cpu_score = cpu / 100

        return (active_score + cpu_score) / 2