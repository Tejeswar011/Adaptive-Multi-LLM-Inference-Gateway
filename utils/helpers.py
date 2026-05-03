import time


def timed(func):
    """Decorator to measure execution time of sync functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


def safe_divide(a, b, default=0.0):
    """Division that returns default instead of raising on zero."""
    return a / b if b != 0 else default
