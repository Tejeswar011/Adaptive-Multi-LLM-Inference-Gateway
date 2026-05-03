"""
Route definitions are in api/main.py.

Endpoints:
    GET  /             - Health check
    POST /generate     - Non-streaming inference
    POST /generate/stream - Streaming inference
    GET  /metrics      - Prometheus metrics
    GET  /gpu          - GPU status
    GET  /cache/stats  - Cache hit/miss stats
    GET  /bandit/stats - UCB bandit stats
"""
