from __future__ import annotations

from .rate_limiter import FailureRateLimiter

failure_rate_limiter = FailureRateLimiter()
# Feed-token keys come from client input, so they get their own storage: a flood
# of fresh token ids must not evict login backoff entries.
feed_token_rate_limiter = FailureRateLimiter()
