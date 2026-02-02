"""
Idempotency service for request deduplication.

Prevents duplicate alerts and log spam when same car
sends multiple frames in quick succession.
"""

import hashlib
import time
from dataclasses import dataclass, field


@dataclass
class IdempotencyService:
    """
    Service for deduplicating requests within a time window.
    
    Uses image hash + camera_id as composite key.
    Default time window is 5 seconds.
    
    Example:
        service = IdempotencyService(window_seconds=5)
        key = service.compute_key(image_bytes, "CAM001")
        if service.is_duplicate(key):
            return cached_response
        service.mark_seen(key, response)
    """
    
    window_seconds: int = 5
    _seen: dict[str, tuple[float, any]] = field(default_factory=dict)
    
    def compute_key(self, image_bytes: bytes, camera_id: str) -> str:
        """
        Compute idempotency key from image and camera.
        
        Uses SHA-256 hash of image bytes combined with camera ID.
        
        Args:
            image_bytes: Raw image data.
            camera_id: Camera identifier.
        
        Returns:
            str: Idempotency key.
        """
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        return f"{camera_id}:{image_hash}"
    
    def is_duplicate(self, key: str) -> bool:
        """
        Check if request is a duplicate within the time window.
        
        Args:
            key: Idempotency key from compute_key.
        
        Returns:
            bool: True if this is a duplicate request.
        """
        self._cleanup_expired()
        
        if key not in self._seen:
            return False
        
        seen_time, _ = self._seen[key]
        return (time.time() - seen_time) < self.window_seconds
    
    def get_cached_response(self, key: str) -> any:
        """
        Get cached response for a duplicate request.
        
        Args:
            key: Idempotency key.
        
        Returns:
            Cached response, or None if not found.
        """
        if key in self._seen:
            _, response = self._seen[key]
            return response
        return None
    
    def mark_seen(self, key: str, response: any = None) -> None:
        """
        Mark a request as seen with optional cached response.
        
        Args:
            key: Idempotency key.
            response: Optional response to cache.
        """
        self._seen[key] = (time.time(), response)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        expired_keys = [
            key for key, (seen_time, _) in self._seen.items()
            if seen_time < cutoff
        ]
        
        for key in expired_keys:
            del self._seen[key]


# Global idempotency service instance
_idempotency_service: IdempotencyService | None = None


def get_idempotency_service() -> IdempotencyService:
    """Get the global idempotency service instance."""
    global _idempotency_service
    if _idempotency_service is None:
        _idempotency_service = IdempotencyService()
    return _idempotency_service
