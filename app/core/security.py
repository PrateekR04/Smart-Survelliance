"""
Security utilities for authentication, authorization, and rate limiting.

Provides Basic Auth for guard app, API key authentication for services,
and JWT token management.
"""

import hashlib
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
basic_auth = HTTPBasic()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class TokenData:
    """JWT token payload data."""
    
    username: str
    exp: datetime


@dataclass
class User:
    """Authenticated user representation."""
    
    username: str
    is_active: bool = True
    is_guard: bool = False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hash.
    
    Args:
        plain_password: The password to verify.
        hashed_password: The bcrypt hashed password.
    
    Returns:
        bool: True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash.
    
    Returns:
        str: Bcrypt hashed password.
    """
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Payload data to encode in the token.
        expires_delta: Optional custom expiration time.
    
    Returns:
        str: Encoded JWT token.
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> TokenData | None:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: The JWT token to decode.
    
    Returns:
        TokenData: Decoded token data, or None if invalid.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        username: str = payload.get("sub")
        exp: int = payload.get("exp")
        
        if username is None:
            return None
        
        return TokenData(
            username=username,
            exp=datetime.fromtimestamp(exp, tz=timezone.utc),
        )
    except JWTError:
        return None


async def verify_basic_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(basic_auth)],
) -> User:
    """
    Verify HTTP Basic Authentication credentials.
    
    This is a placeholder implementation. In production, credentials
    should be verified against a database.
    
    Args:
        credentials: HTTP Basic Auth credentials.
    
    Returns:
        User: Authenticated user.
    
    Raises:
        HTTPException: If credentials are invalid.
    """
    # TODO: Replace with actual database lookup
    # For now, use environment-based guard credentials
    settings = get_settings()
    
    # Simple constant-time comparison for demo
    # In production, look up user in database and verify hashed password
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"),
        b"guard",
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"),
        settings.api_key.encode("utf8"),
    )
    
    if not (is_correct_username and is_correct_password):
        logger.warning(
            "auth_failed",
            username=credentials.username,
            reason="invalid_credentials",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return User(username=credentials.username, is_guard=True)


async def verify_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    """
    Verify API key for service-to-service authentication.
    
    Args:
        api_key: API key from X-API-Key header.
    
    Raises:
        HTTPException: If API key is missing or invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    
    settings = get_settings()
    if not secrets.compare_digest(api_key, settings.api_key):
        logger.warning("api_key_invalid", reason="key_mismatch")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@dataclass
class RateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.
    
    For production, consider using Redis-based rate limiting.
    
    Attributes:
        requests_per_window: Maximum requests allowed per window.
        window_seconds: Size of the sliding window in seconds.
    """
    
    requests_per_window: int
    window_seconds: int
    _requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key.
        
        Args:
            key: Unique identifier for the client (IP, user ID, etc.).
        
        Returns:
            bool: True if request is allowed, False if rate limit exceeded.
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old requests outside the window
        self._requests[key] = [
            req_time for req_time in self._requests[key]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self._requests[key]) < self.requests_per_window:
            self._requests[key].append(now)
            return True
        
        return False
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests for the given key.
        
        Args:
            key: Unique identifier for the client.
        
        Returns:
            int: Number of remaining requests in current window.
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        current_requests = [
            req_time for req_time in self._requests[key]
            if req_time > window_start
        ]
        
        return max(0, self.requests_per_window - len(current_requests))


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Returns:
        RateLimiter: Configured rate limiter.
    """
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(
            requests_per_window=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    return _rate_limiter


async def check_rate_limit(request: Request) -> None:
    """
    Rate limiting dependency for FastAPI routes.
    
    Args:
        request: The incoming HTTP request.
    
    Raises:
        HTTPException: If rate limit is exceeded.
    """
    rate_limiter = get_rate_limiter()
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(
            "rate_limit_exceeded",
            client_ip=client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                "Retry-After": str(rate_limiter.window_seconds),
                "X-RateLimit-Remaining": "0",
            },
        )


def compute_image_hash(image_bytes: bytes) -> str:
    """
    Compute SHA-256 hash of image bytes for idempotency.
    
    Args:
        image_bytes: Raw image data.
    
    Returns:
        str: Hexadecimal hash string.
    """
    return hashlib.sha256(image_bytes).hexdigest()
