"""
Core configuration module for Smart Parking Access Control System.

Uses Pydantic Settings for environment-based configuration with validation.
All secrets are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    Sensitive values have no defaults and must be explicitly set.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Database - can use either DATABASE_URL or individual params
    database_url: str | None = Field(
        default=None,
        description="MySQL connection string with aiomysql driver",
        examples=["mysql+aiomysql://user:pass@localhost:3306/parking_db"],
    )
    db_host: str = Field(default="127.0.0.1", description="Database host")
    db_port: int = Field(default=3306, description="Database port")
    db_user: str = Field(default="root", description="Database username")
    db_password: str = Field(default="", description="Database password")
    db_name: str = Field(default="parking_db", description="Database name")
    
    # ML Model
    ml_model_path: str = Field(
        default="./models/plate_detector.pt",
        description="Path to pre-trained plate detection model weights",
    )
    ocr_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence to accept result",
    )
    detector_confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum detector confidence to accept bounding box",
    )
    
    # Security
    secret_key: str = Field(
        ...,
        min_length=16,
        description="Secret key for JWT token signing",
    )
    api_key: str = Field(
        ...,
        min_length=8,
        description="API key for service authentication",
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        description="Token expiration time in minutes",
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Rate limit window duration in seconds",
    )
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application log level",
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )
    
    # Storage
    image_storage_path: str = Field(
        default="./storage/images",
        description="Directory for storing captured plate images",
    )
    
    # Server
    host: str = Field(
        default="0.0.0.0",
        description="Server bind address",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    
    @model_validator(mode="after")
    def build_database_url_if_needed(self) -> "Settings":
        """Build database URL from individual params if not provided."""
        if self.database_url is None or self.database_url == "":
            # URL-encode the password to handle special characters like @
            encoded_password = quote_plus(self.db_password)
            self.database_url = (
                f"mysql+aiomysql://{self.db_user}:{encoded_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return self
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str | None) -> str | None:
        """Ensure database URL uses async MySQL driver if provided."""
        if v is not None and v != "" and not v.startswith("mysql+aiomysql://"):
            raise ValueError(
                "Database URL must use mysql+aiomysql:// driver for async support"
            )
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Settings are loaded once and cached for the lifetime of the application.
    
    Returns:
        Settings: Application configuration instance.
    
    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()
