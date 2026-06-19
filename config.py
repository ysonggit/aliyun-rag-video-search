"""
Centralized configuration loaded from environment variables.
All secrets and region settings are read from env / .env file — never hardcoded.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

import dashscope

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your values."
        )
    return val


@dataclass(frozen=True)
class DashScopeConfig:
    api_key: str = field(default_factory=lambda: _require("DASHSCOPE_API_KEY"))
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/api/v1",
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "tongyi-embedding-vision-plus")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "qwen-vl-max")
    )


@dataclass(frozen=True)
class DashVectorConfig:
    api_key: str = field(default_factory=lambda: _require("DASHVECTOR_API_KEY"))
    endpoint: str = field(default_factory=lambda: _require("DASHVECTOR_ENDPOINT"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("DV_COLLECTION_NAME", "scene_frames")
    )
    dimension: int = field(
        default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "1152"))
    )


@dataclass(frozen=True)
class OSSConfig:
    access_key_id: str = field(default_factory=lambda: os.getenv("OSS_ACCESS_KEY_ID", ""))
    access_key_secret: str = field(default_factory=lambda: os.getenv("OSS_ACCESS_KEY_SECRET", ""))
    endpoint: str = field(default_factory=lambda: os.getenv("OSS_ENDPOINT", ""))
    bucket_name: str = field(default_factory=lambda: os.getenv("OSS_BUCKET_NAME", ""))
    frame_prefix: str = field(
        default_factory=lambda: os.getenv("OSS_FRAME_PREFIX", "video-frames")
    )


# Singleton instances
dashscope_config = DashScopeConfig()
dashvector_config = DashVectorConfig()
oss_config = OSSConfig()

# Set DashScope SDK to use the configured base URL (required for Singapore/intl)
dashscope.base_http_api_url = dashscope_config.base_url
